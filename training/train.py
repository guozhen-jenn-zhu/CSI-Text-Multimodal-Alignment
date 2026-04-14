"""Training loop for CSI-Text multimodal alignment with DDP + AMP."""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data import build_dataloaders
from models.alignment import build_alignment_model
from models.encoders import ProjectionHead, build_csi_encoder, build_text_encoder
from training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _merge_configs(experiment: dict) -> dict:
    """Load and merge dataset / method YAML files referenced by the experiment config."""
    base_dir = Path("configs")

    dataset_path = base_dir / "dataset" / f"{experiment['dataset']}.yaml"
    if dataset_path.exists():
        with open(dataset_path) as f:
            experiment["dataset_config"] = yaml.safe_load(f)

    method_path = base_dir / "method" / f"{experiment['method']}.yaml"
    if method_path.exists():
        with open(method_path) as f:
            experiment["method_config"] = yaml.safe_load(f)

    return experiment


def _apply_overrides(config: dict, overrides: list[str]) -> None:
    """Apply ``key.subkey=value`` overrides to a nested config dict."""
    for kv in overrides:
        key, value = kv.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        d[keys[-1]] = parsed


def _setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training if environment variables are set.

    Returns (local_rank, rank, world_size). All zero when not distributed.
    """
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return local_rank, rank, world_size
    return 0, 0, 1


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    """End-to-end trainer for CSI-Text alignment with DDP + AMP."""

    def __init__(
        self,
        config: dict,
        local_rank: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.distributed = world_size > 1

        training_cfg = config["training"]
        _set_seed(training_cfg.get("seed", 42) + rank)

        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.max_epochs = training_cfg["max_epochs"]
        self.patience = training_cfg.get("early_stopping_patience", 15)
        self.grad_clip = training_cfg.get("grad_clip", 1.0)
        self.log_every = config.get("logging", {}).get("log_every_n_steps", 50)
        self.accumulation_steps = training_cfg.get("accumulation_steps", 1)

        self.use_amp = training_cfg.get("amp", True) and torch.cuda.is_available()
        amp_dtype_str = training_cfg.get("amp_dtype", "bfloat16")
        if amp_dtype_str == "bfloat16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)

        if self.is_main:
            self._setup_logging()
        self._build_components()
        self._build_dataloaders()
        self._build_optimizer()

        self.start_epoch = 0
        self.best_val_loss = float("inf")

        self.s3_syncer = None
        s3_cfg = config.get("s3", {})
        if s3_cfg.get("enabled", False) and self.is_main:
            try:
                from utils.s3_sync import S3Syncer
                self.s3_syncer = S3Syncer(
                    bucket=s3_cfg["bucket"],
                    prefix=s3_cfg.get("prefix", "Multimodal_CSI_Text"),
                )
            except Exception as e:
                logger.warning("S3 sync unavailable: %s", e)

        if self.is_main:
            logger.info(
                "Trainer ready on %s | DDP=%s world_size=%d | AMP=%s dtype=%s | accum=%d",
                self.device, self.distributed, self.world_size,
                self.use_amp, self.amp_dtype, self.accumulation_steps,
            )

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir = (
            checkpoint_dir / self.config["dataset"] / self.config["method"]
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.checkpoint_dir / "train.log"
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.INFO)

        self.use_wandb = log_cfg.get("use_wandb", False)
        self.wandb_run = None
        if self.use_wandb:
            import wandb
            self.wandb_run = wandb.init(
                project=log_cfg.get("project_name", "csi-text-alignment"),
                config=self.config,
                name=f"{self.config['dataset']}_{self.config['method']}",
            )

    def _build_components(self) -> None:
        ds_cfg = self.config.get("dataset_config", {})
        preproc = ds_cfg.get("preprocessing", {})
        proj_cfg = self.config.get("projection", {})
        method_cfg = self.config.get("method_config", {})
        loss_cfg = method_cfg.get("loss", {})

        input_channels = preproc.get("input_channels", 1)
        input_subcarriers = preproc.get("target_subcarriers", 64)
        input_timesteps = preproc.get("target_time_steps", 500)
        embed_dim = proj_cfg.get("dim", 512)

        enc_cfg = self.config.get("csi_encoder", {})
        self.csi_encoder = build_csi_encoder(
            name=enc_cfg.get("name", "resnet18_1d"),
            input_channels=input_channels,
            input_subcarriers=input_subcarriers,
            input_timesteps=input_timesteps,
            embed_dim=embed_dim,
        )

        txt_cfg = self.config.get("text_encoder", {})
        self.text_encoder = build_text_encoder(name=txt_cfg.get("name", "clip_vit_b32"))
        text_embed_dim = self.text_encoder.embed_dim

        self.csi_projection = ProjectionHead(
            input_dim=embed_dim,
            output_dim=proj_cfg.get("dim", 512),
            hidden_dim=proj_cfg.get("hidden_dim", 1024),
            num_layers=proj_cfg.get("num_layers", 2),
        )

        method_name = self.config["method"]
        if method_name == "language_bind":
            self.text_projection = None
        else:
            self.text_projection = ProjectionHead(
                input_dim=text_embed_dim,
                output_dim=proj_cfg.get("dim", 512),
                hidden_dim=proj_cfg.get("hidden_dim", 1024),
                num_layers=proj_cfg.get("num_layers", 2),
            )

        model_kwargs = self._extract_model_kwargs(method_name, loss_cfg, embed_dim)
        self.model = build_alignment_model(
            method_name=method_name,
            csi_encoder=self.csi_encoder,
            csi_projection=self.csi_projection,
            text_projection=self.text_projection,
            **model_kwargs,
        )
        self.model.to(self.device)

        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.is_main:
            logger.info("Model: %s | trainable params: %s", method_name, f"{param_count:,}")

    @staticmethod
    def _extract_model_kwargs(
        method_name: str, loss_cfg: dict, embed_dim: int
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        temp = loss_cfg.get("temperature")
        if temp is not None:
            kwargs["temperature"] = temp

        if method_name == "siglip":
            kwargs["bias"] = loss_cfg.get("bias", -10.0)
        elif method_name == "centrobind":
            kwargs["embed_dim"] = embed_dim
            kwargs["num_centroids"] = loss_cfg.get("num_centroids", 16)
            kwargs["momentum"] = loss_cfg.get("centroid_momentum", 0.999)
        elif method_name == "barlow_twins":
            kwargs["lambda_coeff"] = loss_cfg.get("lambda_coeff", 0.005)
        elif method_name == "masked_predict":
            kwargs["embed_dim"] = embed_dim
            kwargs["mask_ratio"] = loss_cfg.get("mask_ratio", 0.75)
            kwargs["reconstruction_weight"] = loss_cfg.get("reconstruction_weight", 1.0)
            kwargs["alignment_weight"] = loss_cfg.get("alignment_weight", 0.5)

        return kwargs

    def _resolve_task_root(self) -> str:
        """Resolve the task_root path from dataset + task config."""
        ds_cfg = self.config.get("dataset_config", {})
        data_root = ds_cfg.get("data_root", "/data/fm_downstream")
        task_name = self.config.get("task", "har")

        tasks = ds_cfg.get("tasks", {})
        if isinstance(tasks, dict) and task_name in tasks:
            folder = tasks[task_name].get("folder", task_name)
        else:
            folder = task_name

        return str(Path(data_root) / folder)

    def _resolve_h5_key(self) -> str:
        """Resolve the H5 key for the current dataset/task."""
        ds_cfg = self.config.get("dataset_config", {})
        task_name = self.config.get("task", "har")

        tasks = ds_cfg.get("tasks", {})
        if isinstance(tasks, dict) and task_name in tasks:
            return tasks[task_name].get("h5_key", "CSI_amps")

        return ds_cfg.get("h5_key", "CSI_amps")

    def _build_dataloaders(self) -> None:
        ds_cfg = self.config.get("dataset_config", {})
        preproc = ds_cfg.get("preprocessing", {})
        training_cfg = self.config["training"]

        task_root = self.config.get("task_root") or self._resolve_task_root()
        h5_key = self._resolve_h5_key()

        loader_config: dict[str, Any] = {
            "task_root": task_root,
            "h5_key": h5_key,
            "registry_path": self.config.get("registry_path"),
            "text_embeddings_path": self.config.get("text_embeddings_path"),
            "target_subcarriers": preproc.get("target_subcarriers", 64),
            "target_time_steps": preproc.get("target_time_steps", 500),
            "batch_size": training_cfg.get("batch_size", 128),
            "num_workers": training_cfg.get("num_workers", 4),
            "prefetch_factor": training_cfg.get("prefetch_factor", 2),
            "persistent_workers": training_cfg.get("persistent_workers", True),
            "preload": training_cfg.get("preload", False),
        }

        loaders = build_dataloaders(
            loader_config,
            distributed=self.distributed,
            world_size=self.world_size,
            rank=self.rank,
        )
        self.train_loader: DataLoader = loaders["train"]
        self.val_loader: DataLoader = loaders["val"]
        self.test_loader: DataLoader = loaders["test"]

    def _build_optimizer(self) -> None:
        training_cfg = self.config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_cfg.get("learning_rate", 1e-3),
            weight_decay=training_cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = build_scheduler(self.optimizer, training_cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        if self.distributed and hasattr(self.train_loader, "sampler"):
            sampler = self.train_loader.sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader):
            csi = batch["csi"].to(self.device, non_blocking=True)
            text_emb = batch["text_embeddings"]
            if text_emb is None:
                raise RuntimeError(
                    "text_embeddings are None -- precomputed embeddings required. "
                    "Set text_embeddings_path in config."
                )
            text_emb = text_emb.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(csi, text_emb)
                loss = output["loss"] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1

            global_step = epoch * len(self.train_loader) + step
            if self.is_main and self.log_every > 0 and global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch %d  step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, step, len(self.train_loader), loss.item() * self.accumulation_steps, lr,
                )

        avg_loss = total_loss / max(num_batches, 1)
        return {"train_loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            csi = batch["csi"].to(self.device, non_blocking=True)
            text_emb = batch["text_embeddings"]
            if text_emb is None:
                continue
            text_emb = text_emb.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(csi, text_emb)
            total_loss += output["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def train(self) -> dict[str, float]:
        """Full training loop with early stopping."""
        patience_counter = 0
        best_metrics: dict[str, float] = {}

        if self.is_main:
            logger.info(
                "Starting training: %d epochs, patience %d, accum %d",
                self.max_epochs, self.patience, self.accumulation_steps,
            )

        for epoch in range(self.start_epoch, self.max_epochs):
            t0 = time.perf_counter()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics["val_loss"])

            elapsed = time.perf_counter() - t0

            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  lr=%.2e  time=%.1fs",
                    epoch + 1, self.max_epochs,
                    train_metrics["train_loss"], val_metrics["val_loss"],
                    lr, elapsed,
                )

                all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": lr}
                if self.use_wandb and self.wandb_run is not None:
                    import wandb
                    wandb.log(all_metrics, step=epoch)

                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                    best_metrics = all_metrics.copy()
                    self.save_checkpoint("best.pt", epoch, val_metrics["val_loss"])
                    logger.info("  -> new best val_loss=%.4f -- checkpoint saved", self.best_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            "Early stopping at epoch %d (patience %d exhausted)",
                            epoch + 1, self.patience,
                        )
                        break

                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}.pt", epoch, val_metrics["val_loss"])
            else:
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            if self.distributed:
                dist.barrier()

        if self.is_main:
            self.load_checkpoint("best.pt")
            logger.info("Training complete. Best val_loss=%.4f", self.best_val_loss)

            if self.use_wandb and self.wandb_run is not None:
                import wandb
                wandb.finish()

        return best_metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        if not self.is_main:
            return

        path = self.checkpoint_dir / filename
        model_state = (
            self.model.module.state_dict() if self.distributed else self.model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "val_loss": val_loss,
                "config": self.config,
            },
            path,
        )

        if self.s3_syncer is not None and filename == "best.pt":
            try:
                self.s3_syncer.upload_checkpoint(str(path))
            except Exception as e:
                logger.warning("S3 checkpoint upload failed: %s", e)

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        if not path.exists():
            logger.warning("Checkpoint %s not found -- skipping", path)
            return

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = self.model.module if self.distributed else self.model
        model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["best_val_loss"]
        if self.is_main:
            logger.info(
                "Loaded checkpoint %s (epoch %d, val_loss=%.4f)",
                path, ckpt["epoch"], ckpt["val_loss"],
            )


# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSI-Text alignment model")
    parser.add_argument(
        "--config", type=str, default="configs/experiment/default.yaml",
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values as key.subkey=value",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    args = parser.parse_args()

    local_rank, rank, world_size = _setup_distributed()

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config = _merge_configs(config)

    if args.override:
        _apply_overrides(config, args.override)

    trainer = Trainer(config, local_rank=local_rank, rank=rank, world_size=world_size)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
