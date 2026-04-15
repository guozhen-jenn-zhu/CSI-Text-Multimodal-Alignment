"""Multitask training loop for CSI-Text multimodal alignment.

Extends the single-task :class:`Trainer` to train on multiple tasks
simultaneously via :class:`MultitaskCSITextDataset`, with per-task
validation metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.multitask_dataset import build_multitask_dataloaders
from models.alignment import build_alignment_model
from models.encoders import ProjectionHead, build_csi_encoder, build_text_encoder
from training.scheduler import build_scheduler
from training.train import _apply_overrides, _set_seed, _setup_distributed, _cleanup_distributed

logger = logging.getLogger(__name__)


def _merge_multitask_configs(experiment: dict) -> dict:
    """Load method YAML and (optionally) dataset YAML for multitask runs."""
    base_dir = Path("configs")

    method_path = base_dir / "method" / f"{experiment['method']}.yaml"
    if method_path.exists() and "method_config" not in experiment:
        with open(method_path) as f:
            experiment["method_config"] = yaml.safe_load(f)

    dataset_name = experiment.get("dataset", "csibench")
    dataset_path = base_dir / "dataset" / f"{dataset_name}.yaml"
    if dataset_path.exists() and "dataset_config" not in experiment:
        with open(dataset_path) as f:
            experiment["dataset_config"] = yaml.safe_load(f)

    return experiment


class MultitaskTrainer:
    """Trainer for joint multi-task CSI-Text alignment with DDP + AMP.

    The key difference from the single-task ``Trainer`` is that dataloaders
    are built from a list of task configs (each with its own task_root,
    registry, and embeddings), and validation metrics are reported per-task.
    """

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
        self.max_epochs = int(training_cfg["max_epochs"])
        self.patience = int(training_cfg.get("early_stopping_patience", 15))
        self.grad_clip = float(training_cfg.get("grad_clip", 1.0))
        self.log_every = int(config.get("logging", {}).get("log_every_n_steps", 50))
        self.accumulation_steps = int(training_cfg.get("accumulation_steps", 1))

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
                "MultitaskTrainer ready on %s | DDP=%s world_size=%d | AMP=%s dtype=%s | tasks=%s",
                self.device, self.distributed, self.world_size,
                self.use_amp, self.amp_dtype,
                [tc["task_name"] for tc in config["multitask"]["tasks"]],
            )

    def _setup_logging(self) -> None:
        log_cfg = self.config.get("logging", {})
        checkpoint_dir = Path(log_cfg.get("checkpoint_dir", "checkpoints"))
        task = self.config.get("task", "multitask")
        self.checkpoint_dir = checkpoint_dir / f"{self.config['dataset']}-{task}-{self.config['method']}"
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
                name=f"{self.config['dataset']}_multitask_{self.config['method']}",
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
            dropout=float(enc_cfg.get("dropout", 0.0)),
        )

        txt_cfg = self.config.get("text_encoder", {})
        has_precomputed = all(
            tc.get("text_embeddings_path") is not None
            for tc in self.config["multitask"]["tasks"]
        )
        if has_precomputed:
            text_embed_dim = proj_cfg.get("dim", 512)
            if self.is_main:
                logger.info("Using precomputed text embeddings for all tasks")
        else:
            self.text_encoder = build_text_encoder(name=txt_cfg.get("name", "clip_vit_b32"))
            text_embed_dim = self.text_encoder.embed_dim

        proj_dropout = float(proj_cfg.get("dropout", 0.0))
        self.csi_projection = ProjectionHead(
            input_dim=embed_dim,
            output_dim=proj_cfg.get("dim", 512),
            hidden_dim=proj_cfg.get("hidden_dim", 1024),
            num_layers=proj_cfg.get("num_layers", 2),
            dropout=proj_dropout,
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
                dropout=proj_dropout,
            )

        from training.train import Trainer
        model_kwargs = Trainer._extract_model_kwargs(method_name, loss_cfg, embed_dim)
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

    def _build_dataloaders(self) -> None:
        ds_cfg = self.config.get("dataset_config", {})
        preproc = ds_cfg.get("preprocessing", {})
        training_cfg = self.config["training"]
        mt_cfg = self.config["multitask"]

        loaders = build_multitask_dataloaders(
            task_configs=mt_cfg["tasks"],
            target_subcarriers=preproc.get("target_subcarriers", 64),
            target_time_steps=preproc.get("target_time_steps", 500),
            batch_size=training_cfg.get("batch_size", 128),
            num_workers=training_cfg.get("num_workers", 4),
            prefetch_factor=training_cfg.get("prefetch_factor", 2),
            persistent_workers=training_cfg.get("persistent_workers", True),
            preload=training_cfg.get("preload", False),
            sampling=mt_cfg.get("sampling", "proportional"),
            augmentation=self.config.get("augmentation", {}),
            distributed=self.distributed,
            world_size=self.world_size,
            rank=self.rank,
        )
        self.train_loader: DataLoader = loaders["train"]
        self.val_loader: DataLoader = loaders["val"]
        self.test_loader: DataLoader = loaders["test"]

        self.task_names: list[str] = self.train_loader.dataset.task_names

    def _build_optimizer(self) -> None:
        training_cfg = self.config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
        )
        self.scheduler = build_scheduler(self.optimizer, training_cfg)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        per_task_loss: dict[str, float] = defaultdict(float)
        per_task_count: dict[str, int] = defaultdict(int)

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
                    "text_embeddings are None -- precomputed embeddings required for all tasks."
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

            batch_loss = loss.item() * self.accumulation_steps
            total_loss += batch_loss
            num_batches += 1

            task_ids = batch["task_ids"]
            for tid in task_ids.unique().tolist():
                tname = self.task_names[tid]
                per_task_loss[tname] += batch_loss
                per_task_count[tname] += 1

            global_step = epoch * len(self.train_loader) + step
            if self.is_main and self.log_every > 0 and global_step % self.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "epoch %d  step %d/%d  loss=%.4f  lr=%.2e",
                    epoch, step, len(self.train_loader), batch_loss, lr,
                )

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"train_loss": avg_loss}
        for tname, tloss in per_task_loss.items():
            metrics[f"train_loss/{tname}"] = tloss / max(per_task_count[tname], 1)
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        per_task_loss: dict[str, float] = defaultdict(float)
        per_task_count: dict[str, int] = defaultdict(int)

        for batch in self.val_loader:
            csi = batch["csi"].to(self.device, non_blocking=True)
            text_emb = batch["text_embeddings"]
            if text_emb is None:
                continue
            text_emb = text_emb.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(csi, text_emb)

            batch_loss = output["loss"].item()
            total_loss += batch_loss
            num_batches += 1

            task_ids = batch["task_ids"]
            for tid in task_ids.unique().tolist():
                tname = self.task_names[tid]
                per_task_loss[tname] += batch_loss
                per_task_count[tname] += 1

        avg_loss = total_loss / max(num_batches, 1)
        metrics = {"val_loss": avg_loss}
        for tname, tloss in per_task_loss.items():
            metrics[f"val_loss/{tname}"] = tloss / max(per_task_count[tname], 1)
        return metrics

    def train(self) -> dict[str, float]:
        patience_counter = 0
        best_metrics: dict[str, float] = {}

        if self.is_main:
            logger.info(
                "Starting multitask training: %d epochs, patience %d, tasks %s",
                self.max_epochs, self.patience, self.task_names,
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
                per_task_str = "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in sorted(val_metrics.items())
                    if "/" in k
                )
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  [%s]  lr=%.2e  time=%.1fs",
                    epoch + 1, self.max_epochs,
                    train_metrics["train_loss"], val_metrics["val_loss"],
                    per_task_str, lr, elapsed,
                )

                all_metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": lr}
                if self.use_wandb and self.wandb_run is not None:
                    import wandb
                    wandb.log(all_metrics, step=epoch)

                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                    best_metrics = all_metrics.copy()
                    self.save_checkpoint(f"best_ep{epoch+1}_vl{val_metrics['val_loss']:.4f}.pt", epoch, val_metrics["val_loss"])
                    logger.info("  -> new best val_loss=%.4f -- checkpoint saved", self.best_val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(
                            "Early stopping at epoch %d (patience %d exhausted)",
                            epoch + 1, self.patience,
                        )
                        break

                if (epoch + 1) % 30 == 0:
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

        if filename.startswith("best"):
            for old in self.checkpoint_dir.glob("best_ep*_vl*.pt"):
                if old.name != filename:
                    old.unlink(missing_ok=True)
            best_link = self.checkpoint_dir / "best.pt"
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(filename)

            if self.s3_syncer is not None:
                try:
                    self.s3_syncer.upload_checkpoint(str(path), key_name="best.pt")
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
    parser = argparse.ArgumentParser(description="Multitask CSI-Text alignment training")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to multitask experiment YAML config",
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

    config = _merge_multitask_configs(config)

    if args.override:
        _apply_overrides(config, args.override)

    trainer = MultitaskTrainer(config, local_rank=local_rank, rank=rank, world_size=world_size)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
