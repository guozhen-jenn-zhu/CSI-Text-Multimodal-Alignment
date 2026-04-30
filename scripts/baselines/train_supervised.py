"""Train a supervised ResNet-18 1D classifier on a CSI task.

Reuses the existing CSITextDataset for data loading and ResNet18_1D as the
backbone, adding a linear classification head trained with cross-entropy.

Usage:
    python -m scripts.baselines.train_supervised \
        --task_root /path/to/xrf55/action_recognition \
        --num_classes 55 \
        --h5_key auto \
        --output_dir results/supervised/xrf55

    python -m scripts.baselines.train_supervised \
        --task_root /path/to/signfi/sign_recognition \
        --num_classes 276 \
        --h5_key amplitude \
        --output_dir results/supervised/signfi
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import CSITextDataset
from models.encoders.csi_encoder import ResNet18_1D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SupervisedCSIClassifier(nn.Module):
    """ResNet-18 1D backbone + linear classification head."""

    def __init__(
        self,
        input_channels: int,
        input_subcarriers: int,
        input_timesteps: int,
        num_classes: int,
        embed_dim: int = 256,
        base_width: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = ResNet18_1D(
            input_channels=input_channels,
            input_subcarriers=input_subcarriers,
            input_timesteps=input_timesteps,
            embed_dim=embed_dim,
            base_width=base_width,
            dropout=dropout,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def build_dataloader(
    task_root: str,
    split: str,
    h5_key: str,
    target_subcarriers: int,
    target_time_steps: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = CSITextDataset(
        task_root=task_root,
        split=split,
        h5_key=h5_key,
        target_subcarriers=target_subcarriers,
        target_time_steps=target_time_steps,
        preload=False,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=CSITextDataset.collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        csi = batch["csi"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(csi)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        csi = batch["csi"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast("cuda", dtype=torch.bfloat16):
            logits = model(csi)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised ResNet-18 1D classifier on CSI data"
    )
    parser.add_argument("--task_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--h5_key", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--input_channels", type=int, default=1)
    parser.add_argument("--target_subcarriers", type=int, default=64)
    parser.add_argument("--target_time_steps", type=int, default=500)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--base_width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building dataloaders from %s ...", args.task_root)
    dl_kwargs = dict(
        task_root=args.task_root,
        h5_key=args.h5_key,
        target_subcarriers=args.target_subcarriers,
        target_time_steps=args.target_time_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = build_dataloader(split="train", **dl_kwargs)
    val_loader = build_dataloader(split="val", **dl_kwargs)
    test_loader = build_dataloader(split="test", **dl_kwargs)

    logger.info(
        "Samples — train: %d  val: %d  test: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = SupervisedCSIClassifier(
        input_channels=args.input_channels,
        input_subcarriers=args.target_subcarriers,
        input_timesteps=args.target_time_steps,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        base_width=args.base_width,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s (%.2fM)", f"{n_params:,}", n_params / 1e6)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)
    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: list[dict] = []

    logger.info("Starting training for up to %d epochs ...", args.max_epochs)
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.grad_clip
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "train_acc": round(train_acc, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "lr": lr,
            }
        )

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            improved = " *"
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            logger.info(
                "Epoch %3d | train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f | lr=%.2e%s",
                epoch, train_loss, train_acc, val_loss, val_acc, lr, improved,
            )

        if epochs_no_improve >= args.patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch, args.patience,
            )
            break

    elapsed = time.time() - t0
    logger.info(
        "Training finished in %.1fs. Best epoch: %d (val_loss=%.5f)",
        elapsed, best_epoch, best_val_loss,
    )

    logger.info("Loading best checkpoint (epoch %d) for test evaluation ...", best_epoch)
    model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device))

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    test_f1 = f1_score(test_labels, test_preds, average="weighted")
    test_f1_macro = f1_score(test_labels, test_preds, average="macro")

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("  Accuracy:       %.4f (%.2f%%)", test_acc, test_acc * 100)
    logger.info("  Weighted F1:    %.4f", test_f1)
    logger.info("  Macro F1:       %.4f", test_f1_macro)
    logger.info("  Test loss:      %.5f", test_loss)
    logger.info("=" * 60)

    results = {
        "test_accuracy": round(test_acc, 5),
        "test_accuracy_pct": round(test_acc * 100, 2),
        "test_weighted_f1": round(test_f1, 5),
        "test_macro_f1": round(test_f1_macro, 5),
        "test_loss": round(test_loss, 5),
        "best_epoch": best_epoch,
        "total_epochs": len(history),
        "training_time_s": round(elapsed, 1),
        "num_classes": args.num_classes,
        "num_params": n_params,
        "config": {
            "task_root": args.task_root,
            "h5_key": args.h5_key,
            "input_channels": args.input_channels,
            "target_subcarriers": args.target_subcarriers,
            "target_time_steps": args.target_time_steps,
            "embed_dim": args.embed_dim,
            "base_width": args.base_width,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "seed": args.seed,
        },
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }

    with open(output_dir / "supervised_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    cm = confusion_matrix(test_labels, test_preds)
    np.save(output_dir / "confusion_matrix.npy", cm)

    present_classes = sorted(set(test_labels.tolist()))
    report = classification_report(
        test_labels,
        test_preds,
        labels=present_classes,
        output_dict=False,
        digits=4,
    )
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    summary = (
        f"# Supervised Baseline Results\n\n"
        f"- **Task root**: `{args.task_root}`\n"
        f"- **Classes**: {args.num_classes}\n"
        f"- **Encoder**: ResNet-18 1D (embed_dim={args.embed_dim})\n"
        f"- **Train/Val/Test**: {len(train_loader.dataset)} / "
        f"{len(val_loader.dataset)} / {len(test_loader.dataset)}\n\n"
        f"## Results\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Accuracy | {test_acc*100:.2f}% |\n"
        f"| Weighted F1 | {test_f1:.4f} |\n"
        f"| Macro F1 | {test_f1_macro:.4f} |\n"
        f"| Best Epoch | {best_epoch} |\n"
        f"| Training Time | {elapsed:.0f}s |\n"
    )
    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
