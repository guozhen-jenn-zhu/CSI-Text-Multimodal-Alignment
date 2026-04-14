"""Learning rate schedulers for CSI-Text multimodal alignment training."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)


class WarmupCosineScheduler(LRScheduler):
    """Cosine annealing with linear warmup.

    During the first ``warmup_epochs`` epochs the learning rate ramps linearly
    from ``min_lr`` to the optimizer's initial LR.  After warmup it follows a
    cosine schedule decaying back to ``min_lr`` over the remaining epochs.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * alpha
                for base_lr in self.base_lrs
            ]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        return [
            self.min_lr
            + 0.5 * (base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> LRScheduler | ReduceLROnPlateau:
    """Build a learning rate scheduler from a training config dict.

    Supported ``config["scheduler"]`` values:

    - ``"cosine"`` — :class:`WarmupCosineScheduler` with linear warmup.
    - ``"step"``   — :class:`StepLR` (step_size from config, default 30).
    - ``"plateau"``— :class:`ReduceLROnPlateau` monitoring ``val_loss``.

    Falls back to ``"cosine"`` when the key is absent.
    """
    name = config.get("scheduler", "cosine").lower()

    if name == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=int(config.get("warmup_epochs", 5)),
            total_epochs=int(config.get("max_epochs", 100)),
            min_lr=float(config.get("min_lr", 1e-6)),
        )

    if name == "step":
        return StepLR(
            optimizer,
            step_size=int(config.get("step_size", 30)),
            gamma=float(config.get("gamma", 0.1)),
        )

    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("plateau_patience", 5)),
            min_lr=float(config.get("min_lr", 1e-6)),
        )

    raise ValueError(
        f"Unknown scheduler '{name}'. Available: cosine, step, plateau"
    )
