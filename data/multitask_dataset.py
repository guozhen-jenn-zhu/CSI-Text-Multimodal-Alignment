"""Multitask CSI-Text dataset that concatenates multiple single-task datasets.

Used for joint training across tasks (within a dataset or cross-dataset).
Each sample carries a ``task_id`` so per-task metrics can be computed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, WeightedRandomSampler

from data.dataset import CSIAugmentation, CSITextDataset

logger = logging.getLogger(__name__)


class MultitaskCSITextDataset(Dataset):
    """Wraps several :class:`CSITextDataset` instances into one index space.

    Each ``__getitem__`` returns the same dict as ``CSITextDataset`` plus:
      - ``task_id``:   int   (index into ``task_names``)
      - ``task_name``: str

    Parameters
    ----------
    task_configs : list[dict]
        One entry per task.  Required keys::

            task_name, task_root, h5_key

        Optional keys (forwarded to ``CSITextDataset``)::

            registry_path, text_embeddings_path

    split : str
        ``"train"``, ``"val"``, or ``"test"``.
    target_subcarriers, target_time_steps : int
        Spatial / temporal interpolation targets.
    preload : bool
        If ``True``, every sub-dataset preloads into RAM.
    transform : callable | None
        Applied to CSI arrays (only for training split).
    """

    def __init__(
        self,
        task_configs: list[dict[str, Any]],
        split: str,
        target_subcarriers: int = 64,
        target_time_steps: int = 500,
        preload: bool = False,
        transform: Optional[Any] = None,
    ) -> None:
        self.task_names: list[str] = []
        self.sub_datasets: list[CSITextDataset] = []
        self.offsets: list[int] = []

        offset = 0
        for cfg in task_configs:
            ds = CSITextDataset(
                task_root=cfg["task_root"],
                split=split,
                h5_key=cfg.get("h5_key", "CSI_amps"),
                registry_path=cfg.get("registry_path"),
                text_embeddings_path=cfg.get("text_embeddings_path"),
                target_subcarriers=target_subcarriers,
                target_time_steps=target_time_steps,
                preload=preload,
                transform=transform if split == "train" else None,
            )
            self.task_names.append(cfg["task_name"])
            self.sub_datasets.append(ds)
            self.offsets.append(offset)
            offset += len(ds)

        self.total_samples = offset
        logger.info(
            "MultitaskCSITextDataset: split=%s tasks=%s total=%d",
            split, {n: len(d) for n, d in zip(self.task_names, self.sub_datasets)},
            self.total_samples,
        )

    def _locate(self, idx: int) -> tuple[int, int]:
        """Map a global index to (task_idx, local_idx)."""
        for i in range(len(self.offsets) - 1, -1, -1):
            if idx >= self.offsets[i]:
                return i, idx - self.offsets[i]
        raise IndexError(f"Index {idx} out of range for {self.total_samples} samples")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        task_idx, local_idx = self._locate(idx)
        item = self.sub_datasets[task_idx][local_idx]
        item["task_id"] = task_idx
        item["task_name"] = self.task_names[task_idx]
        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate that adds ``task_ids`` tensor."""
        csi = torch.stack([b["csi"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        task_ids = torch.tensor([b["task_id"] for b in batch], dtype=torch.long)
        text_ids = [b["text_id"] for b in batch]
        sample_ids = [b["sample_id"] for b in batch]
        task_names = [b["task_name"] for b in batch]

        has_embeddings = all(b["text_embedding"] is not None for b in batch)
        text_embeddings = (
            torch.stack([b["text_embedding"] for b in batch])
            if has_embeddings
            else None
        )

        return {
            "csi": csi,
            "text_ids": text_ids,
            "text_embeddings": text_embeddings,
            "labels": labels,
            "sample_ids": sample_ids,
            "task_ids": task_ids,
            "task_names": task_names,
        }

    def build_sample_weights(self) -> np.ndarray:
        """Compute per-sample weights for proportional task balancing.

        Each task contributes equally regardless of its size, so smaller
        tasks are upsampled and larger tasks are downsampled.
        """
        weights = np.empty(self.total_samples, dtype=np.float64)
        for i, ds in enumerate(self.sub_datasets):
            start = self.offsets[i]
            end = start + len(ds)
            weights[start:end] = 1.0 / len(ds)
        weights /= weights.sum()
        return weights


def build_multitask_dataloaders(
    task_configs: list[dict[str, Any]],
    *,
    target_subcarriers: int = 64,
    target_time_steps: int = 500,
    batch_size: int = 128,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    preload: bool = False,
    sampling: str = "proportional",
    augmentation: Optional[dict] = None,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders for multitask training.

    Parameters
    ----------
    task_configs : list[dict]
        Per-task configs (see ``MultitaskCSITextDataset``).
    sampling : str
        ``"proportional"`` (balanced across tasks) or ``"natural"``
        (proportional to dataset sizes, i.e. simple shuffle).
    augmentation : dict | None
        Augmentation config dict (same keys as single-task).
    """
    train_transform = None
    if augmentation and augmentation.get("enabled", False):
        train_transform = CSIAugmentation(
            time_mask_ratio=float(augmentation.get("time_mask_ratio", 0.1)),
            subcarrier_mask_ratio=float(augmentation.get("subcarrier_mask_ratio", 0.1)),
            noise_std=float(augmentation.get("noise_std", 0.05)),
            scale_range=(
                float(augmentation.get("scale_min", 0.9)),
                float(augmentation.get("scale_max", 1.1)),
            ),
            p=float(augmentation.get("p", 0.5)),
        )

    loaders: dict[str, DataLoader] = {}
    pw = persistent_workers and num_workers > 0

    for split in ("train", "val", "test"):
        transform = train_transform if split == "train" else None
        ds = MultitaskCSITextDataset(
            task_configs=task_configs,
            split=split,
            target_subcarriers=target_subcarriers,
            target_time_steps=target_time_steps,
            preload=preload,
            transform=transform,
        )

        sampler = None
        shuffle = split == "train"

        if distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
            shuffle = False
        elif split == "train" and sampling == "proportional" and len(ds.sub_datasets) > 1:
            weights = ds.build_sample_weights()
            sampler = WeightedRandomSampler(
                weights=weights.tolist(),
                num_samples=len(ds),
                replacement=True,
            )
            shuffle = False

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=MultitaskCSITextDataset.collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=pw,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    logger.info(
        "Multitask DataLoaders built: %s",
        {k: len(v.dataset) for k, v in loaders.items()},
    )
    return loaders
