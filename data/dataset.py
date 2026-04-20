"""Unified CSI-Text PyTorch Dataset with many-to-many sampling.

Backed by MetadataLoader for per-sample H5 file access, with optional
in-memory preloading for maximum GPU throughput.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from data.metadata_loader import MetadataLoader
from data.pair_registry import PairRegistry

logger = logging.getLogger(__name__)


class CSIAugmentation:
    """Stochastic augmentations for CSI tensors (training only).

    All operations work on numpy arrays of shape ``[C, K, T]``.
    Each augmentation is applied independently with its own probability.
    """

    def __init__(
        self,
        time_mask_ratio: float = 0.1,
        subcarrier_mask_ratio: float = 0.1,
        noise_std: float = 0.05,
        scale_range: tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ) -> None:
        self.time_mask_ratio = time_mask_ratio
        self.subcarrier_mask_ratio = subcarrier_mask_ratio
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.p = p
        self._rng = np.random.default_rng()

    def __call__(self, csi: np.ndarray) -> np.ndarray:
        csi = csi.copy()
        C, K, T = csi.shape

        if self._rng.random() < self.p and self.time_mask_ratio > 0:
            mask_len = max(1, int(T * self.time_mask_ratio))
            start = self._rng.integers(0, T - mask_len + 1)
            csi[:, :, start : start + mask_len] = 0.0

        if self._rng.random() < self.p and self.subcarrier_mask_ratio > 0:
            mask_len = max(1, int(K * self.subcarrier_mask_ratio))
            start = self._rng.integers(0, K - mask_len + 1)
            csi[:, start : start + mask_len, :] = 0.0

        if self._rng.random() < self.p and self.noise_std > 0:
            csi = csi + self._rng.normal(0, self.noise_std, size=csi.shape).astype(csi.dtype)

        if self._rng.random() < self.p:
            lo, hi = self.scale_range
            scale = self._rng.uniform(lo, hi)
            csi = csi * scale

        return csi


class CSITextDataset(Dataset):
    """Unified CSI-Text dataset with many-to-many pairing.

    Supports two loading modes controlled by ``preload``:
      - **lazy** (default): each ``__getitem__`` opens the per-sample H5.
      - **preload**: all samples loaded into a contiguous tensor at init.

    Each ``__getitem__`` returns a dict:
        - ``csi``:            torch.Tensor  [C, K, T]
        - ``text_id``:        str
        - ``text_embedding``: torch.Tensor  [D] | None
        - ``label``:          int
        - ``sample_id``:      str
    """

    def __init__(
        self,
        task_root: str,
        split: str,
        h5_key: str = "CSI_amps",
        registry_path: Optional[str] = None,
        text_embeddings_path: Optional[str] = None,
        target_subcarriers: int = 64,
        target_time_steps: int = 500,
        preload: bool = False,
        transform: Optional[Any] = None,
    ) -> None:
        self.task_root = Path(task_root)
        self.split = split
        self.h5_key = h5_key
        self.target_subcarriers = target_subcarriers
        self.target_time_steps = target_time_steps
        self.transform = transform
        self.is_train = split == "train"
        self.preload = preload

        self.loader = MetadataLoader(
            task_root=task_root,
            h5_key=h5_key,
        )

        raw_ids = self.loader.get_split_ids(split)
        self.sample_ids = self.loader.filter_valid_ids(raw_ids)
        self._id_to_idx = {sid: i for i, sid in enumerate(self.sample_ids)}
        self.num_samples = len(self.sample_ids)

        self.labels = np.array(
            [self.loader.get_label(sid) for sid in self.sample_ids]
        )
        self._build_label_map()

        self.registry: Optional[PairRegistry] = None
        if registry_path is not None and Path(registry_path).is_file():
            self.registry = PairRegistry.load(registry_path)

        self._preloaded_data: Optional[np.ndarray] = None
        if preload:
            self._preload_all()

        self.text_embeddings: Optional[dict[str, np.ndarray]] = None
        self.embedding_dim: Optional[int] = None
        if text_embeddings_path is not None:
            self._load_text_embeddings(text_embeddings_path)

        self._rng = np.random.default_rng()

        logger.info(
            "CSITextDataset: task_root=%s split=%s samples=%d preload=%s "
            "target=[%d,%d] embeddings=%s",
            task_root, split, self.num_samples, preload,
            target_subcarriers, target_time_steps, self.embedding_dim,
        )

    def _build_label_map(self) -> None:
        """Map raw labels to integer indices.

        Uses ``{task_root}/metadata/label_mapping.json`` when available so that
        class indices stay consistent with the pair-registry / text-prototype
        numbering even when a split omits some classes (e.g. user_id test split
        missing the held-out user U02).  Falls back to sorted-unique assignment.
        """
        lm_path = self.task_root / "metadata" / "label_mapping.json"
        if lm_path.is_file():
            with open(lm_path) as f:
                lm = json.load(f)
            label_to_idx = {str(k): int(v) for k, v in lm["label_to_idx"].items()}
            raw_set = set(self.labels.tolist())
            missing = raw_set - label_to_idx.keys()
            # #region agent log
            import time as _t; _log_data = {"sessionId":"3be70b","runId":"run1","hypothesisId":"H1+H2","location":"data/dataset.py:_build_label_map","message":"label_map_debug","data":{"lm_path":str(lm_path),"raw_set_sample":list(sorted(raw_set))[:5],"raw_set_types":type(list(raw_set)[0]).__name__ if raw_set else "empty","lm_keys_sample":list(sorted(label_to_idx.keys()))[:5],"lm_keys_type":type(list(label_to_idx.keys())[0]).__name__ if label_to_idx else "empty","num_raw":len(raw_set),"num_lm_keys":len(label_to_idx),"num_missing":len(missing),"missing_sample":list(sorted(missing))[:5]},"timestamp":int(_t.time()*1000)}; open("/Users/guozhenzhu/Documents/GitHub/CSI-Text-Multimodal-Alignment/.cursor/debug-3be70b.log","a").write(__import__("json").dumps(_log_data)+"\n")
            # #endregion
            if missing:
                raise ValueError(
                    f"Labels {missing} found in data but absent from "
                    f"{lm_path}; update label_mapping.json."
                )
            self._label_to_int = {lbl: label_to_idx[lbl] for lbl in raw_set}
            self.int_labels = np.array(
                [self._label_to_int[lbl] for lbl in self.labels.tolist()],
                dtype=np.int64,
            )
            self.num_classes = int(lm.get("num_classes", len(label_to_idx)))
            logger.debug(
                "Label map from %s: %d classes, %d present in split",
                lm_path, self.num_classes, len(raw_set),
            )
        else:
            unique_labels = sorted(set(self.labels.tolist()))
            self._label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
            self.int_labels = np.array(
                [self._label_to_int[lbl] for lbl in self.labels.tolist()],
                dtype=np.int64,
            )
            self.num_classes = len(unique_labels)

    def _preload_all(self) -> None:
        """Load all samples into memory as a contiguous array."""
        logger.info("Preloading %d samples into memory...", self.num_samples)
        samples = []
        for sid in self.sample_ids:
            raw = self.loader.load_sample(sid)
            normed = self._normalize_shape(raw)
            samples.append(normed)
        self._preloaded_data = np.stack(samples, axis=0).astype(np.float32)
        logger.info("Preloaded shape: %s", self._preloaded_data.shape)

    def _normalize_shape(self, csi: np.ndarray) -> np.ndarray:
        """Interpolate CSI to [C, target_subcarriers, target_time_steps]."""
        C, K, T = csi.shape

        if K != self.target_subcarriers:
            x_old = np.linspace(0, 1, K)
            x_new = np.linspace(0, 1, self.target_subcarriers)
            out = np.empty((C, self.target_subcarriers, T), dtype=csi.dtype)
            for c in range(C):
                f = interp1d(x_old, csi[c], axis=0, kind="linear")
                out[c] = f(x_new)
            csi = out
            K = self.target_subcarriers

        if T != self.target_time_steps:
            x_old = np.linspace(0, 1, T)
            x_new = np.linspace(0, 1, self.target_time_steps)
            out = np.empty((C, K, self.target_time_steps), dtype=csi.dtype)
            for c in range(C):
                f = interp1d(x_old, csi[c], axis=1, kind="linear")
                out[c] = f(x_new)
            csi = out

        mean = csi.mean()
        std = csi.std()
        if std > 1e-8:
            csi = (csi - mean) / std
        else:
            csi = csi - mean

        return csi

    def _load_text_embeddings(self, path: str) -> None:
        with h5py.File(path, "r") as f:
            self.text_embeddings = {}
            for text_id in f.keys():
                self.text_embeddings[text_id] = f[text_id][:].astype(np.float32)
            if self.text_embeddings:
                first = next(iter(self.text_embeddings.values()))
                self.embedding_dim = first.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_id = self.sample_ids[idx]
        label = int(self.int_labels[idx])

        if self._preloaded_data is not None:
            csi = self._preloaded_data[idx].copy()
        else:
            raw = self.loader.load_sample(sample_id)
            csi = self._normalize_shape(raw)

        if self.transform is not None:
            csi = self.transform(csi)

        csi_tensor = torch.from_numpy(csi.astype(np.float32))

        text_id = ""
        if self.registry is not None:
            positive_texts = self.registry.get_positive_texts(sample_id)
            if positive_texts:
                if self.is_train:
                    text_id = str(self._rng.choice(positive_texts))
                else:
                    text_id = positive_texts[0]

        text_embedding: Optional[torch.Tensor] = None
        if self.text_embeddings is not None and text_id in self.text_embeddings:
            text_embedding = torch.from_numpy(self.text_embeddings[text_id])

        return {
            "csi": csi_tensor,
            "text_id": text_id,
            "text_embedding": text_embedding,
            "label": label,
            "sample_id": sample_id,
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate that stacks CSI tensors and handles optional embeddings."""
        csi = torch.stack([b["csi"] for b in batch])
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        text_ids = [b["text_id"] for b in batch]
        sample_ids = [b["sample_id"] for b in batch]

        has_embeddings = all(b["text_embedding"] is not None for b in batch)
        if has_embeddings:
            text_embeddings = torch.stack([b["text_embedding"] for b in batch])
        else:
            text_embeddings = None

        return {
            "csi": csi,
            "text_ids": text_ids,
            "text_embeddings": text_embeddings,
            "labels": labels,
            "sample_ids": sample_ids,
        }


def build_dataloaders(
    config: dict[str, Any],
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders from an experiment config.

    Parameters
    ----------
    config : dict
        Expected keys::

            task_root:           str   path to task directory
            h5_key:              str   HDF5 dataset key (or "auto")
            registry_path:       str | None
            text_embeddings_path: str | None
            target_subcarriers:  int   (default 64)
            target_time_steps:   int   (default 500)
            batch_size:          int   (default 128)
            num_workers:         int   (default 4)
            prefetch_factor:     int   (default 2)
            persistent_workers:  bool  (default True)
            preload:             bool  (default False)

    distributed : bool
        Whether to use DistributedSampler.
    world_size, rank : int
        For distributed training.

    Returns
    -------
    dict with keys "train", "val", "test" mapping to DataLoaders.
    """
    shared_kwargs: dict[str, Any] = {
        "task_root": config["task_root"],
        "h5_key": config.get("h5_key", "CSI_amps"),
        "registry_path": config.get("registry_path"),
        "text_embeddings_path": config.get("text_embeddings_path"),
        "target_subcarriers": config.get("target_subcarriers", 64),
        "target_time_steps": config.get("target_time_steps", 500),
        "preload": config.get("preload", False),
    }

    aug_cfg = config.get("augmentation", {})
    train_augmentation = None
    if aug_cfg.get("enabled", False):
        train_augmentation = CSIAugmentation(
            time_mask_ratio=float(aug_cfg.get("time_mask_ratio", 0.1)),
            subcarrier_mask_ratio=float(aug_cfg.get("subcarrier_mask_ratio", 0.1)),
            noise_std=float(aug_cfg.get("noise_std", 0.05)),
            scale_range=(
                float(aug_cfg.get("scale_min", 0.9)),
                float(aug_cfg.get("scale_max", 1.1)),
            ),
            p=float(aug_cfg.get("p", 0.5)),
        )
        logger.info("CSI augmentation enabled: %s", aug_cfg)

    batch_size = config.get("batch_size", 128)
    num_workers = config.get("num_workers", 4)
    prefetch_factor = config.get("prefetch_factor", 2)
    persistent_workers = config.get("persistent_workers", True) and num_workers > 0

    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        transform = train_augmentation if split == "train" else None
        ds = CSITextDataset(split=split, transform=transform, **shared_kwargs)

        sampler = None
        shuffle = split == "train"
        if distributed:
            sampler = DistributedSampler(
                ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            shuffle = False

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=CSITextDataset.collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )

    logger.info(
        "DataLoaders built: %s",
        {k: len(v.dataset) for k, v in loaders.items()},
    )
    return loaders
