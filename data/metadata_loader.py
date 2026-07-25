"""Unified metadata-driven data loader for per-sample H5 files.

Reads metadata CSV + JSON split files to index per-sample HDF5 files
across CSI-Bench, XRF55, and SignFi datasets (all preprocessed to a
common per-sample H5 structure).
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# XRF55 dynamic key pattern: extract numeric parts from filename
# e.g. "user04_act06_repeat02_env4.h5" -> "04_06_02"
_XRF55_KEY_PARTS = re.compile(r"user(\d+)_act(\d+)_repeat(\d+)")


def _resolve_xrf55_key(filename: str) -> str:
    """Extract the H5 dataset key from an XRF55 filename."""
    m = _XRF55_KEY_PARTS.search(filename)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    stem = Path(filename).stem
    parts = re.findall(r"\d+", stem)
    if len(parts) >= 3:
        return "_".join(parts[:3])
    raise ValueError(f"Cannot resolve XRF55 H5 key from filename: {filename}")


class MetadataLoader:
    """Loads CSI data from per-sample H5 files indexed by a metadata CSV.

    Parameters
    ----------
    task_root : str or Path
        Root directory for a single task/dataset, containing ``metadata/``
        and ``splits/`` subdirectories alongside the H5 files.
    h5_key : str
        Default HDF5 dataset key for reading CSI data (e.g. ``CSI_amps``).
        For XRF55, set to ``"auto"`` to derive the key from the filename.
    metadata_csv : str, optional
        Relative path within *task_root* to the metadata CSV.
        Default: ``metadata/sample_metadata.csv``.
    splits_dir : str, optional
        Relative path within *task_root* to the split JSON directory.
        Default: ``splits``.
    """

    def __init__(
        self,
        task_root: str | Path,
        h5_key: str = "CSI_amps",
        metadata_csv: str = "metadata/sample_metadata.csv",
        splits_dir: str = "splits",
    ) -> None:
        self.task_root = Path(task_root)
        self.h5_key = h5_key
        self._csv_path = self.task_root / metadata_csv
        self._splits_dir = self.task_root / splits_dir

        if not self._csv_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {self._csv_path}")

        self._df = pd.read_csv(self._csv_path)
        self._index: dict[str, dict[str, Any]] = {}
        self._build_index()

        logger.info(
            "MetadataLoader: task_root=%s  samples=%d  h5_key=%s",
            self.task_root, len(self._index), self.h5_key,
        )

    def _build_index(self) -> None:
        """Build {sample_id -> metadata_dict} from the CSV."""
        for _, row in self._df.iterrows():
            sid = str(row["id"])
            self._index[sid] = {
                "file_path": str(row["file_path"]),
                "label": row["label"],
            }
            for col in self._df.columns:
                if col not in ("id", "file_path", "label"):
                    self._index[sid][col] = row[col]

    def get_split_ids(self, split: str) -> list[str]:
        """Load sample IDs for a given split from the JSON file."""
        candidates = [
            self._splits_dir / f"{split}_id.json",
            self._splits_dir / f"{split}_ids.json",
            self._splits_dir / f"{split}.json",
        ]
        for p in candidates:
            if p.is_file():
                with open(p) as f:
                    payload = json.load(f)
                ids = self._extract_split_ids(payload, split, p)
                logger.info("Loaded %d IDs for split '%s' from %s", len(ids), split, p)
                return ids

        raise FileNotFoundError(
            f"Split file not found for '{split}'. Searched: {[str(c) for c in candidates]}"
        )

    @staticmethod
    def _extract_split_ids(payload: Any, split: str, path: Path) -> list[str]:
        """Normalise a split file payload to a flat list of sample IDs.

        Accepts a bare list, or an object wrapping the list under one of
        ``test`` / ``ids`` / ``sample_ids`` / ``samples`` / the split name.
        """
        if isinstance(payload, dict):
            for key in (split, "test", "ids", "sample_ids", "samples"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError(
                    f"Split file {path} is an object without a recognised ID list "
                    f"key. Found keys: {sorted(payload.keys())}"
                )

        if not isinstance(payload, list):
            raise ValueError(
                f"Split file {path} must contain a list of sample IDs, "
                f"got {type(payload).__name__}"
            )

        return [str(i) for i in payload]

    def get_metadata(self, sample_id: str) -> dict[str, Any]:
        """Return the full metadata dict for a sample."""
        if sample_id not in self._index:
            raise KeyError(f"Sample ID '{sample_id}' not in metadata index")
        return dict(self._index[sample_id])

    def get_label(self, sample_id: str) -> Any:
        return self._index[sample_id]["label"]

    def get_all_sample_ids(self) -> list[str]:
        return list(self._index.keys())

    def _resolve_h5_key(self, file_path: str) -> str:
        """Resolve the HDF5 dataset key for a given file."""
        if self.h5_key != "auto":
            return self.h5_key
        return _resolve_xrf55_key(Path(file_path).name)

    def _resolve_file_path(self, relative_path: str) -> Path:
        """Resolve a relative file_path from the CSV to an absolute path.

        Tries ``task_root / rel`` first.  If not found, scans immediate
        subdirectories of *task_root* (e.g. ``sub_Human/``) as a fallback
        — CSI-Bench standalone tasks store H5s in a subdirectory that may
        not be part of the CSV's ``file_path`` column.
        """
        rel = relative_path.lstrip("./")
        direct = self.task_root / rel
        if direct.is_file():
            return direct

        if not hasattr(self, "_h5_subdir"):
            self._h5_subdir: str | None = None
            for child in self.task_root.iterdir():
                if child.is_dir() and child.name not in ("metadata", "splits"):
                    candidate = child / rel
                    if candidate.is_file():
                        self._h5_subdir = child.name
                        break

        if self._h5_subdir is not None:
            return self.task_root / self._h5_subdir / rel

        return direct

    def file_exists(self, sample_id: str) -> bool:
        """Check whether the H5 file for *sample_id* exists on disk."""
        meta = self._index.get(sample_id)
        if meta is None:
            return False
        return self._resolve_file_path(meta["file_path"]).is_file()

    def filter_valid_ids(self, sample_ids: list[str]) -> list[str]:
        """Return the subset of *sample_ids* whose H5 files exist on disk.

        Logs a warning if any IDs are dropped.
        """
        valid = [sid for sid in sample_ids if self.file_exists(sid)]
        n_dropped = len(sample_ids) - len(valid)
        if n_dropped:
            logger.warning(
                "Dropped %d / %d samples with missing H5 files in %s",
                n_dropped, len(sample_ids), self.task_root,
            )
        return valid

    def load_sample(self, sample_id: str) -> np.ndarray:
        """Load a single CSI sample from its H5 file.

        Returns
        -------
        np.ndarray, shape [1, F, T]
            CSI amplitude tensor with C=1 channel.
        """
        meta = self._index[sample_id]
        fpath = self._resolve_file_path(meta["file_path"])

        if not fpath.is_file():
            raise FileNotFoundError(f"H5 file not found: {fpath} (sample={sample_id})")

        key = self._resolve_h5_key(meta["file_path"])

        with h5py.File(fpath, "r") as f:
            if key not in f:
                available = list(f.keys())
                if len(available) == 1:
                    key = available[0]
                else:
                    raise KeyError(
                        f"Key '{key}' not in {fpath}. Available: {available}"
                    )
            data = f[key][:]

        data = data.astype(np.float32)

        # Input shape: (F, T, 1) -> output: (1, F, T)
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)  # (F, T)
            data = data[np.newaxis, :, :]  # (1, F, T)
        elif data.ndim == 2:
            data = data[np.newaxis, :, :]  # (1, F, T)
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)  # assume (F, T, C) -> (C, F, T)
        else:
            raise ValueError(
                f"Unexpected CSI shape {data.shape} for sample {sample_id}"
            )

        return data

    def load_samples_batch(
        self,
        sample_ids: list[str],
        show_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load multiple samples into contiguous arrays.

        Returns
        -------
        csi_data : np.ndarray, shape [N, C, F, T]
        labels : np.ndarray, shape [N]
        sample_ids : list[str]
        """
        iterator = sample_ids
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(sample_ids, desc="Loading samples")

        samples = []
        labels = []
        for sid in iterator:
            samples.append(self.load_sample(sid))
            labels.append(self.get_label(sid))

        csi_data = np.stack(samples, axis=0)
        labels_arr = np.array(labels)

        return csi_data, labels_arr, list(sample_ids)

    @property
    def num_samples(self) -> int:
        return len(self._index)

    @property
    def label_set(self) -> set:
        return {meta["label"] for meta in self._index.values()}

    def __repr__(self) -> str:
        return (
            f"MetadataLoader(task_root={self.task_root}, "
            f"samples={self.num_samples}, h5_key={self.h5_key!r})"
        )
