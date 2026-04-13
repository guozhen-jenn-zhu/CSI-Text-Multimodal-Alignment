"""CSI-Bench dataset loader.

CSI-Bench stores data in HDF5 files. Each sample is a CSI tensor
X in R^{C x K x T} (channels x subcarriers x time), amplitude-only, normalized.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

TASK_CONFIG = {
    "fall_detection": {
        "num_classes": 2,
        "dataset_type": "ST",
        "window_sec": 5,
        "sample_rate": 100,
    },
    "breath_detection": {
        "num_classes": 2,
        "dataset_type": "ST",
        "window_sec": 10,
        "sample_rate": 30,
    },
    "motion_source": {
        "num_classes": 4,
        "dataset_type": "ST",
        "window_sec": 5,
        "sample_rate": 100,
    },
    "localization": {
        "num_classes": 6,
        "dataset_type": "ST",
        "window_sec": 5,
        "sample_rate": 100,
    },
    "har": {
        "num_classes": 5,
        "dataset_type": "MT",
        "window_sec": 5,
        "sample_rate": 100,
    },
    "user_id": {
        "num_classes": 6,
        "dataset_type": "MT",
        "window_sec": 5,
        "sample_rate": 100,
    },
    "proximity": {
        "num_classes": 4,
        "dataset_type": "MT",
        "window_sec": 5,
        "sample_rate": 100,
    },
}

_MT_TASKS = ["har", "user_id", "proximity"]


class CSIBenchLoader:
    """Loader for CSI-Bench HDF5 datasets."""

    TASK_CONFIG = TASK_CONFIG

    def __init__(self, data_root: str, task: str, split: str = "train") -> None:
        if task not in TASK_CONFIG:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_CONFIG.keys())}"
            )
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.data_root = Path(data_root)
        self.task = task
        self.split = split
        self.config = TASK_CONFIG[task]

        self._h5_path = self._resolve_h5_path()

    def _resolve_h5_path(self) -> Path:
        """Locate the HDF5 file for this task/split combination."""
        candidates = [
            self.data_root / self.task / f"{self.split}.h5",
            self.data_root / self.task / f"{self.split}.hdf5",
            self.data_root / f"{self.task}_{self.split}.h5",
            self.data_root / f"{self.task}_{self.split}.hdf5",
        ]
        for p in candidates:
            if p.is_file():
                return p

        raise FileNotFoundError(
            f"CSI-Bench HDF5 file not found for task='{self.task}', split='{self.split}'. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    def load_samples(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load CSI samples and labels from HDF5.

        Returns
        -------
        samples : np.ndarray, shape [N, C, K, T]
            Amplitude CSI tensors.
        labels : np.ndarray, shape [N]
            Integer class labels for the primary task.
        metadata : dict
            Per-sample metadata (sample_ids, env_ids, device_ids, user_ids)
            when available.
        """
        with h5py.File(self._h5_path, "r") as f:
            samples = np.array(f["csi_data"])
            labels = np.array(f["labels"])

            metadata: dict = {}
            for key in ("sample_ids", "env_ids", "device_ids", "user_ids"):
                if key in f:
                    metadata[key] = np.array(f[key])

        if samples.ndim == 3:
            # Single-channel: [N, K, T] -> [N, 1, K, T]
            samples = samples[:, np.newaxis, :, :]

        logger.info(
            "Loaded %d samples from CSI-Bench/%s/%s  shape=%s",
            len(samples), self.task, self.split, samples.shape,
        )
        return samples, labels, metadata

    def get_multitask_labels(self) -> dict[str, np.ndarray]:
        """For MT (multi-task) datasets, return all label sets.

        Returns
        -------
        dict mapping task name -> label array, e.g.
        {"har": labels, "user_id": labels, "proximity": labels}

        Raises
        ------
        ValueError
            If the current task does not belong to the MT dataset type.
        """
        if self.config["dataset_type"] != "MT":
            raise ValueError(
                f"Task '{self.task}' is dataset_type='{self.config['dataset_type']}', "
                "not 'MT'. Multi-task labels are only available for MT tasks."
            )

        mt_labels: dict[str, np.ndarray] = {}
        with h5py.File(self._h5_path, "r") as f:
            for mt_task in _MT_TASKS:
                key = f"labels_{mt_task}"
                if key in f:
                    mt_labels[mt_task] = np.array(f[key])
                elif mt_task == self.task and "labels" in f:
                    mt_labels[mt_task] = np.array(f["labels"])

        if not mt_labels:
            raise ValueError(
                f"No multi-task labels found in {self._h5_path}. "
                f"Expected keys: {[f'labels_{t}' for t in _MT_TASKS]}"
            )

        return mt_labels

    @property
    def num_classes(self) -> int:
        return self.config["num_classes"]

    @property
    def expected_time_steps(self) -> int:
        return self.config["window_sec"] * self.config["sample_rate"]

    def __repr__(self) -> str:
        return (
            f"CSIBenchLoader(task={self.task!r}, split={self.split!r}, "
            f"h5={self._h5_path})"
        )
