"""SignFi dataset loader.

SignFi stores data in .mat files. CSI format: 200 x 30 x 3
(time x subcarriers x antennas). Complex-valued; we extract amplitude.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.io as sio

logger = logging.getLogger(__name__)

_VARIANT_CONFIG = {
    "lab_276": {
        "num_classes": 276,
        "mat_name": "lab",
        "description": "Lab environment, 276 sign gestures",
    },
    "home_276": {
        "num_classes": 276,
        "mat_name": "home",
        "description": "Home environment, 276 sign gestures",
    },
    "lab_150": {
        "num_classes": 150,
        "mat_name": "lab_150",
        "description": "Lab environment, 150 sign gestures",
    },
}


class SignFiLoader:
    """Loader for SignFi .mat datasets."""

    def __init__(
        self, data_root: str, variant: str = "lab_276", split: str = "train"
    ) -> None:
        if variant not in _VARIANT_CONFIG:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {list(_VARIANT_CONFIG.keys())}"
            )
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.data_root = Path(data_root)
        self.variant = variant
        self.split = split
        self.config = _VARIANT_CONFIG[variant]

        self._mat_path = self._resolve_mat_path()

    def _resolve_mat_path(self) -> Path:
        mat_name = self.config["mat_name"]
        candidates = [
            self.data_root / f"{mat_name}_{self.split}.mat",
            self.data_root / f"{mat_name}.mat",
            self.data_root / self.variant / f"{self.split}.mat",
            self.data_root / f"SignFi_{mat_name}.mat",
            self.data_root / "dataset" / f"{mat_name}_{self.split}.mat",
        ]
        for p in candidates:
            if p.is_file():
                return p

        raise FileNotFoundError(
            f"SignFi .mat file not found for variant='{self.variant}', "
            f"split='{self.split}'. Searched: {[str(c) for c in candidates]}"
        )

    @staticmethod
    def _extract_amplitude(csi: np.ndarray) -> np.ndarray:
        """Extract amplitude from potentially complex CSI data."""
        if np.iscomplexobj(csi):
            return np.abs(csi)
        return csi

    def _load_mat(self) -> tuple[np.ndarray, np.ndarray]:
        """Load raw data and labels from .mat file.

        Handles both MATLAB v5 (scipy) and v7.3 (h5py) formats.
        """
        try:
            mat = sio.loadmat(str(self._mat_path), squeeze_me=True)
            return self._parse_scipy_mat(mat)
        except NotImplementedError:
            return self._parse_h5_mat()

    def _parse_scipy_mat(self, mat: dict) -> tuple[np.ndarray, np.ndarray]:
        data_key = self._find_key(mat, ["csid", "csi_data", "data", "CSIdata"])
        label_key = self._find_key(mat, ["label", "labels", "csil", "Label"])

        if data_key is None or label_key is None:
            available = [k for k in mat.keys() if not k.startswith("_")]
            raise ValueError(
                f"Cannot find CSI data/label keys in {self._mat_path}. "
                f"Available keys: {available}"
            )

        return np.array(mat[data_key]), np.array(mat[label_key])

    def _parse_h5_mat(self) -> tuple[np.ndarray, np.ndarray]:
        """Fallback for MATLAB v7.3+ files using h5py."""
        import h5py

        with h5py.File(self._mat_path, "r") as f:
            data_key = self._find_key(
                f, ["csid", "csi_data", "data", "CSIdata"]
            )
            label_key = self._find_key(
                f, ["label", "labels", "csil", "Label"]
            )
            if data_key is None or label_key is None:
                raise ValueError(
                    f"Cannot find CSI data/label keys in {self._mat_path}. "
                    f"Available keys: {list(f.keys())}"
                )
            data = np.array(f[data_key])
            labels = np.array(f[label_key]).flatten()

        return data, labels

    @staticmethod
    def _find_key(container: dict, candidates: list[str]) -> str | None:
        for key in candidates:
            if key in container:
                return key
        return None

    def _reshape_to_ckt(self, raw: np.ndarray) -> np.ndarray:
        """Reshape raw CSI samples to [N, C, K, T].

        Input can be:
          - [N, T, K, C]  (200 x 30 x 3)  per-sample -> batch [N, 200, 30, 3]
          - [N, T*K*C]    flattened
        """
        if raw.ndim == 4:
            N, T, K, C = raw.shape
            return raw.transpose(0, 3, 2, 1)  # [N, C, K, T]

        if raw.ndim == 3:
            # Single sample [T, K, C] — shouldn't happen for batch, but handle it
            T, K, C = raw.shape
            return raw.transpose(2, 1, 0)[np.newaxis]  # [1, C, K, T]

        if raw.ndim == 2:
            N, flat = raw.shape
            # 200*30*3 = 18000
            if flat == 18000:
                raw = raw.reshape(N, 200, 30, 3)
                return raw.transpose(0, 3, 2, 1)

        raise ValueError(
            f"Unexpected SignFi data shape: {raw.shape}. "
            "Expected [N, 200, 30, 3] or [N, 18000]."
        )

    def load_samples(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load CSI samples (amplitude-extracted) and labels.

        Returns
        -------
        samples : np.ndarray, shape [N, C, K, T]
            C=3 (antennas), K=30 (subcarriers), T=200 (time steps).
        labels : np.ndarray, shape [N]
            Sign gesture class index.
        metadata : dict
            Contains 'variant', 'num_classes', and 'sign_words' (if available).
        """
        raw_data, raw_labels = self._load_mat()

        amplitude = self._extract_amplitude(raw_data)
        samples = self._reshape_to_ckt(amplitude)

        labels = raw_labels.flatten().astype(np.int64)
        if labels.min() == 1:
            labels = labels - 1

        metadata: dict = {
            "variant": self.variant,
            "num_classes": self.config["num_classes"],
        }

        if self.split == "train":
            n_train = samples.shape[0]
            metadata["split_size"] = n_train

        logger.info(
            "Loaded %d samples from SignFi/%s/%s  shape=%s  classes=%d",
            len(samples), self.variant, self.split,
            samples.shape, len(np.unique(labels)),
        )
        return samples, labels, metadata

    @property
    def num_classes(self) -> int:
        return self.config["num_classes"]

    def __repr__(self) -> str:
        return (
            f"SignFiLoader(variant={self.variant!r}, split={self.split!r}, "
            f"mat={self._mat_path})"
        )
