"""Unified CSI preprocessing pipeline.

Normalizes CSI tensors from different datasets (CSI-Bench, XRF55, SignFi)
to a common [C, K_target, T_target] format via interpolation and
per-sample amplitude normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class CSIDatasetLoader(Protocol):
    """Structural type expected by UnifiedPreprocessor.process_dataset."""

    def load_samples(self) -> tuple[np.ndarray, np.ndarray, dict]: ...


class UnifiedPreprocessor:
    """Normalize CSI tensors to a common shape across datasets."""

    def __init__(
        self,
        target_subcarriers: int = 64,
        target_time_steps: int = 500,
    ) -> None:
        self.target_subcarriers = target_subcarriers
        self.target_time_steps = target_time_steps

    def normalize(self, csi: np.ndarray, source_dataset: str) -> np.ndarray:
        """Normalize a CSI tensor to [C, target_subcarriers, target_time_steps].

        Parameters
        ----------
        csi : np.ndarray
            Single CSI sample. Accepted shapes:
              - [C, K, T]  (canonical)
              - [T, K, C]  (time-first, e.g. SignFi raw)
              - [K, T]     (single-channel)
        source_dataset : str
            One of "csibench", "xrf55", "signfi". Used to select the
            appropriate reshape strategy when input is ambiguous.

        Returns
        -------
        np.ndarray, shape [C, target_subcarriers, target_time_steps]
        """
        csi = self._ensure_ckt(csi, source_dataset)
        csi = self._interpolate_subcarriers(csi)
        csi = self._interpolate_time(csi)
        csi = self._amplitude_normalize(csi)
        return csi

    def normalize_batch(
        self, batch: np.ndarray, source_dataset: str
    ) -> np.ndarray:
        """Normalize a batch of CSI tensors [N, ...]."""
        return np.stack(
            [self.normalize(batch[i], source_dataset) for i in range(len(batch))]
        )

    def process_dataset(
        self,
        loader: CSIDatasetLoader,
        output_path: str,
        source_dataset: str | None = None,
    ) -> Path:
        """Process an entire dataset and save normalized HDF5.

        Parameters
        ----------
        loader : CSIDatasetLoader
            Any loader with a ``load_samples()`` method.
        output_path : str
            Destination path for the output HDF5 file.
        source_dataset : str, optional
            Dataset identifier for reshape logic. If None, inferred from
            the loader class name.

        Returns
        -------
        Path to the written HDF5 file.
        """
        if source_dataset is None:
            source_dataset = self._infer_source(loader)

        samples, labels, metadata = loader.load_samples()

        logger.info(
            "Processing %d samples (source=%s) -> [C, %d, %d]",
            len(samples), source_dataset,
            self.target_subcarriers, self.target_time_steps,
        )

        normalized = self.normalize_batch(samples, source_dataset)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(out, "w") as f:
            f.create_dataset(
                "csi_data", data=normalized, dtype=np.float32,
                compression="gzip", compression_opts=4,
            )
            f.create_dataset("labels", data=labels)
            f.attrs["source_dataset"] = source_dataset
            f.attrs["target_subcarriers"] = self.target_subcarriers
            f.attrs["target_time_steps"] = self.target_time_steps

            for key, val in metadata.items():
                if isinstance(val, np.ndarray):
                    f.create_dataset(f"metadata/{key}", data=val)
                else:
                    f.attrs[key] = val

        logger.info("Saved normalized dataset to %s  shape=%s", out, normalized.shape)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_ckt(csi: np.ndarray, source: str) -> np.ndarray:
        """Reshape to [C, K, T] from various input formats."""
        if csi.ndim == 2:
            # [K, T] -> [1, K, T]
            return csi[np.newaxis]

        if csi.ndim == 3:
            # Disambiguation heuristic based on known dataset shapes
            d0, d1, d2 = csi.shape
            if source == "signfi" and d0 > d1 and d0 > d2:
                # [T=200, K=30, C=3] -> [C, K, T]
                return csi.transpose(2, 1, 0)
            # Default: already [C, K, T]
            return csi

        raise ValueError(
            f"Cannot reshape CSI with ndim={csi.ndim} (shape={csi.shape})"
        )

    def _interpolate_subcarriers(self, csi: np.ndarray) -> np.ndarray:
        """Interpolate subcarrier axis to target_subcarriers."""
        C, K, T = csi.shape
        if K == self.target_subcarriers:
            return csi

        x_old = np.linspace(0, 1, K)
        x_new = np.linspace(0, 1, self.target_subcarriers)
        out = np.empty((C, self.target_subcarriers, T), dtype=csi.dtype)

        for c in range(C):
            f = interp1d(x_old, csi[c], axis=0, kind="linear")
            out[c] = f(x_new)

        return out

    def _interpolate_time(self, csi: np.ndarray) -> np.ndarray:
        """Interpolate time axis to target_time_steps."""
        C, K, T = csi.shape
        if T == self.target_time_steps:
            return csi

        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, self.target_time_steps)
        out = np.empty((C, K, self.target_time_steps), dtype=csi.dtype)

        for c in range(C):
            f = interp1d(x_old, csi[c], axis=1, kind="linear")
            out[c] = f(x_new)

        return out

    @staticmethod
    def _amplitude_normalize(csi: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance normalization per sample."""
        mean = csi.mean()
        std = csi.std()
        if std < 1e-8:
            return csi - mean
        return (csi - mean) / std

    @staticmethod
    def _infer_source(loader: object) -> str:
        name = type(loader).__name__.lower()
        if "csibench" in name:
            return "csibench"
        if "xrf" in name:
            return "xrf55"
        if "signfi" in name or "sign" in name:
            return "signfi"
        return "unknown"

    def __repr__(self) -> str:
        return (
            f"UnifiedPreprocessor(target_subcarriers={self.target_subcarriers}, "
            f"target_time_steps={self.target_time_steps})"
        )
