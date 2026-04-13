"""XRF55 dataset loader.

XRF55 stores WiFi CSI as .npy files with naming convention
XX_XX_XX.npy (scene_subject_action). Raw format: 1000 x 1 x 3 x 3 x 30.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ACTION_CLASSES = [
    "applaud", "beckon", "blow_nose", "bow", "brush_teeth",
    "check_time", "clap_overhead", "comb_hair", "cross_arms", "crouch",
    "drink", "eat", "elbow", "fall", "flick_hair",
    "grab", "hand_circle", "hand_down", "hand_left", "hand_right",
    "hand_up", "handshake", "hit", "hug", "jump",
    "kick", "kneel", "lean_back", "lunge", "make_call",
    "nod", "pick_up", "point", "pull", "punch",
    "push", "rub_hands", "salute", "shake_fist", "shake_head",
    "sit_down", "squat", "stand_up", "stretch", "swipe_left",
    "swipe_right", "throw", "tiptoe", "turn_around", "twist",
    "walk", "wave_hand", "wipe_face", "yawn", "zoom_in",
]

_FILE_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)\.npy$")


class XRF55Loader:
    """Loader for XRF55 WiFi CSI dataset."""

    ACTION_CLASSES = ACTION_CLASSES

    def __init__(self, data_root: str, split: str = "train") -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.data_root = Path(data_root)
        self.split = split
        self._data_dir = self._resolve_data_dir()

    def _resolve_data_dir(self) -> Path:
        candidates = [
            self.data_root / "XRF_dataset" / "WiFi" / self.split,
            self.data_root / "Raw_dataset" / "WiFi" / self.split,
            self.data_root / "WiFi" / self.split,
            self.data_root / self.split,
        ]
        for p in candidates:
            if p.is_dir():
                return p

        raise FileNotFoundError(
            f"XRF55 WiFi data directory not found for split='{self.split}'. "
            f"Searched: {[str(c) for c in candidates]}"
        )

    def _parse_filename(self, path: Path) -> dict | None:
        m = _FILE_PATTERN.match(path.name)
        if m is None:
            return None
        return {
            "scene_id": int(m.group(1)),
            "subject_id": int(m.group(2)),
            "action_id": int(m.group(3)),
        }

    def _collect_files(self) -> list[tuple[Path, dict]]:
        npy_files = sorted(self._data_dir.rglob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No .npy files found under {self._data_dir}"
            )

        entries: list[tuple[Path, dict]] = []
        for f in npy_files:
            meta = self._parse_filename(f)
            if meta is not None:
                entries.append((f, meta))

        if not entries:
            raise FileNotFoundError(
                f"Found {len(npy_files)} .npy files but none matched the "
                "expected XX_XX_XX.npy naming convention."
            )
        return entries

    @staticmethod
    def _reshape_sample(raw: np.ndarray) -> np.ndarray:
        """Reshape raw XRF55 sample to [C, K, T].

        Input shape: (1000, 1, 3, 3, 30) -> T=1000, 1, Tx=3, Rx=3, K=30
        Output: [C=9, K=30, T=1000]  (Tx*Rx antenna pairs as channels)
        """
        if raw.shape == (1000, 1, 3, 3, 30):
            # (T, 1, Tx, Rx, K) -> (T, Tx, Rx, K) -> (T, C, K) -> (C, K, T)
            raw = raw.squeeze(1)  # (1000, 3, 3, 30)
            T, tx, rx, K = raw.shape
            raw = raw.reshape(T, tx * rx, K)  # (1000, 9, 30)
            return raw.transpose(1, 2, 0)  # (9, 30, 1000)

        if raw.ndim == 3:
            # Already (T, K, C) or similar — assume (T, K, C)
            return raw.transpose(2, 1, 0)

        raise ValueError(
            f"Unexpected XRF55 sample shape: {raw.shape}. "
            "Expected (1000, 1, 3, 3, 30)."
        )

    def load_samples(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Load WiFi CSI samples and labels.

        Returns
        -------
        samples : np.ndarray, shape [N, C, K, T]
        labels : np.ndarray, shape [N]
            Action class index (0-indexed).
        metadata : dict
            scene_ids, subject_ids arrays.
        """
        entries = self._collect_files()

        samples_list: list[np.ndarray] = []
        labels: list[int] = []
        scene_ids: list[int] = []
        subject_ids: list[int] = []

        for fpath, meta in entries:
            raw = np.load(fpath)
            csi = self._reshape_sample(np.abs(raw) if np.iscomplexobj(raw) else raw)
            samples_list.append(csi)
            labels.append(meta["action_id"])
            scene_ids.append(meta["scene_id"])
            subject_ids.append(meta["subject_id"])

        samples = np.stack(samples_list, axis=0)
        labels_arr = np.array(labels, dtype=np.int64)

        metadata = {
            "scene_ids": np.array(scene_ids, dtype=np.int64),
            "subject_ids": np.array(subject_ids, dtype=np.int64),
        }

        logger.info(
            "Loaded %d samples from XRF55/%s  shape=%s  classes=%d",
            len(samples), self.split, samples.shape, len(set(labels)),
        )
        return samples, labels_arr, metadata

    @property
    def num_classes(self) -> int:
        return len(ACTION_CLASSES)

    def __repr__(self) -> str:
        return f"XRF55Loader(split={self.split!r}, data_dir={self._data_dir})"
