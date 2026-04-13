"""Many-to-many CSI-Text pair registry management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PairRegistry:
    """Manages many-to-many CSI-text pair relationships.

    Storage format:
    {
        "texts": {text_id: text_string},
        "sample_to_texts": {sample_id: [text_id, ...]},
        "text_to_samples": {text_id: [sample_id, ...]},
        "class_to_texts": {class_id: [text_id, ...]},
        "class_to_samples": {class_id: [sample_id, ...]},
        "sample_to_class": {sample_id: class_id}
    }
    """

    def __init__(self) -> None:
        self.texts: dict[str, str] = {}
        self.sample_to_texts: dict[str, list[str]] = {}
        self.text_to_samples: dict[str, list[str]] = {}
        self.class_to_texts: dict[str, list[str]] = {}
        self.class_to_samples: dict[str, list[str]] = {}
        self.sample_to_class: dict[str, str] = {}

        self._text_id_list: list[str] | None = None
        self._sample_id_list: list[str] | None = None

    def _invalidate_cache(self) -> None:
        self._text_id_list = None
        self._sample_id_list = None

    def add_text(self, text_id: str, text: str, class_id: str) -> None:
        """Register a text string under the given id and class."""
        self._invalidate_cache()
        self.texts[text_id] = text
        self.class_to_texts.setdefault(class_id, [])
        if text_id not in self.class_to_texts[class_id]:
            self.class_to_texts[class_id].append(text_id)

    def add_sample(
        self,
        sample_id: str,
        text_ids: list[str],
        class_id: str,
    ) -> None:
        """Register a CSI sample with its associated text ids and class."""
        self._invalidate_cache()
        self.sample_to_texts[sample_id] = list(text_ids)
        self.sample_to_class[sample_id] = class_id

        self.class_to_samples.setdefault(class_id, [])
        if sample_id not in self.class_to_samples[class_id]:
            self.class_to_samples[class_id].append(sample_id)

        for tid in text_ids:
            self.text_to_samples.setdefault(tid, [])
            if sample_id not in self.text_to_samples[tid]:
                self.text_to_samples[tid].append(sample_id)

    def get_positive_texts(self, sample_id: str) -> list[str]:
        """Return all text ids associated with *sample_id*."""
        return list(self.sample_to_texts.get(sample_id, []))

    def get_positive_samples(self, text_id: str) -> list[str]:
        """Return all sample ids associated with *text_id*."""
        return list(self.text_to_samples.get(text_id, []))

    def sample_positive_text(self, sample_id: str, rng: np.random.Generator | None = None) -> str:
        """Randomly sample one positive text id for a CSI sample."""
        candidates = self.get_positive_texts(sample_id)
        if not candidates:
            raise ValueError(f"No positive texts for sample {sample_id}")
        rng = rng or np.random.default_rng()
        return str(rng.choice(candidates))

    def sample_negative_text(
        self,
        sample_id: str,
        rng: np.random.Generator | None = None,
    ) -> str:
        """Sample a text id that is NOT associated with this sample's class."""
        sample_class = self.sample_to_class.get(sample_id)
        if sample_class is None:
            raise ValueError(f"Unknown sample {sample_id}")

        positive_set = set(self.class_to_texts.get(sample_class, []))
        all_text_ids = self.get_all_text_ids()
        negatives = [tid for tid in all_text_ids if tid not in positive_set]

        if not negatives:
            raise ValueError(f"No negative texts available for sample {sample_id}")

        rng = rng or np.random.default_rng()
        return str(rng.choice(negatives))

    def get_class_for_sample(self, sample_id: str) -> str | None:
        """Return the class id for a sample, or None if unknown."""
        return self.sample_to_class.get(sample_id)

    def get_all_text_ids(self) -> list[str]:
        if self._text_id_list is None:
            self._text_id_list = list(self.texts.keys())
        return self._text_id_list

    def get_all_sample_ids(self) -> list[str]:
        if self._sample_id_list is None:
            self._sample_id_list = list(self.sample_to_texts.keys())
        return self._sample_id_list

    def stats(self) -> dict[str, int]:
        return {
            "num_texts": len(self.texts),
            "num_samples": len(self.sample_to_texts),
            "num_classes": len(self.class_to_samples),
            "total_pairs": sum(len(v) for v in self.sample_to_texts.values()),
            "avg_texts_per_sample": (
                sum(len(v) for v in self.sample_to_texts.values()) / max(len(self.sample_to_texts), 1)
            ),
            "avg_samples_per_text": (
                sum(len(v) for v in self.text_to_samples.values()) / max(len(self.text_to_samples), 1)
            ),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "texts": self.texts,
            "sample_to_texts": self.sample_to_texts,
            "text_to_samples": self.text_to_samples,
            "class_to_texts": self.class_to_texts,
            "class_to_samples": self.class_to_samples,
            "sample_to_class": self.sample_to_class,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("PairRegistry saved to %s (%s)", path, self.stats())

    @classmethod
    def load(cls, path: str | Path) -> PairRegistry:
        with open(path) as f:
            data = json.load(f)

        registry = cls()
        registry.texts = data["texts"]
        registry.sample_to_texts = data["sample_to_texts"]
        registry.text_to_samples = data["text_to_samples"]
        registry.class_to_texts = data.get("class_to_texts", {})
        registry.class_to_samples = data.get("class_to_samples", {})
        registry.sample_to_class = data.get("sample_to_class", {})

        logger.info("PairRegistry loaded from %s (%s)", path, registry.stats())
        return registry

    @classmethod
    def from_generator_output(cls, registry_dict: dict) -> PairRegistry:
        """Build a PairRegistry from the dict returned by TextGenerator.build_pair_registry.

        That dict has keys: texts, sample_to_texts, text_to_samples.
        Class-level indices are rebuilt from sample_labels when available.
        """
        reg = cls()
        reg.texts = dict(registry_dict["texts"])
        reg.sample_to_texts = {
            k: list(v) for k, v in registry_dict["sample_to_texts"].items()
        }
        reg.text_to_samples = {
            k: list(v) for k, v in registry_dict["text_to_samples"].items()
        }
        return reg

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"PairRegistry(samples={s['num_samples']}, texts={s['num_texts']}, "
            f"classes={s['num_classes']}, pairs={s['total_pairs']})"
        )
