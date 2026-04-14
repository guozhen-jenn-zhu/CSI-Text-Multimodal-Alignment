"""
Text generation pipeline for CSI-Text multimodal alignment.

Generates semantic captions (Level 1), composite captions (Level 2),
and builds many-to-many CSI-text pair registries for training.

Registry output follows schema v2.0 with task-local class IDs.
"""

import json
import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from data.text_generation.templates.task_templates import (
    get_task_templates,
    generate_composite_caption,
    CSIBENCH_COMPOSITE_TEMPLATES,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"


# ============================================================================
# LLM Paraphraser
# ============================================================================

@dataclass
class LLMParaphraser:
    """Generates paraphrased variants of template texts via OpenAI API or local fallback."""

    model: str = "gpt-4o-mini"
    num_paraphrases: int = 5
    temperature: float = 0.8
    _client: Any = field(default=None, repr=False)

    def __post_init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI()
            logger.info(f"LLMParaphraser initialized with model={self.model}")
        except (ImportError, Exception) as e:
            logger.warning(f"OpenAI client unavailable ({e}). Paraphrasing disabled; using templates directly.")
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def paraphrase(self, text: str) -> List[str]:
        """Generate paraphrased variants of the input text."""
        if not self.available:
            return [text]

        prompt = (
            f"Generate {self.num_paraphrases} paraphrased versions of the following sentence. "
            f"Each paraphrase should preserve the original meaning but vary in wording and structure. "
            f"Return only the paraphrases, one per line, with no numbering or extra formatting.\n\n"
            f"Sentence: {text}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=512,
            )
            raw = response.choices[0].message.content.strip()
            paraphrases = [line.strip() for line in raw.split("\n") if line.strip()]
            return paraphrases if paraphrases else [text]
        except Exception as e:
            logger.error(f"Paraphrase API call failed: {e}")
            return [text]

    def paraphrase_batch(self, texts: List[str]) -> List[List[str]]:
        """Paraphrase a batch of texts."""
        return [self.paraphrase(t) for t in texts]


# ============================================================================
# Quality Filter
# ============================================================================

class QualityFilter:
    """Filters and validates generated texts with simple heuristics."""

    MIN_LENGTH = 10
    MAX_LENGTH = 300

    def __init__(self, min_length: int = MIN_LENGTH, max_length: int = MAX_LENGTH):
        self.min_length = min_length
        self.max_length = max_length

    def filter_texts(self, texts: List[str]) -> List[str]:
        """Apply all quality checks and return valid, deduplicated texts."""
        filtered = []
        seen = set()
        for text in texts:
            text = text.strip()
            if not self._is_valid(text):
                continue
            normalized = text.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            filtered.append(text)
        return filtered

    def _is_valid(self, text: str) -> bool:
        if not text:
            return False
        if len(text) < self.min_length:
            return False
        if len(text) > self.max_length:
            return False
        if not any(c.isalpha() for c in text):
            return False
        return True


# ============================================================================
# Text Generator
# ============================================================================

class TextGenerator:
    """Main pipeline for generating text descriptions aligned with CSI samples.

    Requires both ``dataset_name`` and ``task_name`` so that templates are
    scoped to a single task with task-local class IDs (schema v2).
    """

    SUPPORTED_DATASETS = {"csibench", "xrf55", "signfi"}

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        use_llm_paraphrase: bool = False,
        llm_model: str = "gpt-4o-mini",
        num_paraphrases: int = 5,
    ):
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from {self.SUPPORTED_DATASETS}")

        self.dataset_name = dataset_name
        self.task_name = task_name
        self.use_llm_paraphrase = use_llm_paraphrase
        self.quality_filter = QualityFilter()

        if use_llm_paraphrase:
            self.paraphraser = LLMParaphraser(model=llm_model, num_paraphrases=num_paraphrases)
        else:
            self.paraphraser = None

        self._templates = get_task_templates(dataset_name, task_name)
        self._class_map = {entry["class_id"]: entry for entry in self._templates}

    def generate_level1_texts(self) -> Dict[int, List[str]]:
        """Generate Level 1 semantic captions for all classes.

        Returns:
            Mapping of class_id -> list of text descriptions.
        """
        result: Dict[int, List[str]] = {}

        for entry in self._templates:
            class_id = entry["class_id"]
            base_texts = list(entry["templates"])

            if self.use_llm_paraphrase and self.paraphraser and self.paraphraser.available:
                augmented = []
                for text in base_texts:
                    augmented.append(text)
                    augmented.extend(self.paraphraser.paraphrase(text))
                base_texts = augmented

            result[class_id] = self.quality_filter.filter_texts(base_texts)

        logger.info(
            f"Generated Level 1 texts for {self.dataset_name}/{self.task_name}: "
            f"{len(result)} classes, {sum(len(v) for v in result.values())} total texts"
        )
        return result

    def generate_level2_texts(self, sample_metadata: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
        """Generate Level 2 composite captions for CSI-Bench Multitask samples.

        Each sample's metadata must contain: user_id, activity, proximity.

        Args:
            sample_metadata: {sample_id: {"user_id": ..., "activity": ..., "proximity": ...}}

        Returns:
            {sample_id: [composite_text_1, composite_text_2, ...]}
        """
        if self.dataset_name != "csibench":
            raise ValueError("Level 2 composite texts are only supported for the csibench dataset")

        num_patterns = len(CSIBENCH_COMPOSITE_TEMPLATES["template_patterns"])
        result: Dict[str, List[str]] = {}

        for sample_id, meta in sample_metadata.items():
            texts = []
            for tpl_idx in range(num_patterns):
                try:
                    caption = generate_composite_caption(
                        user_id=meta["user_id"],
                        activity=meta["activity"],
                        proximity=meta["proximity"],
                        template_idx=tpl_idx,
                    )
                    texts.append(caption)
                except KeyError:
                    continue

            if self.use_llm_paraphrase and self.paraphraser and self.paraphraser.available:
                augmented = list(texts)
                for text in texts:
                    augmented.extend(self.paraphraser.paraphrase(text))
                texts = augmented

            filtered = self.quality_filter.filter_texts(texts)
            if filtered:
                result[sample_id] = filtered

        logger.info(f"Generated Level 2 texts for {len(result)} samples")
        return result

    @staticmethod
    def _make_text_id(text: str) -> str:
        """Deterministic short hash for a text string."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def build_pair_registry(
        self,
        sample_ids: List[str],
        sample_labels: Dict[str, int],
        sample_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Build a many-to-many CSI-text pair registry (schema v2).

        Args:
            sample_ids: List of CSI sample identifiers.
            sample_labels: {sample_id: class_id} mapping (task-local 0-indexed).
            sample_metadata: Optional per-sample metadata for Level 2 texts.

        Returns:
            Registry dict with schema v2 fields.
        """
        level1_texts = self.generate_level1_texts()

        level2_texts: Dict[str, List[str]] = {}
        if sample_metadata and self.dataset_name == "csibench":
            level2_texts = self.generate_level2_texts(sample_metadata)

        texts: Dict[str, str] = {}
        sample_to_texts: Dict[str, List[str]] = {}
        sample_to_class: Dict[str, str] = {}
        class_to_texts: Dict[str, List[str]] = {}

        for class_id, class_texts_list in level1_texts.items():
            class_key = str(class_id)
            class_to_texts[class_key] = []
            for text in class_texts_list:
                text_id = self._make_text_id(text)
                texts[text_id] = text
                class_to_texts[class_key].append(text_id)

        for sample_id in sample_ids:
            class_id = sample_labels[sample_id]
            class_key = str(class_id)
            sample_to_class[sample_id] = class_key

            associated = list(level1_texts.get(class_id, []))
            if sample_id in level2_texts:
                associated.extend(level2_texts[sample_id])
            associated = self.quality_filter.filter_texts(associated)

            text_ids = []
            for text in associated:
                text_id = self._make_text_id(text)
                texts[text_id] = text
                text_ids.append(text_id)

                if text_id not in class_to_texts.get(class_key, []):
                    class_to_texts.setdefault(class_key, []).append(text_id)

            sample_to_texts[sample_id] = text_ids

        classes: Dict[str, Dict[str, str]] = {}
        for entry in self._templates:
            classes[str(entry["class_id"])] = {"name": entry["class_name"]}

        logger.info(
            f"Built pair registry: {len(sample_to_texts)} samples, "
            f"{len(texts)} unique texts, "
            f"{sum(len(v) for v in sample_to_texts.values())} total pairs"
        )

        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset_name,
            "task": self.task_name,
            "num_classes": len(classes),
            "classes": classes,
            "texts": texts,
            "sample_to_texts": sample_to_texts,
            "sample_to_class": sample_to_class,
            "class_to_texts": class_to_texts,
        }

    def save_registry(self, registry: Dict[str, Any], output_path: str) -> None:
        """Serialize pair registry to JSON."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Registry saved to {output_path}")

    def generate_all(
        self,
        sample_ids: List[str],
        sample_labels: Dict[str, int],
        sample_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full pipeline: generate texts, build registry, and optionally save.

        Returns:
            The pair registry dict.
        """
        registry = self.build_pair_registry(sample_ids, sample_labels, sample_metadata)

        if output_path:
            self.save_registry(registry, output_path)

        return registry


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate CSI-Text pair registry for a dataset/task."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["csibench", "xrf55", "signfi"],
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task name (e.g. har, fall_detection, action_recognition)",
    )
    parser.add_argument(
        "--task_root", type=str, required=True,
        help="Root directory of the task (containing metadata/ and splits/)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the pair registry JSON",
    )
    parser.add_argument(
        "--h5_key", type=str, default="CSI_amps",
        help="HDF5 dataset key (default: CSI_amps, use 'auto' for XRF55)",
    )
    parser.add_argument(
        "--label_mapping", type=str, default=None,
        help="Path to label_mapping.json (default: task_root/metadata/label_mapping.json)",
    )
    parser.add_argument("--use_llm", action="store_true", help="Enable LLM paraphrasing")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--num_paraphrases", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from pathlib import Path
    from data.metadata_loader import MetadataLoader

    label_map_path = args.label_mapping or str(Path(args.task_root) / "metadata" / "label_mapping.json")
    with open(label_map_path) as f:
        label_mapping = json.load(f)
    label_to_idx = label_mapping["label_to_idx"]

    loader = MetadataLoader(task_root=args.task_root, h5_key=args.h5_key)
    all_ids = loader.get_all_sample_ids()
    sample_labels = {}
    for sid in all_ids:
        raw = loader.get_label(sid)
        raw_key = str(raw)
        if raw_key in label_to_idx:
            sample_labels[sid] = label_to_idx[raw_key]
        else:
            logger.warning("Unmapped label %r for sample %s", raw, sid)
    all_ids = [sid for sid in all_ids if sid in sample_labels]

    sample_metadata = None
    if args.dataset == "csibench":
        sample_metadata = {}
        for sid in all_ids:
            meta = loader.get_metadata(sid)
            if all(k in meta for k in ("user", "activity", "distance")):
                user_raw = str(meta["user"])
                act_raw = str(meta["activity"])
                dist_raw = meta["distance"]
                user_key = f"user_{int(user_raw[1:])}" if user_raw.startswith("U") and user_raw[1:].isdigit() else user_raw
                act_map = {"seated-breathing": "seated_breathing", "wavinghand": "waving"}
                activity = act_map.get(act_raw, act_raw)
                proximity = f"{float(dist_raw)}m" if dist_raw is not None else None
                if user_key and activity and proximity:
                    sample_metadata[sid] = {
                        "user_id": user_key,
                        "activity": activity,
                        "proximity": proximity,
                    }

    gen = TextGenerator(
        dataset_name=args.dataset,
        task_name=args.task,
        use_llm_paraphrase=args.use_llm,
        llm_model=args.llm_model,
        num_paraphrases=args.num_paraphrases,
    )

    gen.generate_all(
        sample_ids=all_ids,
        sample_labels=sample_labels,
        sample_metadata=sample_metadata,
        output_path=args.output,
    )
    logger.info("Done. Registry saved to %s", args.output)


if __name__ == "__main__":
    main()
