"""Generate Level-1-only pair registries with controlled template counts.

Produces 4 separate pair_registry.json files (1, 3, 5, 10 templates per
class) for the HAR template count ablation (Table 4d rerun).

For the 10-template variant, this script combines the 5 base templates
from task_templates.py with 5 additional templates from
har_extended_templates.py.  The 1/3/5 variants are subsets of the base 5.

Usage:
    python scripts/paper_experiments/generate_template_count_registries.py \
        --task_root data/Dataset/csibench/multitask/har \
        --output_dir data/generated_texts/ablation_template_count

Output layout:
    <output_dir>/1tpl/pair_registry.json
    <output_dir>/3tpl/pair_registry.json
    <output_dir>/5tpl/pair_registry.json
    <output_dir>/10tpl/pair_registry.json
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"
VARIANTS = [1, 3, 5, 10]


def _make_text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _load_base_templates():
    from data.text_generation.templates.task_templates import CSIBENCH_TEMPLATES
    return CSIBENCH_TEMPLATES["har"]


def _load_extended_templates():
    from data.text_generation.templates.har_extended_templates import HAR_EXTENDED_TEMPLATES
    return HAR_EXTENDED_TEMPLATES


def _validate_extended(extended):
    for entry in extended:
        for tpl in entry["templates"]:
            if tpl.startswith("TODO:"):
                raise ValueError(
                    f"Placeholder template found for class '{entry['class_name']}': "
                    f"\"{tpl}\". Replace all TODO placeholders in "
                    "data/text_generation/templates/har_extended_templates.py "
                    "before running this script."
                )


def _build_combined_templates(base, extended):
    """Merge base (5/class) and extended (5/class) into a single list of 10/class."""
    ext_map = {e["class_id"]: e["templates"] for e in extended}
    combined = []
    for entry in base:
        cid = entry["class_id"]
        merged = list(entry["templates"]) + list(ext_map.get(cid, []))
        combined.append({
            "class_name": entry["class_name"],
            "class_id": cid,
            "templates": merged,
        })
    return combined


def build_registry(
    templates_by_class: list[dict],
    n_templates: int,
    sample_ids: list[str],
    sample_labels: dict[str, int],
) -> dict:
    """Build a Level-1-only pair registry with exactly n_templates per class."""

    texts: dict[str, str] = {}
    class_to_texts: dict[str, list[str]] = {}
    classes: dict[str, dict[str, str]] = {}

    for entry in templates_by_class:
        cid = entry["class_id"]
        class_key = str(cid)
        classes[class_key] = {"name": entry["class_name"]}

        selected = entry["templates"][:n_templates]
        if len(selected) < n_templates:
            raise ValueError(
                f"Class '{entry['class_name']}' has {len(entry['templates'])} "
                f"templates, but {n_templates} were requested."
            )

        tids = []
        for text in selected:
            tid = _make_text_id(text)
            texts[tid] = text
            tids.append(tid)
        class_to_texts[class_key] = tids

    sample_to_texts: dict[str, list[str]] = {}
    sample_to_class: dict[str, str] = {}

    for sid in sample_ids:
        cid = sample_labels[sid]
        class_key = str(cid)
        sample_to_class[sid] = class_key
        sample_to_texts[sid] = list(class_to_texts[class_key])

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "csibench",
        "task": "har",
        "num_classes": len(classes),
        "classes": classes,
        "texts": texts,
        "sample_to_texts": sample_to_texts,
        "sample_to_class": sample_to_class,
        "class_to_texts": class_to_texts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate Level-1-only pair registries with controlled template counts."
    )
    parser.add_argument(
        "--task_root", type=str, required=True,
        help="HAR task root (e.g. .../csibench/multitask/har)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="data/generated_texts/ablation_template_count",
        help="Output directory for generated registries",
    )
    parser.add_argument(
        "--h5_key", type=str, default="CSI_amps",
    )
    parser.add_argument(
        "--variants", type=int, nargs="+", default=VARIANTS,
        help="Template count variants to generate (default: 1 3 5 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # --- Load templates ---
    base = _load_base_templates()
    needs_extended = any(v > 5 for v in args.variants)

    if needs_extended:
        extended = _load_extended_templates()
        _validate_extended(extended)
        combined = _build_combined_templates(base, extended)
    else:
        combined = base

    # --- Load sample IDs and labels ---
    from data.metadata_loader import MetadataLoader

    task_root = Path(args.task_root)
    lm_path = task_root / "metadata" / "label_mapping.json"
    with open(lm_path) as f:
        label_mapping = json.load(f)
    label_to_idx = {str(k): int(v) for k, v in label_mapping["label_to_idx"].items()}

    loader = MetadataLoader(task_root=str(task_root), h5_key=args.h5_key)
    all_ids = loader.get_all_sample_ids()

    sample_labels: dict[str, int] = {}
    for sid in all_ids:
        raw = str(loader.get_label(sid))
        if raw in label_to_idx:
            sample_labels[sid] = label_to_idx[raw]
        else:
            logger.warning("Unmapped label %r for sample %s — skipping", raw, sid)
    all_ids = [sid for sid in all_ids if sid in sample_labels]

    logger.info("Loaded %d samples from %s", len(all_ids), args.task_root)

    # --- Generate registries ---
    output_dir = Path(args.output_dir)
    for n_tpl in sorted(args.variants):
        registry = build_registry(combined, n_tpl, all_ids, sample_labels)
        out_path = output_dir / f"{n_tpl}tpl" / "pair_registry.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(registry, f, indent=2)

        n_texts = len(registry["texts"])
        n_pairs = sum(len(v) for v in registry["sample_to_texts"].values())
        logger.info(
            "Generated %dtpl: %d unique texts, %d samples, %d total pairs -> %s",
            n_tpl, n_texts, len(all_ids), n_pairs, out_path,
        )

    print(f"\nGenerated {len(args.variants)} registries under {output_dir}/")


if __name__ == "__main__":
    main()
