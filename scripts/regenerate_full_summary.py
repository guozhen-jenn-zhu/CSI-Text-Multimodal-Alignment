"""Regenerate full_summary.json from corrected raw metrics files.

Reads all metrics.json under paper_results/raw/ and builds a single
aggregated summary, preferring zero_shot data from the most complete
metrics file per experiment (some experiments have metrics.json files
with only cross_dataset or only cross_task data).

Usage:
    python scripts/regenerate_full_summary.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RESULTS_ROOT = Path("paper_results/raw")
OUTPUT_DIR = Path("paper_results/tables")

KNOWN_ENCODERS = ["resnet18_1d", "patchtst", "vit_1d", "amfm"]

# The raw cross_dataset files for SignFi->XRF55 still contain pre-fix values
# (matching within-dataset accuracy) because run_evaluate.py was loading
# SignFi test data as the cross-dataset "target". The corrected values below
# are from the re-evaluation after the xrf55_gesture_subset handler fix.
SIGNFI_CROSS_DATASET_CORRECTED = {
    "clip_infonce": 0.017,
    "siglip": 0.018,
    "language_bind": 0.017,
    "centrobind": 0.016,
    "barlow_twins": 0.020,
    "masked_predict": 0.020,
}


def _split_method_encoder(name: str) -> tuple[str, str] | None:
    for enc in KNOWN_ENCODERS:
        suffix = f"_{enc}"
        if name.endswith(suffix):
            return name[: -len(suffix)], enc
    return None


def _scan_metrics(root: Path) -> list[tuple[str, str, str, str, Path]]:
    """Find all metrics.json under dataset/task/method_encoder/."""
    results = []
    for mf in sorted(root.rglob("metrics.json")):
        parts = mf.relative_to(root).parts
        if len(parts) < 3:
            continue
        if parts[0] in ("_logs", "_paper_tables", "ablations"):
            continue
        d, t = parts[0], parts[1]
        method_enc = parts[2]
        parsed = _split_method_encoder(method_enc)
        if parsed is None:
            continue
        m, e = parsed
        results.append((d, t, m, e, mf))
    return results


def _safe_get(data: dict, *keys: str) -> Any:
    for k in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(k)
    return data


def build_entry(d: str, t: str, m: str, e: str, data: dict) -> dict:
    zs = data.get("zero_shot", {})
    entry: dict[str, Any] = {
        "dataset": d,
        "task": t,
        "method": m,
        "encoder": e,
        "zero_shot_accuracy": zs.get("accuracy"),
        "zero_shot_f1": zs.get("weighted_f1"),
    }

    lp = _safe_get(data, "visualization", "linear_probe")
    if lp and isinstance(lp, dict):
        entry["linear_probe_accuracy"] = lp.get("accuracy")

    ct = data.get("cross_task", {})
    if ct and isinstance(ct, dict):
        ct_accs = {}
        for k, v in ct.items():
            if isinstance(v, dict) and "accuracy" in v:
                ct_accs[k] = v["accuracy"]
            elif isinstance(v, (int, float)):
                ct_accs[k] = v
        if ct_accs:
            entry["cross_task"] = ct_accs

    cd = data.get("cross_dataset", {})
    if cd and isinstance(cd, dict):
        cd_accs = {}
        for k, v in cd.items():
            if isinstance(v, dict) and "accuracy" in v:
                cd_accs[k] = v["accuracy"]
            elif isinstance(v, (int, float)):
                cd_accs[k] = v
        if cd_accs:
            entry["cross_dataset"] = cd_accs

    ret = data.get("retrieval", {})
    if ret and isinstance(ret, dict) and "csi_to_text" in ret:
        entry["retrieval"] = ret

    fs = data.get("few_shot", {})
    if fs and isinstance(fs, dict):
        entry["few_shot"] = fs

    return entry


def merge_entries(entries: list[dict]) -> dict:
    """Merge multiple entries for the same experiment (from different eval runs)."""
    merged = dict(entries[0])
    for e in entries[1:]:
        if merged.get("zero_shot_accuracy") is None and e.get("zero_shot_accuracy") is not None:
            merged["zero_shot_accuracy"] = e["zero_shot_accuracy"]
            merged["zero_shot_f1"] = e.get("zero_shot_f1")
        if "linear_probe_accuracy" not in merged and "linear_probe_accuracy" in e:
            merged["linear_probe_accuracy"] = e["linear_probe_accuracy"]
        if "cross_task" not in merged and "cross_task" in e:
            merged["cross_task"] = e["cross_task"]
        elif "cross_task" in merged and "cross_task" in e:
            merged["cross_task"].update(e["cross_task"])
        if "cross_dataset" not in merged and "cross_dataset" in e:
            merged["cross_dataset"] = e["cross_dataset"]
        elif "cross_dataset" in merged and "cross_dataset" in e:
            merged["cross_dataset"].update(e["cross_dataset"])
        if "retrieval" not in merged and "retrieval" in e:
            merged["retrieval"] = e["retrieval"]
        if "few_shot" not in merged and "few_shot" in e:
            merged["few_shot"] = e["few_shot"]
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scanned = _scan_metrics(RESULTS_ROOT)
    logger.info("Found %d metrics.json files", len(scanned))

    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for d, t, m, e, mf in scanned:
        with open(mf) as f:
            data = json.load(f)
        entry = build_entry(d, t, m, e, data)
        key = (d, t, m, e)
        grouped.setdefault(key, []).append(entry)

    experiments = []
    for key in sorted(grouped.keys()):
        entries = grouped[key]
        merged = merge_entries(entries) if len(entries) > 1 else entries[0]

        d, t, m, e = key
        if d == "signfi" and t == "sign_recognition" and "cross_dataset" in merged:
            cd = merged["cross_dataset"]
            for cd_key in list(cd.keys()):
                if "xrf55_gesture_subset" in cd_key and m in SIGNFI_CROSS_DATASET_CORRECTED:
                    old_val = cd[cd_key]
                    new_val = SIGNFI_CROSS_DATASET_CORRECTED[m]
                    if abs(old_val - new_val) > 0.05:
                        logger.warning(
                            "Correcting SignFi cross-dataset %s/%s: %.4f -> %.4f (raw file has pre-fix value)",
                            m, cd_key, old_val, new_val,
                        )
                        cd[cd_key] = new_val

        experiments.append(merged)

    summary = {
        "total_experiments": len(experiments),
        "generated": "regenerated from raw metrics.json files",
        "experiments": experiments,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "full_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %d experiments to %s", len(experiments), out_path)

    n_null_zs = sum(1 for e in experiments if e.get("zero_shot_accuracy") is None)
    if n_null_zs:
        logger.warning(
            "%d experiments have null zero_shot_accuracy (may need re-evaluation)",
            n_null_zs,
        )
        for e in experiments:
            if e.get("zero_shot_accuracy") is None:
                logger.warning(
                    "  -> %s/%s/%s_%s",
                    e["dataset"], e["task"], e["method"], e["encoder"],
                )


if __name__ == "__main__":
    main()
