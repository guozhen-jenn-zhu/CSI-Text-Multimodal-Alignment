"""Automatically patch the paper draft with new supervised baselines.

Run after training fall_detection and motion_source supervised baselines:
    python scripts/patch_paper_supervised.py

Reads supervised_results.json and rewrites the paper draft sections to
remove Yang et al. caveats and insert the new controlled numbers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "paper_results" / "supervised_baselines"
DRAFT_DIR = REPO_ROOT / "paper" / "draft"

ZS_BEST = {
    "har": (98.7, "CLIP-InfoNCE"),
    "fall_detection": (95.9, "SigLIP"),
    "motion_source": (99.6, "CLIP-InfoNCE"),
    "proximity": (73.6, "CLIP-InfoNCE"),
    "user_id": (100.0, "SigLIP"),
    "xrf55": (78.1, "CLIP-InfoNCE"),
    "signfi": (95.8, "CLIP-InfoNCE"),
}

SUPERVISED = {}


def load_results() -> bool:
    for task in ["csibench_fall_detection", "csibench_motion_source"]:
        path = RESULTS_DIR / task / "supervised_results.json"
        if not path.is_file():
            print(f"ERROR: {path} not found. Run training first.")
            return False
        with open(path) as f:
            data = json.load(f)
        key = task.replace("csibench_", "")
        SUPERVISED[key] = round(data["test_accuracy_pct"], 1)

    for task, path_name in [
        ("har", "csibench_har"),
        ("proximity", "csibench_proximity"),
        ("user_id", "csibench_user_id"),
        ("xrf55", "xrf55_action_recognition"),
        ("signfi", "signfi_sign_recognition"),
    ]:
        path = RESULTS_DIR / path_name / "supervised_results.json"
        with open(path) as f:
            data = json.load(f)
        SUPERVISED[task] = round(data["test_accuracy_pct"], 1)

    return True


def count_zs_wins() -> tuple[int, list[str]]:
    wins = []
    for task in ["har", "fall_detection", "motion_source", "proximity", "user_id", "xrf55", "signfi"]:
        zs_acc, _ = ZS_BEST[task]
        sup_acc = SUPERVISED[task]
        if zs_acc >= sup_acc:
            wins.append(task)
    return len(wins), wins


def patch_experiments() -> None:
    path = DRAFT_DIR / "06_experiments.md"
    text = path.read_text()

    fall_sup = SUPERVISED["fall_detection"]
    motion_sup = SUPERVISED["motion_source"]

    text = text.replace("*94.9$^\\dagger$*", f"*{fall_sup}*")
    text = text.replace("*99.6$^\\dagger$*", f"*{motion_sup}*")

    old_setup = (
        "**Supervised baselines.** We compare against task-specific supervised "
        "ResNet-18 classifiers trained with cross-entropy loss on categorical labels. "
        "For five tasks (HAR, proximity, user ID, XRF55, SignFi), we train supervised "
        "baselines using the same ResNet-18 1D architecture, optimizer, and data splits "
        "as the alignment models, ensuring a fair apples-to-apples comparison. For fall "
        "detection and motion source recognition, we report supervised results from "
        "CSI-Bench [Yang et al., 2025]$^\\dagger$, which used the same ResNet-18 "
        "architecture but potentially different hyperparameters. The supervised accuracy "
        "per task is reported in Table 1."
    )
    new_setup = (
        "**Supervised baselines.** We compare against task-specific supervised "
        "ResNet-18 classifiers trained on the same data splits with full label "
        "supervision. All supervised baselines use the same ResNet-18 1D architecture, "
        "optimizer, and hyperparameters as the alignment models, trained with "
        "cross-entropy loss on categorical labels. The supervised accuracy per task "
        "is reported in Table 1."
    )
    text = text.replace(old_setup, new_setup)

    n_wins, win_tasks = count_zs_wins()

    task_descs = []
    for t in win_tasks:
        zs_acc, zs_method = ZS_BEST[t]
        sup_acc = SUPERVISED[t]
        name = {
            "har": "HAR", "fall_detection": "fall detection",
            "motion_source": "motion source", "proximity": "proximity",
            "user_id": "user identification", "xrf55": "XRF55", "signfi": "SignFi",
        }[t]
        if zs_acc > sup_acc:
            task_descs.append(f"{zs_method} achieves {zs_acc}% on {name} versus {sup_acc}% supervised (+{zs_acc - sup_acc:.1f}pp)")
        else:
            task_descs.append(f"{name} ties at {zs_acc}%")

    print(f"\nZero-shot wins/ties on {n_wins} of 7 tasks: {win_tasks}")
    print(f"Task descriptions for Finding 1:\n" + "\n".join(f"  - {d}" for d in task_descs))
    print(f"\nNew supervised numbers: Fall={fall_sup}%, Motion Source={motion_sup}%")
    print(f"\nThe experiments file has been patched with the new Table 1 numbers")
    print(f"and the uniform supervised baseline description.")
    print(f"\n*** IMPORTANT: You must manually update the Finding 1 narrative ***")
    print(f"*** paragraph in 06_experiments.md to reflect the new win count  ***")
    print(f"*** ({n_wins} of 7) and remove the Yang et al. caveats.          ***")

    path.write_text(text)


def patch_protocols() -> None:
    path = DRAFT_DIR / "05_protocols.md"
    text = path.read_text()

    old = (
        "**Baselines.** We compare against: (1) task-specific supervised ResNet-18 "
        "classifiers (trained by us on the same data splits for five tasks; from "
        "Yang et al. [2025] for fall detection and motion source), (2) random "
        "chance, and (3) majority class."
    )
    new = (
        "**Baselines.** We compare against: (1) task-specific supervised ResNet-18 "
        "classifiers trained on the same data splits with full label supervision, "
        "using the same encoder architecture as the alignment models, (2) random "
        "chance, and (3) majority class."
    )
    text = text.replace(old, new)
    path.write_text(text)


def main() -> None:
    if not load_results():
        sys.exit(1)

    patch_experiments()
    patch_protocols()

    print("\nDone. Files patched:")
    print(f"  {DRAFT_DIR / '06_experiments.md'}")
    print(f"  {DRAFT_DIR / '05_protocols.md'}")
    print(f"\nRemaining manual updates needed:")
    print(f"  - 06_experiments.md: Finding 1 narrative paragraph")
    print(f"  - 07_discussion.md: 'What Works' paragraph")
    print(f"  - 08_conclusion.md: Finding 1 bullet")


if __name__ == "__main__":
    main()
