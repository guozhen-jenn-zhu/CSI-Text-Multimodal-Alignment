"""Read new supervised baseline results and print paper update instructions.

Run after training fall_detection and motion_source supervised baselines:
    python scripts/update_paper_supervised.py

This reads the supervised_results.json files and prints the exact values
to update in the paper draft, plus the sed-style changes needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "paper_results" / "supervised_baselines"

TASKS = {
    "csibench_fall_detection": {
        "label": "Fall Detection",
        "old_value": "94.9",
        "old_source": "Yang et al. 2025",
        "zs_best": 95.9,
        "zs_method": "SigLIP",
    },
    "csibench_motion_source": {
        "label": "Motion Source",
        "old_value": "99.6",
        "old_source": "Yang et al. 2025",
        "zs_best": 99.6,
        "zs_method": "CLIP-InfoNCE / LanguageBind",
    },
}


def main() -> None:
    missing = []
    results = {}

    for task_key, info in TASKS.items():
        results_file = RESULTS_DIR / task_key / "supervised_results.json"
        if not results_file.is_file():
            missing.append(str(results_file))
            continue
        with open(results_file) as f:
            results[task_key] = json.load(f)

    if missing:
        print("ERROR: Missing supervised results files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun the training first:")
        print("  bash scripts/train_supervised_fall_motion.sh")
        sys.exit(1)

    print("=" * 70)
    print("NEW SUPERVISED BASELINE RESULTS")
    print("=" * 70)

    for task_key, info in TASKS.items():
        r = results[task_key]
        acc = r["test_accuracy_pct"]
        f1 = r["test_weighted_f1"]
        epoch = r["best_epoch"]
        n_train = r["train_samples"]
        n_val = r["val_samples"]
        n_test = r["test_samples"]

        delta = info["zs_best"] - acc
        winner = "Zero-shot" if delta > 0 else ("Tied" if delta == 0 else "Supervised")

        print(f"\n{info['label']}:")
        print(f"  Old supervised (from {info['old_source']}): {info['old_value']}%")
        print(f"  NEW supervised (our ResNet-18 1D):     {acc:.2f}%")
        print(f"  Weighted F1:                           {f1:.4f}")
        print(f"  Best epoch:                            {epoch}")
        print(f"  Train/Val/Test:                        {n_train}/{n_val}/{n_test}")
        print(f"  Best zero-shot ({info['zs_method']}):  {info['zs_best']:.1f}%")
        print(f"  Delta (ZS - Supervised):               {delta:+.1f}pp  -->  {winner}")

    print("\n" + "=" * 70)
    print("PAPER UPDATE INSTRUCTIONS")
    print("=" * 70)

    fall_acc = results["csibench_fall_detection"]["test_accuracy_pct"]
    motion_acc = results["csibench_motion_source"]["test_accuracy_pct"]

    print(f"""
1. In paper/draft/06_experiments.md Table 1:
   - Replace *94.9$^\\dagger$* with *{fall_acc:.1f}* for Fall Detection
   - Replace *99.6$^\\dagger$* with *{motion_acc:.1f}* for Motion Source

2. In Section 6.1 (Supervised baselines paragraph):
   - Remove the sentence about Yang et al. baselines
   - Change to: "We compare against task-specific supervised ResNet-18
     classifiers trained on the same data splits with full label
     supervision, using the same architecture and hyperparameters as
     the alignment models."

3. In Section 6.2 (Finding 1):
   - Recount how many tasks ZS matches/exceeds supervised:
     Fall Detection: ZS {TASKS['csibench_fall_detection']['zs_best']:.1f}% vs Supervised {fall_acc:.1f}% --> {'ZS wins' if TASKS['csibench_fall_detection']['zs_best'] > fall_acc else 'Supervised wins' if fall_acc > TASKS['csibench_fall_detection']['zs_best'] else 'Tied'}
     Motion Source:  ZS {TASKS['csibench_motion_source']['zs_best']:.1f}% vs Supervised {motion_acc:.1f}% --> {'ZS wins' if TASKS['csibench_motion_source']['zs_best'] > motion_acc else 'Supervised wins' if motion_acc > TASKS['csibench_motion_source']['zs_best'] else 'Tied'}
   - Remove the Yang et al. caveat paragraph

4. In paper/draft/05_protocols.md Section 5.1:
   - Change baselines text back to: "task-specific supervised ResNet-18
     classifiers trained on the same data splits with full label
     supervision"

5. In paper/draft/07_discussion.md and 08_conclusion.md:
   - Update the supervised comparison numbers
   - Remove Yang et al. caveats
""")


if __name__ == "__main__":
    main()
