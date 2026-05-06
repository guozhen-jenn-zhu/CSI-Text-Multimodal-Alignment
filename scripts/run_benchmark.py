"""Run the full benchmark: methods x encoders x tasks/datasets."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path


METHODS = ["clip_infonce", "siglip", "language_bind", "centrobind", "barlow_twins", "masked_predict"]
ENCODERS = ["resnet18_1d", "patchtst", "vit_1d"]

CSIBENCH_TASKS = [
    "fall_detection", "breathing_detection", "localization",
    "motion_source", "har", "user_id", "proximity",
]

DATASET_TASKS = {
    "csibench": CSIBENCH_TASKS,
    "xrf55": ["action_recognition"],
    "signfi": ["sign_recognition"],
}


def _build_cmd(
    config: str,
    dataset: str,
    task: str,
    method: str,
    encoder: str,
    num_gpus: int,
    extra_overrides: list[str],
    data_root: str | None = None,
) -> list[str]:
    overrides = [
        f"dataset={dataset}",
        f"task={task}",
        f"method={method}",
        f"csi_encoder.name={encoder}",
    ]
    # Mirror scripts/pipelines/train_local.sh: data_root must already include
    # the source_prefix layer (e.g. data/Dataset/csibench). Without this the
    # trainer falls back to the SageMaker default "/data/fm_downstream" which
    # does not exist on local checkouts.
    if data_root:
        overrides.append(f"dataset_config.data_root={data_root}/{dataset}")
    overrides += extra_overrides

    if num_gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "-m", "training.train",
            "--config", config,
            "--override",
        ] + overrides
    else:
        cmd = [
            sys.executable, "-m", "training.train",
            "--config", config,
            "--override",
        ] + overrides

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_TASKS.keys()),
        choices=list(DATASET_TASKS.keys()),
    )
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--encoders", nargs="+", default=ENCODERS, choices=ENCODERS)
    parser.add_argument("--num_gpus", type=int, default=0, help="0 = auto-detect")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Dataset",
        help="Root data directory holding csibench/, xrf55/, signfi/ subfolders "
             "(default: data/Dataset, matching the released local layout). "
             "Pass an empty string to fall back to the in-config default.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    if args.num_gpus <= 0:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1
    else:
        num_gpus = args.num_gpus

    experiments = []
    for dataset in args.datasets:
        tasks = DATASET_TASKS[dataset]
        for task, method, encoder in itertools.product(tasks, args.methods, args.encoders):
            experiments.append((dataset, task, method, encoder))

    print(f"Benchmark: {len(experiments)} experiments, {num_gpus} GPU(s)")

    results = []
    for i, (dataset, task, method, encoder) in enumerate(experiments, 1):
        label = f"[{i}/{len(experiments)}] {dataset}/{task}/{method}/{encoder}"
        cmd = _build_cmd(
            args.config,
            dataset,
            task,
            method,
            encoder,
            num_gpus,
            args.override,
            data_root=args.data_root or None,
        )

        if args.dry_run:
            print(f"{label}: {' '.join(cmd)}")
            continue

        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
        results.append((label, status, elapsed))
        print(f"{label}: {status} ({elapsed:.0f}s)")

    if results:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for label, status, elapsed in results:
            print(f"  {status:12s}  {elapsed:6.0f}s  {label}")


if __name__ == "__main__":
    main()
