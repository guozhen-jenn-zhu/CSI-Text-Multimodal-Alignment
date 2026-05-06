"""Run ablation studies with proper config overrides."""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time


ABLATIONS = {
    "text_granularity": {
        "description": "Effect of text granularity levels",
        "sweep": [
            {"text_generation.text_level": "level1"},
            {"text_generation.text_level": "level1_level2"},
        ],
    },
    "many_to_many_ratio": {
        "description": "Effect of many-to-many paraphrase count",
        "sweep": [
            {"text_generation.num_paraphrases": "1"},
            {"text_generation.num_paraphrases": "3"},
            {"text_generation.num_paraphrases": "5"},
            {"text_generation.num_paraphrases": "10"},
        ],
    },
    "csi_encoder": {
        "description": "Comparison of CSI encoder architectures",
        "sweep": [
            {"csi_encoder.name": "resnet18_1d"},
            {"csi_encoder.name": "patchtst"},
            {"csi_encoder.name": "vit_1d"},
        ],
    },
    "text_encoder": {
        "description": "Comparison of text encoder backbones",
        "sweep": [
            {"text_encoder.name": "clip_vit_b32"},
            {"text_encoder.name": "sentence_transformer"},
        ],
    },
    "embedding_dim": {
        "description": "Effect of shared embedding dimension",
        "sweep": [
            {"projection.dim": "128"},
            {"projection.dim": "256"},
            {"projection.dim": "512"},
        ],
    },
    "batch_size": {
        "description": "Effect of batch size on contrastive learning",
        "sweep": [
            {"training.batch_size": "64"},
            {"training.batch_size": "128"},
            {"training.batch_size": "256"},
            {"training.batch_size": "512"},
        ],
    },
}


def _build_cmd(
    config: str,
    overrides: dict,
    num_gpus: int,
    base_overrides: list[str],
    dataset: str | None = None,
    data_root: str | None = None,
) -> list[str]:
    override_args = [f"{k}={v}" for k, v in overrides.items()]
    # Mirror scripts/pipelines/train_local.sh: data_root must already include
    # the source_prefix layer (e.g. data/Dataset/csibench). Without this the
    # trainer falls back to the SageMaker default "/data/fm_downstream" which
    # does not exist on local checkouts.
    if data_root and dataset:
        override_args.append(f"dataset_config.data_root={data_root}/{dataset}")
    override_args += base_overrides

    if num_gpus > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "-m", "training.train",
            "--config", config,
            "--override",
        ] + override_args
    else:
        cmd = [
            sys.executable, "-m", "training.train",
            "--config", config,
            "--override",
        ] + override_args

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument(
        "--ablation", nargs="+", default=list(ABLATIONS.keys()),
        choices=list(ABLATIONS.keys()),
        help="Which ablation(s) to run",
    )
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="csibench",
        help="Dataset key whose source_prefix is appended to --data_root "
             "(default: csibench, matching configs/experiment/default.yaml).",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/Dataset",
        help="Root data directory holding csibench/, xrf55/, signfi/ subfolders "
             "(default: data/Dataset, matching the released local layout). "
             "Pass an empty string to fall back to the in-config default.",
    )
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.num_gpus <= 0:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1
    else:
        num_gpus = args.num_gpus

    for abl_name in args.ablation:
        abl = ABLATIONS[abl_name]
        print(f"\n{'='*60}")
        print(f"Ablation: {abl_name} -- {abl['description']}")
        print(f"{'='*60}")

        for i, overrides in enumerate(abl["sweep"], 1):
            label = f"  [{i}/{len(abl['sweep'])}] {overrides}"
            cmd = _build_cmd(
                args.config,
                overrides,
                num_gpus,
                args.override,
                dataset=args.dataset,
                data_root=args.data_root or None,
            )

            if args.dry_run:
                print(f"{label}: {' '.join(cmd)}")
                continue

            print(f"\n{label}")
            t0 = time.time()
            result = subprocess.run(cmd)
            elapsed = time.time() - t0
            status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
            print(f"  -> {status} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
