"""Launch distributed training via torchrun."""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DDP training with torchrun")
    parser.add_argument(
        "--config", type=str, default="configs/experiment/default.yaml",
        help="Experiment config YAML path",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=0,
        help="Number of GPUs (0 = auto-detect)",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Config overrides as key=value",
    )
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.num_gpus <= 0:
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1
    else:
        num_gpus = args.num_gpus

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "-m", "training.train",
        "--config", args.config,
    ]
    if args.override:
        cmd.extend(["--override"] + args.override)
    if args.resume:
        cmd.extend(["--resume", args.resume])

    print(f"Launching DDP with {num_gpus} GPU(s):")
    print("  " + " ".join(cmd))

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
