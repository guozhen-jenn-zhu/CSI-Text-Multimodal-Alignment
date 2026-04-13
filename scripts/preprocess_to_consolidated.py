"""Consolidate per-sample H5 files into a single HDF5 per split.

This is an optional preprocessing step for faster DataLoader throughput:
instead of opening thousands of small H5 files, the training loop reads
from a single consolidated file with contiguous arrays.

Usage:
    python scripts/preprocess_to_consolidated.py \
        --task_root /data/fm_downstream/HumanActivityRecognition \
        --h5_key CSI_amps \
        --output_dir data/processed/csibench/har \
        --target_subcarriers 64 \
        --target_time_steps 500
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def normalize_shape(
    csi: np.ndarray,
    target_subcarriers: int,
    target_time_steps: int,
) -> np.ndarray:
    """Interpolate and normalize a single CSI sample to [C, K_target, T_target]."""
    C, K, T = csi.shape

    if K != target_subcarriers:
        x_old = np.linspace(0, 1, K)
        x_new = np.linspace(0, 1, target_subcarriers)
        out = np.empty((C, target_subcarriers, T), dtype=csi.dtype)
        for c in range(C):
            f = interp1d(x_old, csi[c], axis=0, kind="linear")
            out[c] = f(x_new)
        csi = out
        K = target_subcarriers

    if T != target_time_steps:
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_time_steps)
        out = np.empty((C, K, target_time_steps), dtype=csi.dtype)
        for c in range(C):
            f = interp1d(x_old, csi[c], axis=1, kind="linear")
            out[c] = f(x_new)
        csi = out

    mean = csi.mean()
    std = csi.std()
    if std > 1e-8:
        csi = (csi - mean) / std
    else:
        csi = csi - mean

    return csi


def consolidate_split(
    task_root: Path,
    split: str,
    h5_key: str,
    output_path: Path,
    target_subcarriers: int,
    target_time_steps: int,
) -> int:
    """Consolidate all samples for a split into one HDF5 file.

    Returns the number of samples written.
    """
    from data.metadata_loader import MetadataLoader

    loader = MetadataLoader(task_root=str(task_root), h5_key=h5_key)

    try:
        split_ids = loader.get_split_ids(split)
    except FileNotFoundError:
        logger.warning("Split '%s' not found in %s -- skipping", split, task_root)
        return 0

    if not split_ids:
        logger.warning("No samples in split '%s'", split)
        return 0

    first = loader.load_sample(split_ids[0])
    first_normed = normalize_shape(first, target_subcarriers, target_time_steps)
    C, K, T = first_normed.shape

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(split_ids)

    logger.info(
        "Consolidating %d samples for split '%s' -> %s  shape=(%d, %d, %d, %d)",
        n, split, output_path, n, C, K, T,
    )

    with h5py.File(output_path, "w") as f:
        csi_ds = f.create_dataset(
            "csi_data",
            shape=(n, C, K, T),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            chunks=(min(64, n), C, K, T),
        )
        labels_list = []
        sid_list = []

        for i, sid in enumerate(split_ids):
            raw = loader.load_sample(sid)
            normed = normalize_shape(raw, target_subcarriers, target_time_steps)
            csi_ds[i] = normed.astype(np.float32)
            labels_list.append(loader.get_label(sid))
            sid_list.append(sid)

            if (i + 1) % 1000 == 0:
                logger.info("  processed %d/%d", i + 1, n)

        f.create_dataset("labels", data=np.array(labels_list))
        dt = h5py.special_dtype(vlen=str)
        sid_ds = f.create_dataset("sample_ids", (n,), dtype=dt)
        for i, sid in enumerate(sid_list):
            sid_ds[i] = sid

        f.attrs["target_subcarriers"] = target_subcarriers
        f.attrs["target_time_steps"] = target_time_steps
        f.attrs["split"] = split
        f.attrs["num_samples"] = n

    logger.info("Done: %s (%d samples)", output_path, n)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate per-sample H5 files into single HDF5 per split"
    )
    parser.add_argument("--task_root", type=str, required=True)
    parser.add_argument("--h5_key", type=str, default="CSI_amps")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--target_subcarriers", type=int, default=64)
    parser.add_argument("--target_time_steps", type=int, default=500)
    parser.add_argument("--s3_sync", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    task_root = Path(args.task_root)
    output_dir = Path(args.output_dir)

    total = 0
    for split in args.splits:
        out_path = output_dir / f"{split}.h5"
        n = consolidate_split(
            task_root=task_root,
            split=split,
            h5_key=args.h5_key,
            output_path=out_path,
            target_subcarriers=args.target_subcarriers,
            target_time_steps=args.target_time_steps,
        )
        total += n

    print(f"\nConsolidated {total} total samples into {output_dir}/")

    if args.s3_sync:
        try:
            from utils.s3_sync import S3Syncer
            syncer = S3Syncer(bucket="rnd-sagemaker", prefix="Multimodal_CSI_Text")
            syncer.upload_dataset(str(output_dir))
            logger.info("Synced consolidated data to S3")
        except Exception as e:
            logger.warning("S3 sync failed: %s", e)


if __name__ == "__main__":
    main()
