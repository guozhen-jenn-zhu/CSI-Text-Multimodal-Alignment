"""Generate precomputed text embeddings and save as HDF5.

Reads a pair_registry.json, encodes every unique text string with the
specified text encoder, and writes an HDF5 file keyed by text_id.

Usage:
    python scripts/generate_text_embeddings.py \
        --registry_path /data/csibench/multitask/har/pair_registry.json \
        --output_path /data/csibench/multitask/har/text_embeddings.h5 \
        --encoder clip_vit_b32 \
        --batch_size 256
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_embeddings(
    registry_path: str,
    output_path: str,
    encoder_name: str = "clip_vit_b32",
    batch_size: int = 256,
    device: str = "cuda",
) -> Path:
    """Encode all texts from a pair registry and save as HDF5."""
    from data.pair_registry import PairRegistry
    from models.encoders import build_text_encoder

    registry = PairRegistry.load(registry_path)
    texts = registry.texts
    text_ids = list(texts.keys())
    text_strings = [texts[tid] for tid in text_ids]

    logger.info(
        "Encoding %d unique texts with '%s' on %s ...",
        len(text_ids), encoder_name, device,
    )

    encoder = build_text_encoder(encoder_name)
    encoder = encoder.to(device)
    encoder.eval()

    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(text_strings), batch_size):
        batch = text_strings[i : i + batch_size]
        with torch.no_grad():
            emb = encoder.encode(batch)
        all_embeddings.append(emb.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info("  encoded %d / %d texts", min(i + batch_size, len(text_strings)), len(text_strings))

    embeddings = np.concatenate(all_embeddings, axis=0)
    embed_dim = embeddings.shape[1]
    logger.info("Embedding shape: [%d, %d]", len(text_ids), embed_dim)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        for idx, tid in enumerate(text_ids):
            f.create_dataset(tid, data=embeddings[idx].astype(np.float32))

    logger.info("Saved text embeddings to %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text embeddings HDF5")
    parser.add_argument("--registry_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--encoder", type=str, default="clip_vit_b32",
        choices=["clip_vit_b32", "clip_vit_l14", "sentence_transformer"],
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    generate_embeddings(
        registry_path=args.registry_path,
        output_path=args.output_path,
        encoder_name=args.encoder,
        batch_size=args.batch_size,
        device=args.device if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()
