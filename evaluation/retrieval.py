"""CSI-to-text and text-to-CSI retrieval evaluation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluate cross-modal retrieval between CSI embeddings and text prototypes.

    Supports both CSI-to-text (given a CSI sample, retrieve the correct class
    text) and text-to-CSI (given a class text, retrieve CSI samples of that
    class) retrieval with Recall@K metrics.
    """

    def __init__(self, ks: tuple[int, ...] = (1, 5, 10)) -> None:
        self.ks = ks

    @torch.no_grad()
    def evaluate(
        self,
        csi_embeddings: torch.Tensor,
        labels: np.ndarray,
        class_texts: dict[int, list[str]],
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> dict[str, dict[str, float]]:
        """Run CSI-to-text and text-to-CSI retrieval evaluation.

        Args:
            csi_embeddings: ``[N, D]`` pre-computed, projected CSI embeddings.
            labels: ``[N]`` ground-truth class IDs.
            class_texts: ``{class_id: [text1, text2, ...]}`` per class.
            text_encoder: Frozen text encoder with ``.encode()`` method.
            text_projection: Optional projection head applied after encoding.
            device: Target device string.

        Returns:
            Dict with ``csi_to_text`` and ``text_to_csi`` sub-dicts, each
            containing ``recall@K``, ``median_rank``, and ``mean_rank``.
        """
        class_ids = sorted(class_texts.keys())

        all_text_embs = []
        text_class_ids = []
        for cid in class_ids:
            texts = class_texts[cid]
            emb = text_encoder.encode(texts).to(device)
            if text_projection is not None:
                emb = text_projection(emb)
            emb = F.normalize(emb, dim=-1)
            all_text_embs.append(emb)
            text_class_ids.extend([cid] * len(texts))

        text_embs = torch.cat(all_text_embs, dim=0)
        text_class_ids = np.array(text_class_ids)

        prototypes = []
        for cid in class_ids:
            texts = class_texts[cid]
            emb = text_encoder.encode(texts).to(device)
            if text_projection is not None:
                emb = text_projection(emb)
            proto = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            prototypes.append(proto)
        prototypes = torch.cat(prototypes, dim=0)

        csi_embs = F.normalize(csi_embeddings.to(device), dim=-1)

        csi_to_text = self._csi_to_text_retrieval(
            csi_embs, labels, prototypes, class_ids,
        )

        text_to_csi = self._text_to_csi_retrieval(
            csi_embs, labels, prototypes, class_ids,
        )

        return {
            "csi_to_text": csi_to_text,
            "text_to_csi": text_to_csi,
        }

    def _csi_to_text_retrieval(
        self,
        csi_embs: torch.Tensor,
        labels: np.ndarray,
        prototypes: torch.Tensor,
        class_ids: list[int],
    ) -> dict[str, float]:
        """For each CSI sample, rank class prototypes by similarity.

        The correct retrieval is the prototype of the sample's ground-truth class.
        """
        sims = csi_embs @ prototypes.t()
        sorted_indices = sims.argsort(dim=-1, descending=True).cpu().numpy()

        cid_to_idx = {cid: i for i, cid in enumerate(class_ids)}
        gt_indices = np.array([cid_to_idx[int(lbl)] for lbl in labels])

        ranks = np.array([
            np.where(sorted_indices[i] == gt_indices[i])[0][0]
            for i in range(len(labels))
        ])

        results: dict[str, float] = {}
        for k in self.ks:
            if k <= len(class_ids):
                results[f"recall@{k}"] = float((ranks < k).mean())
        results["median_rank"] = float(np.median(ranks) + 1)
        results["mean_rank"] = float(np.mean(ranks) + 1)
        return results

    def _text_to_csi_retrieval(
        self,
        csi_embs: torch.Tensor,
        labels: np.ndarray,
        prototypes: torch.Tensor,
        class_ids: list[int],
    ) -> dict[str, float]:
        """For each class prototype, rank all CSI samples by similarity.

        Recall@K = fraction of correct CSI samples in the top-K retrieved.
        Averaged across all class prototypes.
        """
        sims = prototypes @ csi_embs.t()
        sorted_indices = sims.argsort(dim=-1, descending=True).cpu().numpy()

        results: dict[str, float] = {}
        max_k = max(self.ks)
        per_class_recalls: dict[str, list[float]] = {f"recall@{k}": [] for k in self.ks}
        all_median_ranks: list[float] = []

        for i, cid in enumerate(class_ids):
            gt_mask = labels == cid
            n_relevant = gt_mask.sum()
            if n_relevant == 0:
                continue

            retrieved_order = sorted_indices[i]
            is_relevant = gt_mask[retrieved_order]

            for k in self.ks:
                top_k_relevant = is_relevant[:k].sum()
                recall_k = float(top_k_relevant) / min(k, n_relevant)
                per_class_recalls[f"recall@{k}"].append(recall_k)

            relevant_positions = np.where(is_relevant)[0]
            all_median_ranks.append(float(np.median(relevant_positions) + 1))

        for k in self.ks:
            key = f"recall@{k}"
            if per_class_recalls[key]:
                results[key] = float(np.mean(per_class_recalls[key]))
        if all_median_ranks:
            results["median_rank"] = float(np.mean(all_median_ranks))

        return results


@torch.no_grad()
def run_retrieval_evaluation(
    model: nn.Module,
    dataloader,
    class_texts: dict[int, list[str]],
    text_encoder: nn.Module,
    text_projection: Optional[nn.Module] = None,
    csi_projection: Optional[nn.Module] = None,
    device: str = "cuda",
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """End-to-end retrieval evaluation pipeline.

    Encodes all CSI samples, then runs retrieval metrics.

    Returns:
        Dict with ``csi_to_text`` and ``text_to_csi`` sub-dicts containing
        Recall@K and rank statistics.
    """
    model.eval()
    csi_encoder = model.csi_encoder if hasattr(model, "csi_encoder") else model
    if csi_projection is None and hasattr(model, "csi_projection"):
        csi_projection = model.csi_projection
    if text_projection is None and hasattr(model, "text_projection"):
        text_projection = model.text_projection

    all_embeddings: list[torch.Tensor] = []
    all_labels: list[np.ndarray] = []

    for batch in dataloader:
        csi = batch["csi"].to(device)
        features = csi_encoder(csi)
        if csi_projection is not None:
            features = csi_projection(features)
        all_embeddings.append(features.cpu())
        all_labels.append(batch["labels"].numpy())

    csi_embeddings = torch.cat(all_embeddings, dim=0)
    labels = np.concatenate(all_labels)

    evaluator = RetrievalEvaluator(ks=ks)
    results = evaluator.evaluate(
        csi_embeddings, labels, class_texts,
        text_encoder, text_projection, device=device,
    )

    c2t = results.get("csi_to_text", {})
    t2c = results.get("text_to_csi", {})
    logger.info(
        "Retrieval: CSI->Text R@1=%.4f R@5=%.4f  Text->CSI R@1=%.4f R@5=%.4f",
        c2t.get("recall@1", 0), c2t.get("recall@5", 0),
        t2c.get("recall@1", 0), t2c.get("recall@5", 0),
    )
    return results
