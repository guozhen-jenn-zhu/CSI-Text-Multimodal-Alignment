"""Zero-shot classification via text prototypes."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

logger = logging.getLogger(__name__)


class ZeroShotClassifier:
    """Zero-shot classification using text class prototypes."""

    def __init__(
        self,
        class_texts: dict[int, list[str]],
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            class_texts: ``{class_id: [text1, text2, ...]}`` per class.
            text_encoder: Frozen text encoder with ``.encode()`` method.
            text_projection: Optional projection head applied after encoding.
        """
        self.device = device
        self.class_ids = sorted(class_texts.keys())
        self.num_classes = len(self.class_ids)
        self.prototypes = self._build_prototypes(
            class_texts, text_encoder, text_projection,
        )

    @torch.no_grad()
    def _build_prototypes(
        self,
        class_texts: dict[int, list[str]],
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module],
    ) -> torch.Tensor:
        """Build class prototypes by averaging text embeddings per class.

        Returns:
            Tensor of shape ``[num_classes, embed_dim]``, L2-normalised.
        """
        prototypes: list[torch.Tensor] = []
        for cid in self.class_ids:
            texts = class_texts[cid]
            emb = text_encoder.encode(texts).to(self.device)
            if text_projection is not None:
                emb = text_projection(emb)
            proto = F.normalize(emb.mean(dim=0, keepdim=True), dim=-1)
            prototypes.append(proto)
        return torch.cat(prototypes, dim=0)

    @torch.no_grad()
    def classify(self, csi_embeddings: torch.Tensor) -> np.ndarray:
        """Classify CSI embeddings by nearest prototype (cosine similarity).

        Args:
            csi_embeddings: ``[N, embed_dim]`` — already projected and normalised.

        Returns:
            Predicted class ids ``[N]``.
        """
        csi_embeddings = F.normalize(csi_embeddings.to(self.device), dim=-1)
        sims = csi_embeddings @ self.prototypes.t()
        indices = sims.argmax(dim=-1).cpu().numpy()
        return np.array([self.class_ids[i] for i in indices])

    def evaluate(
        self, csi_embeddings: torch.Tensor, labels: np.ndarray,
    ) -> dict:
        """Full evaluation with standard metrics.

        Returns:
            Dict with ``accuracy``, ``weighted_f1``, ``per_class_f1``,
            ``confusion_matrix``, ``predictions``.
        """
        preds = self.classify(csi_embeddings)
        acc = accuracy_score(labels, preds)
        w_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        per_f1 = f1_score(
            labels, preds, average=None, labels=self.class_ids, zero_division=0,
        )
        cm = confusion_matrix(labels, preds, labels=self.class_ids)

        return {
            "accuracy": float(acc),
            "weighted_f1": float(w_f1),
            "per_class_f1": {cid: float(f) for cid, f in zip(self.class_ids, per_f1)},
            "confusion_matrix": cm,
            "predictions": preds,
        }


@torch.no_grad()
def run_zero_shot_evaluation(
    model: nn.Module,
    dataloader,
    class_texts: dict[int, list[str]],
    text_encoder: nn.Module,
    text_projection: Optional[nn.Module] = None,
    csi_projection: Optional[nn.Module] = None,
    device: str = "cuda",
) -> dict:
    """End-to-end zero-shot evaluation pipeline.

    Encodes all CSI samples via the model's encoder (and optional projection),
    builds text prototypes, classifies, and reports metrics.

    Args:
        model: Alignment model containing ``csi_encoder`` attribute.
        dataloader: Yields batches with ``"csi"`` and ``"labels"`` keys.
        class_texts: ``{class_id: [text1, text2, ...]}`` for prototype building.
        text_encoder: Frozen text encoder.
        text_projection: Optional text projection head.
        csi_projection: Optional CSI projection head.  Falls back to
            ``model.csi_projection`` if available.
        device: Target device.

    Returns:
        Dict with metrics and ``csi_embeddings`` / ``labels`` arrays.
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

    classifier = ZeroShotClassifier(
        class_texts, text_encoder, text_projection, device=device,
    )
    metrics = classifier.evaluate(csi_embeddings, labels)

    metrics["csi_embeddings"] = csi_embeddings
    metrics["labels"] = labels
    metrics["prototypes"] = classifier.prototypes.cpu()

    logger.info(
        "Zero-shot evaluation: accuracy=%.4f  weighted_f1=%.4f",
        metrics["accuracy"],
        metrics["weighted_f1"],
    )
    return metrics
