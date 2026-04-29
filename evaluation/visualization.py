"""Embedding visualization and analysis tools."""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """Visualization tools for aligned CSI-text embeddings."""

    CSI_MARKER = "o"
    TEXT_MARKER = "*"
    CSI_ALPHA = 0.6
    TEXT_ALPHA = 1.0
    TEXT_MARKER_SIZE = 200

    def plot_tsne(
        self,
        csi_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[dict[int, str]] = None,
        perplexity: float = 30.0,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
        compact_legend: Optional[bool] = None,
    ) -> plt.Figure:
        """Plot t-SNE of CSI and text embeddings coloured by class.

        CSI points are shown as circles, text prototypes as stars.
        """
        combined = np.concatenate([csi_embeddings, text_embeddings], axis=0)
        n_csi = len(csi_embeddings)

        effective_perplexity = min(perplexity, max(5.0, len(combined) / 4.0))
        tsne = TSNE(n_components=2, perplexity=effective_perplexity, random_state=42)
        proj = tsne.fit_transform(combined)

        csi_proj = proj[:n_csi]
        text_proj = proj[n_csi:]

        return self._scatter_plot(
            csi_proj, text_proj, labels, class_names,
            title="t-SNE: CSI & Text Embeddings",
            save_path=save_path, figsize=figsize,
            compact_legend=compact_legend,
        )

    def plot_umap(
        self,
        csi_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[dict[int, str]] = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        save_path: Optional[str] = None,
        figsize: tuple[int, int] = (10, 8),
        compact_legend: Optional[bool] = None,
    ) -> plt.Figure:
        """Plot UMAP of CSI and text embeddings coloured by class."""
        import umap

        combined = np.concatenate([csi_embeddings, text_embeddings], axis=0)
        n_csi = len(csi_embeddings)

        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=min_dist, random_state=42,
        )
        proj = reducer.fit_transform(combined)

        csi_proj = proj[:n_csi]
        text_proj = proj[n_csi:]

        return self._scatter_plot(
            csi_proj, text_proj, labels, class_names,
            title="UMAP: CSI & Text Embeddings",
            save_path=save_path, figsize=figsize,
            compact_legend=compact_legend,
        )

    def _scatter_plot(
        self,
        csi_2d: np.ndarray,
        text_2d: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[dict[int, str]],
        title: str,
        save_path: Optional[str],
        figsize: tuple[int, int],
        compact_legend: Optional[bool] = None,
    ) -> plt.Figure:
        from matplotlib.lines import Line2D

        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        if compact_legend is None:
            compact_legend = n_classes > 20

        cmap = plt.cm.get_cmap("tab20", 20)

        fig, ax = plt.subplots(figsize=figsize)
        for i, cls in enumerate(unique_labels):
            mask = labels == cls
            colour = cmap(i % 20)
            label_kw = {}
            if not compact_legend:
                name = class_names.get(cls, str(cls)) if class_names else str(cls)
                label_kw["label"] = f"{name} (CSI)"
            ax.scatter(
                csi_2d[mask, 0], csi_2d[mask, 1],
                c=[colour], marker=self.CSI_MARKER,
                alpha=self.CSI_ALPHA, s=30, **label_kw,
            )

        text_labels = unique_labels[:len(text_2d)]
        for i, cls in enumerate(text_labels):
            colour = cmap(i % 20)
            label_kw = {}
            if not compact_legend:
                name = class_names.get(cls, str(cls)) if class_names else str(cls)
                label_kw["label"] = f"{name} (text)"
            ax.scatter(
                text_2d[i, 0], text_2d[i, 1],
                c=[colour], marker=self.TEXT_MARKER,
                alpha=self.TEXT_ALPHA, s=self.TEXT_MARKER_SIZE,
                edgecolors="black", linewidths=0.8, **label_kw,
            )

        ax.set_title(title, fontsize=14)
        if compact_legend:
            handles = [
                Line2D([0], [0], marker="o", color="grey", linestyle="",
                       alpha=0.6, markersize=6, label="CSI embeddings"),
                Line2D([0], [0], marker="*", color="grey", linestyle="",
                       markersize=12, markeredgecolor="black",
                       markeredgewidth=0.8, label="Text prototypes"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=10)
        else:
            ax.legend(
                bbox_to_anchor=(1.02, 1), loc="upper left",
                fontsize=8, markerscale=0.8,
            )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info("Saved plot to %s", save_path)

        return fig


class LinearProbe:
    """Linear probe evaluation on frozen aligned embeddings."""

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "lbfgs",
    ) -> None:
        self.max_iter = max_iter
        self.C = C
        self.solver = solver

    def evaluate(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict:
        """Train logistic regression on frozen embeddings and evaluate.

        Returns:
            Dict with ``accuracy``, ``weighted_f1``, ``per_class_f1``.
        """
        clf = LogisticRegression(
            max_iter=self.max_iter, C=self.C,
            solver=self.solver, multi_class="multinomial",
        )
        clf.fit(train_embeddings, train_labels)
        preds = clf.predict(test_embeddings)

        acc = accuracy_score(test_labels, preds)
        w_f1 = f1_score(test_labels, preds, average="weighted", zero_division=0)
        per_f1 = f1_score(test_labels, preds, average=None, zero_division=0)

        unique_labels = np.unique(np.concatenate([train_labels, test_labels]))

        logger.info("Linear probe: accuracy=%.4f  weighted_f1=%.4f", acc, w_f1)
        return {
            "accuracy": float(acc),
            "weighted_f1": float(w_f1),
            "per_class_f1": {
                int(c): float(f) for c, f in zip(unique_labels, per_f1)
            },
            "predictions": preds,
        }


class FailureCaseAnalyzer:
    """Analyse failure cases in zero-shot classification."""

    def analyze(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        csi_embeddings: np.ndarray,
        text_prototypes: np.ndarray,
        class_names: Optional[dict[int, str]] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Identify and categorise failure cases.

        Returns:
            Dict with ``confusion_matrix``, ``error_rate``,
            ``hardest_classes``, ``failure_details``.
        """
        unique_classes = np.unique(np.concatenate([predictions, labels]))
        cm = confusion_matrix(labels, predictions, labels=unique_classes)

        incorrect_mask = predictions != labels
        error_rate = float(incorrect_mask.mean())

        per_class_error: dict[int, float] = {}
        for cls in unique_classes:
            cls_mask = labels == cls
            if cls_mask.sum() == 0:
                continue
            per_class_error[int(cls)] = float(
                (predictions[cls_mask] != cls).mean()
            )

        hardest = sorted(per_class_error.items(), key=lambda x: -x[1])

        failure_details: list[dict] = []
        fail_indices = np.where(incorrect_mask)[0]
        for idx in fail_indices:
            true_cls = int(labels[idx])
            pred_cls = int(predictions[idx])

            emb = csi_embeddings[idx]
            norms = np.linalg.norm(emb)
            if norms > 0:
                emb_normed = emb / norms
            else:
                emb_normed = emb

            proto_norms = np.linalg.norm(text_prototypes, axis=1, keepdims=True)
            proto_norms = np.maximum(proto_norms, 1e-8)
            proto_normed = text_prototypes / proto_norms

            sims = emb_normed @ proto_normed.T

            detail: dict = {
                "index": int(idx),
                "true_class": true_cls,
                "predicted_class": pred_cls,
                "similarity_to_true": float(sims[true_cls]) if true_cls < len(sims) else None,
                "similarity_to_predicted": float(sims[pred_cls]) if pred_cls < len(sims) else None,
                "similarity_margin": None,
            }
            if detail["similarity_to_true"] is not None and detail["similarity_to_predicted"] is not None:
                detail["similarity_margin"] = (
                    detail["similarity_to_predicted"] - detail["similarity_to_true"]
                )

            if class_names:
                detail["true_class_name"] = class_names.get(true_cls, str(true_cls))
                detail["predicted_class_name"] = class_names.get(pred_cls, str(pred_cls))

            if metadata:
                for key, values in metadata.items():
                    if hasattr(values, "__getitem__") and idx < len(values):
                        detail[f"meta_{key}"] = values[idx]

            failure_details.append(detail)

        confusion_pairs: dict[str, int] = {}
        for d in failure_details:
            pair_key = f"{d['true_class']}->{d['predicted_class']}"
            confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1
        top_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])

        result = {
            "confusion_matrix": cm,
            "error_rate": error_rate,
            "per_class_error_rate": per_class_error,
            "hardest_classes": [
                {
                    "class_id": cid,
                    "class_name": class_names.get(cid, str(cid)) if class_names else str(cid),
                    "error_rate": erate,
                }
                for cid, erate in hardest
            ],
            "top_confusion_pairs": top_confusions[:20],
            "num_failures": int(incorrect_mask.sum()),
            "total_samples": len(labels),
            "failure_details": failure_details,
        }

        logger.info(
            "Failure analysis: %d / %d errors (%.2f%%). Hardest class: %s (%.2f%%)",
            result["num_failures"], result["total_samples"],
            error_rate * 100,
            hardest[0][0] if hardest else "N/A",
            hardest[0][1] * 100 if hardest else 0.0,
        )
        return result
