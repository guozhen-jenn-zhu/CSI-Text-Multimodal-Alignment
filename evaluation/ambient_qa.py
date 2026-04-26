"""Ambient scene QA evaluation via text-based multiple choice."""

from __future__ import annotations

import copy
import itertools
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from data.text_generation.templates.task_templates import (
    _ACTIVITY_DESCRIPTIONS,
    _PROXIMITY_DESCRIPTIONS,
    _USER_DESCRIPTIONS,
    generate_composite_caption,
)

logger = logging.getLogger(__name__)


class AmbientQA:
    """Ambient scene QA evaluation via text-based multiple choice.

    For each attribute type, a natural-language question is posed with a fixed
    set of answer options.  The CSI embedding is matched against the encoded
    answer texts to select the best answer.
    """

    QUESTION_TEMPLATES: dict[str, dict] = {
        "activity": {
            "question": "What is the person doing?",
            "answers": {},
        },
        "proximity": {
            "question": "How far is the person from the sensor?",
            "answers": {
                0: "Very close, about 0.5 meters",
                1: "Nearby, about 1.5 meters",
                2: "A moderate distance, about 2.5 meters",
                3: "Far away, about 3.5 meters",
            },
        },
        "user_identity": {
            "question": "Who is present?",
            "answers": {},
        },
        "motion_source": {
            "question": "What is causing the motion?",
            "answers": {
                0: "A fan is oscillating",
                1: "A human is moving",
                2: "A cleaning robot is moving",
                3: "A pet is moving",
            },
        },
        "fall_event": {
            "question": "Is a fall occurring?",
            "answers": {
                0: "A person is falling",
                1: "No fall is occurring",
            },
        },
    }

    def __init__(
        self,
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> None:
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.device = device
        self._answer_cache: dict[str, torch.Tensor] = {}

    def register_answers(self, attribute: str, answers: dict[int, str]) -> None:
        """Register or override answer options for an attribute."""
        if attribute not in self.QUESTION_TEMPLATES:
            self.QUESTION_TEMPLATES[attribute] = {
                "question": attribute.replace("_", " ").capitalize() + "?",
                "answers": {},
            }
        self.QUESTION_TEMPLATES[attribute]["answers"] = copy.deepcopy(answers)
        self._answer_cache.pop(attribute, None)

    @torch.no_grad()
    def build_answer_embeddings(self, attribute: str) -> torch.Tensor:
        """Embed all answer options for *attribute*.

        Returns:
            ``[num_answers, D]`` L2-normalised embeddings, ordered by answer id.
        """
        if attribute in self._answer_cache:
            return self._answer_cache[attribute]

        answers = self.QUESTION_TEMPLATES[attribute]["answers"]
        if not answers:
            raise ValueError(
                f"No answers registered for attribute '{attribute}'. "
                "Call register_answers() first."
            )

        sorted_ids = sorted(answers.keys())
        question = self.QUESTION_TEMPLATES[attribute]["question"]
        texts = [f"{question} {answers[aid]}" for aid in sorted_ids]

        emb = self.text_encoder.encode(texts).to(self.device)
        if self.text_projection is not None:
            emb = self.text_projection(emb)
        emb = F.normalize(emb, dim=-1)

        self._answer_cache[attribute] = emb
        return emb

    @torch.no_grad()
    def answer_question(
        self, csi_embedding: torch.Tensor, attribute: str,
    ) -> int:
        """Select the best answer for *attribute* given a single CSI embedding.

        Args:
            csi_embedding: ``[D]`` or ``[1, D]``.

        Returns:
            Predicted answer id (from the attribute's answer dict keys).
        """
        answer_embs = self.build_answer_embeddings(attribute)
        csi_embedding = F.normalize(
            csi_embedding.to(self.device).view(1, -1), dim=-1,
        )
        sims = (csi_embedding @ answer_embs.t()).squeeze(0)
        idx = sims.argmax().item()
        sorted_ids = sorted(self.QUESTION_TEMPLATES[attribute]["answers"].keys())
        return sorted_ids[idx]

    @torch.no_grad()
    def classify_batch(
        self, csi_embeddings: torch.Tensor, attribute: str,
    ) -> np.ndarray:
        """Classify a batch of CSI embeddings for *attribute*.

        Args:
            csi_embeddings: ``[N, D]``.

        Returns:
            Predicted answer ids ``[N]``.
        """
        answer_embs = self.build_answer_embeddings(attribute)
        csi_embeddings = F.normalize(csi_embeddings.to(self.device), dim=-1)
        sims = csi_embeddings @ answer_embs.t()
        indices = sims.argmax(dim=-1).cpu().numpy()
        sorted_ids = sorted(self.QUESTION_TEMPLATES[attribute]["answers"].keys())
        return np.array([sorted_ids[i] for i in indices])

    def evaluate_single_attribute(
        self,
        csi_embeddings: torch.Tensor,
        labels: np.ndarray,
        attribute: str,
    ) -> dict:
        """Evaluate single-attribute QA accuracy.

        Returns:
            Dict with ``accuracy``, ``weighted_f1``, ``predictions``.
        """
        preds = self.classify_batch(csi_embeddings, attribute)
        acc = accuracy_score(labels, preds)
        w_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        return {
            "attribute": attribute,
            "accuracy": float(acc),
            "weighted_f1": float(w_f1),
            "predictions": preds,
        }

    def evaluate_multi_attribute(
        self,
        csi_embeddings: torch.Tensor,
        labels_dict: dict[str, np.ndarray],
        attributes: list[str],
    ) -> dict:
        """Evaluate multi-attribute QA.

        For each attribute, per-attribute accuracy is computed independently.
        Joint accuracy requires all attributes predicted correctly per sample.

        Args:
            labels_dict: ``{"activity": labels_array, "proximity": labels_array, ...}``
            attributes: List of attribute keys to evaluate.

        Returns:
            Dict with ``per_attribute`` results and ``joint_accuracy``.
        """
        per_attr: dict[str, dict] = {}
        all_correct = np.ones(len(csi_embeddings), dtype=bool)

        for attr in attributes:
            if attr not in labels_dict:
                logger.warning("Attribute '%s' missing from labels_dict, skipping.", attr)
                continue
            result = self.evaluate_single_attribute(
                csi_embeddings, labels_dict[attr], attr,
            )
            per_attr[attr] = result
            all_correct &= (result["predictions"] == labels_dict[attr])

        return {
            "per_attribute": per_attr,
            "joint_accuracy": float(all_correct.mean()) if len(per_attr) > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Level 2: Composite (holistic scene) QA
    # ------------------------------------------------------------------

    COMPOSITE_QUESTION = "Describe the scene:"

    @torch.no_grad()
    def build_composite_embeddings(self) -> tuple[torch.Tensor, list[tuple[str, str, str]]]:
        """Embed all 140 composite scene descriptions.

        Returns:
            embeddings: ``[140, D]`` L2-normalised embeddings.
            index_to_tuple: list mapping each row index to
                ``(user_id, activity, proximity)`` string keys.
        """
        if "_composite" in self._answer_cache:
            return self._answer_cache["_composite"], self._composite_index

        activities = list(_ACTIVITY_DESCRIPTIONS.keys())
        proximities = list(_PROXIMITY_DESCRIPTIONS.keys())
        users = list(_USER_DESCRIPTIONS.keys())

        index_to_tuple: list[tuple[str, str, str]] = []
        texts: list[str] = []
        # Iterate activity-outermost so that indices are grouped by activity
        # (needed for the confusion matrix and heatmap visualizations).
        for activity, prox, user in itertools.product(activities, proximities, users):
            caption = generate_composite_caption(user, activity, prox, template_idx=0)
            texts.append(f"{self.COMPOSITE_QUESTION} {caption}")
            index_to_tuple.append((user, activity, prox))

        emb = self.text_encoder.encode(texts).to(self.device)
        if self.text_projection is not None:
            emb = self.text_projection(emb)
        emb = F.normalize(emb, dim=-1)

        self._answer_cache["_composite"] = emb
        self._composite_index = index_to_tuple
        return emb, index_to_tuple

    @torch.no_grad()
    def classify_composite_batch(
        self, csi_embeddings: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Match CSI embeddings to composite scene descriptions.

        Args:
            csi_embeddings: ``[N, D]``.

        Returns:
            pred_indices: ``[N]`` predicted composite index (into
                ``index_to_tuple``).
            sim_matrix: ``[N, 140]`` cosine similarity matrix.
        """
        comp_emb, _ = self.build_composite_embeddings()
        csi_embeddings = F.normalize(csi_embeddings.to(self.device), dim=-1)
        sims = csi_embeddings @ comp_emb.t()  # [N, 140]
        pred_indices = sims.argmax(dim=-1).cpu().numpy()
        return pred_indices, sims.cpu().numpy()

    def evaluate_composite_qa(
        self,
        csi_embeddings: torch.Tensor,
        gt_tuples: list[tuple[str, str, str]],
    ) -> dict:
        """Evaluate Level 2 composite (holistic) scene QA.

        Args:
            csi_embeddings: ``[N, D]``.
            gt_tuples: list of ``(user_id, activity, proximity)`` ground-truth
                string keys, one per sample.

        Returns:
            Dict with composite_accuracy, top-k, MRR, decomposed per-attribute
            accuracy, and the similarity matrix.
        """
        comp_emb, index_to_tuple = self.build_composite_embeddings()
        tuple_to_idx = {t: i for i, t in enumerate(index_to_tuple)}

        missing = [t for t in gt_tuples if t not in tuple_to_idx]
        if missing:
            unique_missing = sorted(set(missing))
            raise ValueError(
                f"{len(missing)} ground-truth tuples not found in composite "
                f"index ({len(unique_missing)} unique). First 5: {unique_missing[:5]}"
            )
        gt_indices = np.array([tuple_to_idx[t] for t in gt_tuples], dtype=np.int64)

        pred_indices, sim_matrix = self.classify_composite_batch(csi_embeddings)

        exact_match = (pred_indices == gt_indices)
        composite_accuracy = float(exact_match.mean())

        ranks = np.zeros(len(gt_indices), dtype=np.int64)
        for i, gt_idx in enumerate(gt_indices):
            sorted_indices = np.argsort(-sim_matrix[i])
            rank = int(np.where(sorted_indices == gt_idx)[0][0]) + 1
            ranks[i] = rank

        top3 = float((ranks <= 3).mean())
        top5 = float((ranks <= 5).mean())
        mrr = float((1.0 / ranks).mean())

        pred_tuples = [index_to_tuple[i] for i in pred_indices]
        decomposed = {}
        attr_names = ["user_identity", "activity", "proximity"]
        for attr_pos, attr_name in enumerate(attr_names):
            correct = sum(
                1 for pt, gt in zip(pred_tuples, gt_tuples)
                if pt[attr_pos] == gt[attr_pos]
            )
            decomposed[attr_name] = float(correct / len(gt_tuples))

        logger.info(
            "CompositeQA: accuracy=%.4f  top3=%.4f  top5=%.4f  MRR=%.4f",
            composite_accuracy, top3, top5, mrr,
        )
        for attr_name, acc in decomposed.items():
            logger.info("CompositeQA decomposed [%s]: %.4f", attr_name, acc)

        return {
            "composite_accuracy": composite_accuracy,
            "top3_accuracy": top3,
            "top5_accuracy": top5,
            "mrr": mrr,
            "decomposed_per_attribute": decomposed,
            "num_candidates": len(index_to_tuple),
            "predictions": pred_indices,
            "gt_indices": gt_indices,
            "sim_matrix": sim_matrix,
            "index_to_tuple": index_to_tuple,
        }

    # ------------------------------------------------------------------
    # Level 2: Visualization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_composite_confusion(
        pred_indices: np.ndarray,
        gt_indices: np.ndarray,
        index_to_tuple: list[tuple[str, str, str]],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """Build a structured confusion matrix over composite classes.

        The matrix is ordered by (activity, proximity, user) so that
        attribute-level error patterns are visually apparent.

        Returns:
            ``[C, C]`` confusion matrix where C = len(index_to_tuple).
        """
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix as _cm

        n_classes = len(index_to_tuple)
        cm = _cm(gt_indices, pred_indices, labels=np.arange(n_classes))
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        if save_path is None:
            return cm

        activities = list(dict.fromkeys(t[1] for t in index_to_tuple))
        n_per_activity = n_classes // len(activities)

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")

        for boundary in range(1, len(activities)):
            pos = boundary * n_per_activity - 0.5
            ax.axhline(pos, color="black", linewidth=1.2)
            ax.axvline(pos, color="black", linewidth=1.2)

        mid_positions = [
            (i * n_per_activity + (i + 1) * n_per_activity) / 2 - 0.5
            for i in range(len(activities))
        ]
        act_short = [a.split("_")[0].capitalize() for a in activities]
        ax.set_xticks(mid_positions)
        ax.set_xticklabels(act_short, fontsize=10)
        ax.set_yticks(mid_positions)
        ax.set_yticklabels(act_short, fontsize=10)

        ax.set_xlabel("Predicted composite", fontsize=12)
        ax.set_ylabel("True composite", fontsize=12)
        ax.set_title(
            f"Composite Scene QA Confusion ({n_classes} classes)",
            fontsize=13, fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised frequency")
        fig.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved composite confusion matrix to %s", save_path)
        return cm

    @staticmethod
    def plot_similarity_heatmap(
        sim_matrix: np.ndarray,
        gt_indices: np.ndarray,
        index_to_tuple: list[tuple[str, str, str]],
        labels_dict: dict[str, np.ndarray],
        save_path: Optional[str] = None,
        samples_per_activity: int = 1,
    ) -> None:
        """Plot a heatmap of CSI-to-composite cosine similarity.

        Selects a few representative samples (one per activity) and shows
        their similarity to all 140 composite descriptions.
        """
        import matplotlib.pyplot as plt

        activities = list(dict.fromkeys(t[1] for t in index_to_tuple))
        activity_labels = labels_dict.get("activity")
        if activity_labels is None:
            logger.warning("Cannot plot similarity heatmap: no activity labels")
            return

        selected_indices: list[int] = []
        selected_labels: list[str] = []

        activity_keys = list(_ACTIVITY_DESCRIPTIONS.keys())
        for act_key in activity_keys:
            act_idx_in_composite = [
                i for i, t in enumerate(index_to_tuple) if t[1] == act_key
            ]
            if not act_idx_in_composite:
                continue
            sample_mask = np.isin(gt_indices, act_idx_in_composite)
            candidates = np.where(sample_mask)[0]
            if len(candidates) == 0:
                continue
            chosen = candidates[:samples_per_activity]
            for c in chosen:
                selected_indices.append(c)
                gt_tuple = index_to_tuple[gt_indices[c]]
                selected_labels.append(
                    f"{gt_tuple[1]} / {gt_tuple[0]} / {gt_tuple[2]}"
                )

        if not selected_indices:
            logger.warning("No samples selected for similarity heatmap")
            return

        sub_sim = sim_matrix[selected_indices]  # [K, 140]
        n_composites = sub_sim.shape[1]

        fig, ax = plt.subplots(figsize=(18, max(3, len(selected_indices) * 0.8 + 2)))
        im = ax.imshow(sub_sim, cmap="RdYlBu_r", aspect="auto", vmin=-0.1, vmax=0.5)

        n_per_activity = n_composites // len(activities)
        for boundary in range(1, len(activities)):
            pos = boundary * n_per_activity - 0.5
            ax.axvline(pos, color="black", linewidth=1.0, linestyle="--")

        mid_positions = [
            (i * n_per_activity + (i + 1) * n_per_activity) / 2 - 0.5
            for i in range(len(activities))
        ]
        act_short = [a.split("_")[0].capitalize() for a in activities]
        ax.set_xticks(mid_positions)
        ax.set_xticklabels(act_short, fontsize=10)

        for i, gt_idx in enumerate(
            [gt_indices[si] for si in selected_indices]
        ):
            ax.plot(gt_idx, i, marker="*", color="lime", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.8)

        ax.set_yticks(range(len(selected_labels)))
        ax.set_yticklabels(selected_labels, fontsize=9)
        ax.set_xlabel("Composite descriptions (grouped by activity)", fontsize=11)
        ax.set_ylabel("CSI sample (activity / user / distance)", fontsize=11)
        ax.set_title(
            "CSI-to-Composite Similarity (★ = ground truth)",
            fontsize=13, fontweight="bold",
        )
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Cosine similarity")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved similarity heatmap to %s", save_path)
        else:
            plt.close(fig)

    @torch.no_grad()
    def full_evaluation(
        self,
        model: nn.Module,
        dataloader,
        tasks_and_labels: dict[str, np.ndarray],
        csi_projection: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> dict:
        """Run full QA evaluation across all applicable attributes.

        Args:
            model: Alignment model with ``csi_encoder`` attribute.
            dataloader: Yields batches with ``"csi"`` key.
            tasks_and_labels: ``{attribute: labels_array}`` for each attribute
                to evaluate.
            csi_projection: Optional CSI projection head.
            device: Target device.

        Returns:
            Dict with per-attribute metrics, joint accuracy, and embeddings.
        """
        model.eval()
        csi_encoder = model.csi_encoder if hasattr(model, "csi_encoder") else model
        if csi_projection is None and hasattr(model, "csi_projection"):
            csi_projection = model.csi_projection

        all_embeddings: list[torch.Tensor] = []
        for batch in dataloader:
            csi = batch["csi"].to(device)
            features = csi_encoder(csi)
            if csi_projection is not None:
                features = csi_projection(features)
            all_embeddings.append(features.cpu())

        csi_embeddings = torch.cat(all_embeddings, dim=0)
        attributes = [
            attr for attr in tasks_and_labels
            if attr in self.QUESTION_TEMPLATES
            and self.QUESTION_TEMPLATES[attr]["answers"]
        ]

        results = self.evaluate_multi_attribute(
            csi_embeddings, tasks_and_labels, attributes,
        )
        results["csi_embeddings"] = csi_embeddings

        for attr, res in results.get("per_attribute", {}).items():
            logger.info(
                "AmbientQA [%s]: accuracy=%.4f  weighted_f1=%.4f",
                attr, res["accuracy"], res["weighted_f1"],
            )
        logger.info("AmbientQA joint accuracy: %.4f", results["joint_accuracy"])
        return results
