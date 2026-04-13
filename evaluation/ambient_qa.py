"""Ambient scene QA evaluation via text-based multiple choice."""

from __future__ import annotations

import copy
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

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
                0: "A human is moving",
                1: "A pet is moving",
                2: "A cleaning robot is moving",
                3: "A fan is oscillating",
            },
        },
        "fall_event": {
            "question": "Is a fall occurring?",
            "answers": {
                0: "No fall is occurring",
                1: "A person is falling",
            },
        },
        "breathing": {
            "question": "Is someone breathing in the room?",
            "answers": {
                0: "The room is empty",
                1: "Someone is breathing",
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
