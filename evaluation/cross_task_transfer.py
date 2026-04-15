"""Cross-task and cross-dataset zero-shot transfer evaluation."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from evaluation.zero_shot import ZeroShotClassifier, run_zero_shot_evaluation

logger = logging.getLogger(__name__)

TASK_TO_DATASET: dict[str, str] = {
    "har": "csibench",
    "fall_detection": "csibench",
    "motion_source": "csibench",
    "proximity": "csibench",
    "user_id": "csibench",
    "sign_recognition": "signfi",
    "action_recognition": "xrf55",
}


class CrossTaskTransfer:
    """Evaluate zero-shot transfer across tasks within or across datasets.

    A model trained on a *source* task (e.g. HAR) is evaluated on every other
    task using only the target task's text class prototypes — no fine-tuning.
    """

    DATASET_TASKS: dict[str, list[str]] = {
        "csibench": ["har", "fall_detection", "motion_source", "proximity", "user_id"],
        "signfi": ["sign_recognition"],
        "xrf55": ["action_recognition"],
        "all": [
            "har", "fall_detection", "motion_source", "proximity", "user_id",
            "sign_recognition", "action_recognition",
        ],
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

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        source_task: str,
        target_task: str,
        target_dataloader,
        target_class_texts: dict[int, list[str]],
        csi_projection: Optional[nn.Module] = None,
    ) -> dict:
        """Evaluate a model trained on *source_task* zero-shot on *target_task*.

        Args:
            model: Alignment model trained on *source_task*.
            source_task: Name of the source training task (for logging).
            target_task: Name of the target evaluation task.
            target_dataloader: DataLoader for the target task's test split.
            target_class_texts: ``{class_id: [texts]}`` for the target task.
            csi_projection: Optional CSI projection head override.

        Returns:
            Dict with ``source_task``, ``target_task``, and standard metrics.
        """
        metrics = run_zero_shot_evaluation(
            model=model,
            dataloader=target_dataloader,
            class_texts=target_class_texts,
            text_encoder=self.text_encoder,
            text_projection=self.text_projection,
            csi_projection=csi_projection,
            device=self.device,
        )
        metrics["source_task"] = source_task
        metrics["target_task"] = target_task

        logger.info(
            "Cross-task transfer %s -> %s: accuracy=%.4f  weighted_f1=%.4f",
            source_task, target_task, metrics["accuracy"], metrics["weighted_f1"],
        )
        return metrics

    def evaluate_all_pairs(
        self,
        dataset_name: str,
        model: nn.Module,
        target_dataloaders: dict[str, object],
        target_class_texts_map: dict[str, dict[int, list[str]]],
        source_task: Optional[str] = None,
        csi_projection: Optional[nn.Module] = None,
    ) -> dict[str, dict]:
        """Evaluate cross-task transfer from *source_task* to all other tasks.

        When *source_task* is given, every other task in
        ``DATASET_TASKS[dataset_name]`` becomes a target.  When it is
        ``None``, all pairwise combinations are evaluated.

        Args:
            dataset_name: Dataset identifier (e.g. ``"csibench"``).
            model: Trained alignment model.
            target_dataloaders: ``{task_name: DataLoader}``
            target_class_texts_map: ``{task_name: {class_id: [texts]}}``
            source_task: The task the model was trained on. If ``None``,
                all pairwise directions are evaluated.
            csi_projection: Optional CSI projection head override.

        Returns:
            ``{"{src}->{tgt}": metrics_dict}``
        """
        all_tasks = self.DATASET_TASKS.get(dataset_name, [])
        if not all_tasks:
            logger.warning("No tasks defined for dataset '%s'.", dataset_name)
            return {}

        if source_task is not None:
            targets = [t for t in all_tasks if t != source_task]
        else:
            targets = list(all_tasks)

        results: dict[str, dict] = {}
        for tgt in targets:
            if tgt not in target_dataloaders or tgt not in target_class_texts_map:
                logger.warning(
                    "Skipping %s -> %s: missing dataloader or class texts.",
                    source_task or "*", tgt,
                )
                continue
            src_label = source_task or "all"
            key = f"{src_label}->{tgt}"
            results[key] = self.evaluate(
                model=model,
                source_task=src_label,
                target_task=tgt,
                target_dataloader=target_dataloaders[tgt],
                target_class_texts=target_class_texts_map[tgt],
                csi_projection=csi_projection,
            )
        return results


class CrossDatasetTransfer:
    """Evaluate zero-shot transfer across different CSI datasets.

    The core assumption is that the unified preprocessing pipeline normalises
    different CSI formats into a common ``[C, K, T]`` representation, so the
    same CSI encoder can consume data from any source.
    """

    TRANSFER_DIRECTIONS: list[tuple[str, str]] = [
        ("csibench_har", "xrf55"),
        ("xrf55", "csibench_har"),
        ("signfi", "xrf55_gesture_subset"),
    ]

    def __init__(
        self,
        text_encoder: nn.Module,
        text_projection: Optional[nn.Module] = None,
        device: str = "cuda",
    ) -> None:
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        source_name: str,
        target_name: str,
        target_dataloader,
        target_class_texts: dict[int, list[str]],
        csi_projection: Optional[nn.Module] = None,
    ) -> dict:
        """Evaluate a model trained on *source_name* zero-shot on *target_name*.

        Args:
            model: Alignment model trained on the source dataset.
            source_name: Identifier for the source dataset (for logging).
            target_name: Identifier for the target dataset.
            target_dataloader: DataLoader for the target dataset's test split.
            target_class_texts: ``{class_id: [texts]}`` for target classes.
            csi_projection: Optional CSI projection head override.

        Returns:
            Dict with ``source_dataset``, ``target_dataset``, and metrics.
        """
        metrics = run_zero_shot_evaluation(
            model=model,
            dataloader=target_dataloader,
            class_texts=target_class_texts,
            text_encoder=self.text_encoder,
            text_projection=self.text_projection,
            csi_projection=csi_projection,
            device=self.device,
        )
        metrics["source_dataset"] = source_name
        metrics["target_dataset"] = target_name

        logger.info(
            "Cross-dataset transfer %s -> %s: accuracy=%.4f  weighted_f1=%.4f",
            source_name, target_name, metrics["accuracy"], metrics["weighted_f1"],
        )
        return metrics

    def evaluate_all_directions(
        self,
        model: nn.Module,
        source_name: str,
        target_configs: dict[str, dict],
        csi_projection: Optional[nn.Module] = None,
    ) -> dict[str, dict]:
        """Evaluate all predefined transfer directions from *source_name*.

        Args:
            source_name: The source dataset the model was trained on.
            target_configs: ``{target_name: {"dataloader": ..., "class_texts": ...}}``
            csi_projection: Optional CSI projection head override.

        Returns:
            ``{"{src}->{tgt}": metrics_dict}``
        """
        results: dict[str, dict] = {}
        for src, tgt in self.TRANSFER_DIRECTIONS:
            if src != source_name:
                continue
            if tgt not in target_configs:
                logger.warning(
                    "Skipping %s -> %s: target config not provided.", src, tgt,
                )
                continue
            cfg = target_configs[tgt]
            key = f"{src}->{tgt}"
            results[key] = self.evaluate(
                model=model,
                source_name=src,
                target_name=tgt,
                target_dataloader=cfg["dataloader"],
                target_class_texts=cfg["class_texts"],
                csi_projection=csi_projection,
            )
        return results
