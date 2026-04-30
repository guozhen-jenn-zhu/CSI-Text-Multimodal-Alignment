"""Few-shot linear probe evaluation on frozen aligned embeddings."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class FewShotLinearProbe:
    """Evaluate few-shot linear probes on frozen aligned CSI embeddings.

    For each K in ``shots``, samples K examples per class from the training
    embeddings, fits a logistic regression, and evaluates on the full test set.
    Repeats ``n_trials`` times with different random samples and reports
    mean and standard deviation.
    """

    def __init__(
        self,
        shots: tuple[int, ...] = (1, 5, 10, 50),
        n_trials: int = 10,
        max_iter: int = 1000,
        C: float = 1.0,
        solver: str = "lbfgs",
        seed: int = 42,
    ) -> None:
        self.shots = shots
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.seed = seed

    def _sample_few_shot(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        k: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample K examples per class from the training set."""
        classes = np.unique(labels)
        selected_idx: list[int] = []
        for cls in classes:
            cls_indices = np.where(labels == cls)[0]
            n_available = len(cls_indices)
            n_sample = min(k, n_available)
            chosen = rng.choice(cls_indices, size=n_sample, replace=False)
            selected_idx.extend(chosen.tolist())
        idx = np.array(selected_idx)
        return embeddings[idx], labels[idx]

    def evaluate(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        """Run few-shot evaluation across all K values.

        Returns:
            Dict mapping ``"{k}-shot"`` to result dicts containing
            ``accuracy_mean``, ``accuracy_std``, ``f1_mean``, ``f1_std``.
        """
        results: dict[str, dict[str, float]] = {}

        for k in self.shots:
            min_class_count = min(
                np.sum(train_labels == cls) for cls in np.unique(train_labels)
            )
            if k > min_class_count:
                logger.info(
                    "Skipping %d-shot: smallest class has only %d samples",
                    k, min_class_count,
                )
                continue

            trial_accs: list[float] = []
            trial_f1s: list[float] = []

            for trial in range(self.n_trials):
                rng = np.random.RandomState(self.seed + trial)
                fs_emb, fs_lbl = self._sample_few_shot(
                    train_embeddings, train_labels, k, rng,
                )

                clf = LogisticRegression(
                    max_iter=self.max_iter,
                    C=self.C,
                    solver=self.solver,
                )
                clf.fit(fs_emb, fs_lbl)
                preds = clf.predict(test_embeddings)

                trial_accs.append(accuracy_score(test_labels, preds))
                trial_f1s.append(
                    f1_score(test_labels, preds, average="weighted", zero_division=0)
                )

            key = f"{k}-shot"
            results[key] = {
                "accuracy_mean": float(np.mean(trial_accs)),
                "accuracy_std": float(np.std(trial_accs)),
                "f1_mean": float(np.mean(trial_f1s)),
                "f1_std": float(np.std(trial_f1s)),
                "n_trials": self.n_trials,
                "k": k,
            }
            logger.info(
                "Few-shot %d-shot: accuracy=%.4f +/- %.4f  (n=%d trials)",
                k, results[key]["accuracy_mean"],
                results[key]["accuracy_std"], self.n_trials,
            )

        return results
