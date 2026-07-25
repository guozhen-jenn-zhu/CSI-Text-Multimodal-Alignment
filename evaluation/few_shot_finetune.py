"""Few-shot fine-tuning of the CSI branch for cross-domain adaptation.

Given a held-out domain split (e.g. ``cross_env``), sample K labelled
examples per class as *support*, fine-tune the CSI branch with a fresh
linear classification head, and score the remaining samples (*query*).
Comparing the result against the split's 0-shot accuracy shows how much a
handful of in-domain labels recovers from the domain gap.

Two adaptation scopes are supported and reported side by side:

``projection``
    Encoder frozen; ``csi_projection`` + head trained.  Encoder features
    are computed once and cached, so trials are cheap.
``encoder``
    Encoder + ``csi_projection`` + head trained (full CSI branch).
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

SCOPES = ("projection", "encoder")


class FewShotFinetune:
    """Within-domain few-shot fine-tuning on a held-out domain split.

    Parameters
    ----------
    shots : sequence of int
        Number of labelled examples per class to adapt on.
    scopes : sequence of str
        Subset of :data:`SCOPES` to evaluate.
    n_trials : int
        Random support draws per K; results are reported as mean +/- std.
    epochs : int
        Full-batch gradient steps taken on the support set.
    lr, weight_decay : float
        Adam hyperparameters.
    batch_size : int
        Batch size used for the (no-grad) query forward pass.
    seed : int
        Base seed; trial ``t`` uses ``seed + t``.  The same seed drives the
        support draw and the head initialisation for every scope, so scopes
        are compared on identical support sets.
    freeze_bn : bool
        Keep BatchNorm layers in eval mode during fine-tuning.  Support sets
        here hold only ``K * num_classes`` samples (as few as two), so batch
        statistics estimated from them are extremely noisy and, over the
        training steps, overwrite the pretrained running statistics that the
        query pass then relies on.  Freezing also keeps the scope comparison
        honest: any gain comes from the labelled shots rather than from
        unsupervised BatchNorm adaptation to the target domain.
    """

    def __init__(
        self,
        shots: Sequence[int] = (1, 5, 10, 20),
        scopes: Sequence[str] = SCOPES,
        n_trials: int = 5,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        seed: int = 42,
        freeze_bn: bool = True,
    ) -> None:
        unknown = [s for s in scopes if s not in SCOPES]
        if unknown:
            raise ValueError(f"Unknown scope(s) {unknown}. Available: {list(SCOPES)}")
        self.shots = tuple(int(k) for k in shots)
        self.scopes = tuple(scopes)
        self.n_trials = int(n_trials)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.freeze_bn = bool(freeze_bn)

    def _set_train_mode(self, module: nn.Module) -> None:
        """Put *module* in train mode, keeping BatchNorm frozen if configured."""
        module.train()
        if not self.freeze_bn:
            return
        for sub in module.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                sub.eval()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _split_support_query(
        labels: np.ndarray, k: int, rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw K indices per class as support; the complement is the query."""
        support: list[int] = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            support.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
        support_arr = np.array(sorted(support), dtype=np.int64)
        mask = np.ones(len(labels), dtype=bool)
        mask[support_arr] = False
        return support_arr, np.where(mask)[0]

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode(
        self,
        encoder: nn.Module,
        projection: Optional[nn.Module],
        csi: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Run the CSI branch over *csi* in eval mode, returning CPU features."""
        encoder.eval()
        if projection is not None:
            projection.eval()
        chunks: list[torch.Tensor] = []
        for start in range(0, len(csi), self.batch_size):
            batch = csi[start : start + self.batch_size].to(device)
            feats = encoder(batch)
            if projection is not None:
                feats = projection(feats)
            chunks.append(feats.detach().cpu())
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # A single fine-tune run
    # ------------------------------------------------------------------

    def _run_once(
        self,
        scope: str,
        csi: torch.Tensor,
        cached_feats: Optional[torch.Tensor],
        labels: np.ndarray,
        support_idx: np.ndarray,
        query_idx: np.ndarray,
        num_classes: int,
        encoder: nn.Module,
        projection: Optional[nn.Module],
        device: torch.device,
        trial_seed: int,
    ) -> tuple[float, float]:
        """Fine-tune on the support set and score the query set."""
        torch.manual_seed(trial_seed)

        support_y = torch.from_numpy(labels[support_idx]).long().to(device)
        query_y = labels[query_idx]

        if scope == "projection":
            # Encoder is frozen, so its features are precomputed once and the
            # projection is re-trained on top of them.
            proj = copy.deepcopy(projection).to(device) if projection is not None else None
            support_x = cached_feats[support_idx].to(device)
            if proj is not None:
                with torch.no_grad():
                    proj.eval()
                    feat_dim = proj(support_x[:1]).shape[1]
            else:
                feat_dim = cached_feats.shape[1]
            head = nn.Linear(feat_dim, num_classes).to(device)
            trainable = list(head.parameters())
            if proj is not None:
                trainable += list(proj.parameters())
                self._set_train_mode(proj)

            def forward_support() -> torch.Tensor:
                out = proj(support_x) if proj is not None else support_x
                return head(out)

        else:
            enc = copy.deepcopy(encoder).to(device)
            proj = copy.deepcopy(projection).to(device) if projection is not None else None
            support_csi = csi[support_idx].to(device)
            with torch.no_grad():
                enc.eval()
                probe = enc(support_csi[:1])
                if proj is not None:
                    proj.eval()
                    probe = proj(probe)
            feat_dim = probe.shape[1]
            head = nn.Linear(feat_dim, num_classes).to(device)
            trainable = list(head.parameters()) + list(enc.parameters())
            self._set_train_mode(enc)
            if proj is not None:
                trainable += list(proj.parameters())
                self._set_train_mode(proj)

            def forward_support() -> torch.Tensor:
                out = enc(support_csi)
                if proj is not None:
                    out = proj(out)
                return head(out)

        head.train()
        optimizer = torch.optim.Adam(
            trainable, lr=self.lr, weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(forward_support(), support_y)
            loss.backward()
            optimizer.step()

        # ---- query evaluation ----
        head.eval()
        if proj is not None:
            proj.eval()
        with torch.no_grad():
            if scope == "projection":
                query_x = cached_feats[query_idx]
                logits: list[torch.Tensor] = []
                for start in range(0, len(query_x), self.batch_size):
                    batch = query_x[start : start + self.batch_size].to(device)
                    out = proj(batch) if proj is not None else batch
                    logits.append(head(out).cpu())
            else:
                enc.eval()
                logits = []
                # Index per batch: materialising csi[query_idx] in one go would
                # duplicate the whole split in RAM on every run.
                for start in range(0, len(query_idx), self.batch_size):
                    idx = query_idx[start : start + self.batch_size]
                    batch = csi[idx].to(device)
                    out = enc(batch)
                    if proj is not None:
                        out = proj(out)
                    logits.append(head(out).cpu())
            preds = torch.cat(logits, dim=0).argmax(dim=-1).numpy()

        return (
            float(accuracy_score(query_y, preds)),
            float(f1_score(query_y, preds, average="weighted", zero_division=0)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        csi: torch.Tensor,
        labels: np.ndarray,
        encoder: nn.Module,
        projection: Optional[nn.Module],
        device: torch.device,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Sweep scopes x shots x trials on one held-out domain split.

        Parameters
        ----------
        csi : torch.Tensor
            All CSI samples for the split, shape ``[N, C, K, T]``, on CPU.
        labels : np.ndarray
            Integer class labels, shape ``[N]``.
        encoder, projection : nn.Module
            Pretrained CSI branch.  Copies are made per run, so these are
            never mutated.

        Notes
        -----
        For a given ``(k, trial)`` every scope sees the identical support
        draw and head initialisation, so the scopes can be compared pairwise.

        Returns
        -------
        dict
            ``{scope: {"{k}-shot": {accuracy_mean, accuracy_std, f1_mean,
            f1_std, k, n_trials}}}``.  A scope maps to ``{}`` when every K
            was skipped.
        """
        labels = np.asarray(labels)
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError(
                f"Few-shot fine-tuning needs >=2 classes in the split, found {len(classes)}"
            )

        # Contiguous 0..C-1 labels for the classification head.
        remap = {int(c): i for i, c in enumerate(classes)}
        mapped = np.array([remap[int(y)] for y in labels], dtype=np.int64)
        num_classes = len(classes)
        class_counts = {int(c): int(np.sum(labels == c)) for c in classes}
        min_count = min(class_counts.values())

        cached_feats: Optional[torch.Tensor] = None
        if "projection" in self.scopes:
            cached_feats = self._encode(encoder, None, csi, device)
            logger.info(
                "Cached frozen encoder features for projection-scope trials: %s",
                tuple(cached_feats.shape),
            )

        results: dict[str, dict[str, dict[str, float]]] = {s: {} for s in self.scopes}

        for k in self.shots:
            if min_count <= k:
                logger.info(
                    "Skipping %d-shot: smallest class has %d samples "
                    "(need > %d to leave a query set)", k, min_count, k,
                )
                continue

            for scope in self.scopes:
                accs: list[float] = []
                f1s: list[float] = []
                for trial in range(self.n_trials):
                    rng = np.random.RandomState(self.seed + trial)
                    support_idx, query_idx = self._split_support_query(mapped, k, rng)
                    acc, f1 = self._run_once(
                        scope=scope,
                        csi=csi,
                        cached_feats=cached_feats,
                        labels=mapped,
                        support_idx=support_idx,
                        query_idx=query_idx,
                        num_classes=num_classes,
                        encoder=encoder,
                        projection=projection,
                        device=device,
                        trial_seed=self.seed + trial,
                    )
                    accs.append(acc)
                    f1s.append(f1)

                results[scope][f"{k}-shot"] = {
                    "accuracy_mean": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)),
                    "f1_mean": float(np.mean(f1s)),
                    "f1_std": float(np.std(f1s)),
                    "k": k,
                    "n_trials": self.n_trials,
                    "num_support": int(k * num_classes),
                    "num_query": int(len(labels) - k * num_classes),
                }
                logger.info(
                    "Few-shot finetune [%s] %d-shot: accuracy=%.4f +/- %.4f "
                    "(%d trials)",
                    scope, k, results[scope][f"{k}-shot"]["accuracy_mean"],
                    results[scope][f"{k}-shot"]["accuracy_std"], self.n_trials,
                )

        return results
