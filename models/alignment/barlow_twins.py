from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowTwins(nn.Module):
    """Barlow Twins: Non-contrastive cross-correlation alignment."""

    def __init__(
        self,
        csi_encoder: nn.Module,
        text_projection: nn.Module,
        csi_projection: nn.Module,
        lambda_coeff: float = 0.005,
    ) -> None:
        super().__init__()
        self.csi_encoder = csi_encoder
        self.text_projection = text_projection
        self.csi_projection = csi_projection
        self.lambda_coeff = lambda_coeff

    @staticmethod
    def _standardize(z: torch.Tensor) -> torch.Tensor:
        """Zero-mean, unit-variance normalization across the batch dimension."""
        return (z - z.mean(dim=0)) / (z.std(dim=0) + 1e-5)

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(
        self, csi: torch.Tensor, text_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            csi: Raw CSI input, shape ``[B, C, K, T]``.
            text_embeddings: Precomputed text features, shape ``[B, D_text]``.

        Returns:
            Dict with ``loss``, ``csi_embeddings``, ``text_embeddings``.
        """
        csi_features = self.csi_encoder(csi)
        csi_proj = self.csi_projection(csi_features)
        text_proj = self.text_projection(text_embeddings)

        batch_size = csi_proj.size(0)

        z_csi = self._standardize(csi_proj)
        z_text = self._standardize(text_proj)

        cross_corr = (z_csi.t() @ z_text) / batch_size  # [D, D]

        invariance_loss = (cross_corr.diagonal() - 1.0).pow(2).sum()
        redundancy_loss = self._off_diagonal(cross_corr).pow(2).sum()
        loss = invariance_loss + self.lambda_coeff * redundancy_loss

        csi_norm = F.normalize(csi_proj, dim=-1)
        text_norm = F.normalize(text_proj, dim=-1)

        return {
            "loss": loss,
            "csi_embeddings": csi_norm,
            "text_embeddings": text_norm,
        }
