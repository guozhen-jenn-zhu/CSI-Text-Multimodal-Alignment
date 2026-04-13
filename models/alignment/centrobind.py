from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentroBind(nn.Module):
    """CENTROBIND: Adaptive centroid-based multimodal alignment.

    Maintains a learnable centroid bank updated via EMA and aligns both
    modalities to the centroids as well as to each other.
    """

    def __init__(
        self,
        csi_encoder: nn.Module,
        text_projection: nn.Module,
        csi_projection: nn.Module,
        embed_dim: int = 512,
        num_centroids: int = 16,
        temperature: float = 0.07,
        momentum: float = 0.999,
    ) -> None:
        super().__init__()
        self.csi_encoder = csi_encoder
        self.text_projection = text_projection
        self.csi_projection = csi_projection
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.momentum = momentum
        self.num_centroids = num_centroids

        self.centroids = nn.Parameter(torch.randn(num_centroids, embed_dim))
        nn.init.xavier_uniform_(self.centroids.data)

        self.register_buffer(
            "ema_centroids", torch.zeros(num_centroids, embed_dim)
        )
        self.register_buffer("ema_initialized", torch.tensor(False))

    @torch.no_grad()
    def _update_ema_centroids(self, embeddings: torch.Tensor) -> None:
        """Update running centroid estimates with soft assignments."""
        assignments = F.softmax(
            embeddings @ self.centroids.t(), dim=-1
        )  # [N, K]
        new_centroids = assignments.t() @ embeddings  # [K, D]
        new_centroids = F.normalize(new_centroids, dim=-1)

        if not self.ema_initialized:
            self.ema_centroids.copy_(new_centroids)
            self.ema_initialized.fill_(True)
        else:
            self.ema_centroids.mul_(self.momentum).add_(
                new_centroids, alpha=1.0 - self.momentum
            )
        self.centroids.data.copy_(F.normalize(self.ema_centroids, dim=-1))

    def _centroid_contrastive_loss(
        self, embeddings: torch.Tensor, temperature: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss between embeddings and their nearest centroids."""
        sim = embeddings @ self.centroids.t() / temperature  # [B, K]
        targets = sim.detach().argmax(dim=-1)
        return F.cross_entropy(sim, targets)

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
        csi_proj = F.normalize(self.csi_projection(csi_features), dim=-1)
        text_proj = F.normalize(self.text_projection(text_embeddings), dim=-1)

        temperature = self.log_temperature.exp()

        # Direct cross-modal contrastive loss
        logits = csi_proj @ text_proj.t() / temperature
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        cross_modal_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)
        )

        # Centroid alignment losses
        csi_centroid_loss = self._centroid_contrastive_loss(csi_proj, temperature)
        text_centroid_loss = self._centroid_contrastive_loss(text_proj, temperature)

        if self.training:
            all_embeddings = torch.cat([csi_proj, text_proj], dim=0)
            self._update_ema_centroids(all_embeddings)

        loss = cross_modal_loss + 0.5 * (csi_centroid_loss + text_centroid_loss)

        return {
            "loss": loss,
            "csi_embeddings": csi_proj,
            "text_embeddings": text_proj,
        }
