from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCrossModalPrediction(nn.Module):
    """Generative alignment via masked cross-modal prediction.

    Randomly masks portions of the CSI input and trains the model to predict
    the corresponding text embedding, combined with a contrastive alignment
    objective on the unmasked branch.
    """

    def __init__(
        self,
        csi_encoder: nn.Module,
        text_projection: nn.Module,
        csi_projection: nn.Module,
        embed_dim: int = 512,
        mask_ratio: float = 0.75,
        reconstruction_weight: float = 1.0,
        alignment_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.csi_encoder = csi_encoder
        self.text_projection = text_projection
        self.csi_projection = csi_projection
        self.mask_ratio = mask_ratio
        self.reconstruction_weight = reconstruction_weight
        self.alignment_weight = alignment_weight

        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    @staticmethod
    def _random_mask(
        csi: torch.Tensor, mask_ratio: float
    ) -> torch.Tensor:
        """Apply a random temporal mask by zeroing out time-step slices.

        Args:
            csi: ``[B, C, K, T]``
            mask_ratio: Fraction of time steps to zero out.

        Returns:
            Masked CSI tensor of the same shape.
        """
        B, C, K, T = csi.shape
        num_masked = max(1, int(T * mask_ratio))
        noise = torch.rand(B, T, device=csi.device)
        ids_to_mask = noise.argsort(dim=-1)[:, :num_masked]

        mask = torch.ones(B, T, device=csi.device)
        mask.scatter_(1, ids_to_mask, 0.0)
        return csi * mask.unsqueeze(1).unsqueeze(2)  # broadcast over C, K

    def forward(
        self, csi: torch.Tensor, text_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            csi: Raw CSI input, shape ``[B, C, K, T]``.
            text_embeddings: Precomputed text features, shape ``[B, D_text]``.

        Returns:
            Dict with ``loss``, ``csi_embeddings``, ``text_embeddings``,
            ``reconstruction_loss``, and ``alignment_loss``.
        """
        text_proj = F.normalize(self.text_projection(text_embeddings), dim=-1)

        # --- masked branch: reconstruction objective ---
        masked_csi = self._random_mask(csi, self.mask_ratio)
        masked_features = self.csi_encoder(masked_csi)
        masked_proj = self.csi_projection(masked_features)
        predicted_text = self.reconstruction_head(masked_proj)
        predicted_text = F.normalize(predicted_text, dim=-1)
        reconstruction_loss = 1.0 - (predicted_text * text_proj).sum(dim=-1).mean()

        # --- full branch: contrastive alignment objective ---
        csi_features = self.csi_encoder(csi)
        csi_proj = F.normalize(self.csi_projection(csi_features), dim=-1)

        logits = csi_proj @ text_proj.t() / 0.07
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        alignment_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.t(), labels)
        )

        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.alignment_weight * alignment_loss
        )

        return {
            "loss": loss,
            "csi_embeddings": csi_proj,
            "text_embeddings": text_proj,
            "reconstruction_loss": reconstruction_loss,
            "alignment_loss": alignment_loss,
        }
