from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIP(nn.Module):
    """SigLIP: Sigmoid pairwise loss for many-to-many alignment."""

    def __init__(
        self,
        csi_encoder: nn.Module,
        text_projection: nn.Module,
        csi_projection: nn.Module,
        temperature: float = 0.1,
        bias: float = -10.0,
    ) -> None:
        super().__init__()
        self.csi_encoder = csi_encoder
        self.text_projection = text_projection
        self.csi_projection = csi_projection
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(
        self,
        csi: torch.Tensor,
        text_embeddings: torch.Tensor,
        positive_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            csi: Raw CSI input, shape ``[B, C, K, T]``.
            text_embeddings: Precomputed text features, shape ``[B, D_text]``.
            positive_mask: Optional ``[B, B]`` binary mask where 1 = positive pair.
                If ``None``, the diagonal is treated as positive (standard pairing).

        Returns:
            Dict with ``loss``, ``csi_embeddings``, ``text_embeddings``.
        """
        csi_features = self.csi_encoder(csi)
        csi_proj = F.normalize(self.csi_projection(csi_features), dim=-1)
        text_proj = F.normalize(self.text_projection(text_embeddings), dim=-1)

        temperature = self.log_temperature.exp()
        logits = csi_proj @ text_proj.t() / temperature + self.bias

        batch_size = logits.size(0)
        if positive_mask is None:
            positive_mask = torch.eye(batch_size, device=logits.device)

        targets = 2.0 * positive_mask - 1.0  # +1 for positives, -1 for negatives
        loss = -F.logsigmoid(targets * logits).mean()

        return {
            "loss": loss,
            "csi_embeddings": csi_proj,
            "text_embeddings": text_proj,
        }
