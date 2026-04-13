from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageBind(nn.Module):
    """Anchor-based binding with frozen text encoder.

    Text embeddings serve as fixed targets; only the CSI encoder and its
    projection head are trained.
    """

    def __init__(
        self,
        csi_encoder: nn.Module,
        csi_projection: nn.Module,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.csi_encoder = csi_encoder
        self.csi_projection = csi_projection
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    def forward(
        self, csi: torch.Tensor, text_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            csi: Raw CSI input, shape ``[B, C, K, T]``.
            text_embeddings: Precomputed (frozen) text features, shape ``[B, D_text]``.

        Returns:
            Dict with ``loss``, ``csi_embeddings``, ``text_embeddings``.
        """
        csi_features = self.csi_encoder(csi)
        csi_proj = F.normalize(self.csi_projection(csi_features), dim=-1)
        text_proj = F.normalize(text_embeddings.detach(), dim=-1)

        temperature = self.log_temperature.exp()
        logits = csi_proj @ text_proj.t() / temperature

        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "csi_embeddings": csi_proj,
            "text_embeddings": text_proj,
        }
