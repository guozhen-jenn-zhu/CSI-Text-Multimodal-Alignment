from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection head mapping encoder output to a shared embedding space.

    Produces L2-normalised embeddings suitable for contrastive losses.
    """

    def __init__(self, input_dim: int, output_dim: int = 512,
                 hidden_dim: int = 1024, num_layers: int = 2,
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for i in range(num_layers):
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(out_d))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_d = out_d
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)
