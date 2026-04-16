from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ResNet-18 adapted for 1D CSI signals
# ---------------------------------------------------------------------------

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNet18_1D(nn.Module):
    """ResNet-18 with 1D convolutions for CSI data.

    Input : [B, C, K, T]  (batch, antenna-channels, subcarriers, timesteps)
    Output: [B, embed_dim]
    """

    def __init__(self, input_channels: int, input_subcarriers: int,
                 input_timesteps: int, embed_dim: int = 512,
                 base_width: int = 64, dropout: float = 0.0):
        super().__init__()
        in_chan = input_channels * input_subcarriers
        self.conv1 = nn.Conv1d(in_chan, base_width, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_width)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(base_width, base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width, base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(base_width * 8, embed_dim)

        self._init_weights()

    # ---- internal helpers ----

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers: list[nn.Module] = [
            BasicBlock1D(in_channels, out_channels, stride, downsample)
        ]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- forward ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, K, T = x.shape
        x = x.reshape(B, C * K, T)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# PatchTST – Patch-based Transformer for time-series CSI
# ---------------------------------------------------------------------------

class PatchTST_CSI(nn.Module):
    """Patch Time-Series Transformer for CSI data.

    Input : [B, C, K, T]
    Output: [B, embed_dim]
    """

    def __init__(self, input_channels: int, input_subcarriers: int,
                 input_timesteps: int, embed_dim: int = 512,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 pool: str = "cls"):
        super().__init__()
        assert pool in ("cls", "mean")
        self.pool = pool
        in_chan = input_channels * input_subcarriers
        self.patch_len = patch_len
        self.stride = stride

        num_patches = (input_timesteps - patch_len) // stride + 1
        self.patch_proj = nn.Linear(in_chan * patch_len, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, D, T] -> [B, N, D*patch_len]"""
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        B, D, N, P = patches.shape
        return patches.permute(0, 2, 1, 3).reshape(B, N, D * P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, K, T = x.shape
        x = x.reshape(B, C * K, T)

        x = self._create_patches(x)          # [B, N, D*P]
        x = self.patch_proj(x)               # [B, N, d_model]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)       # [B, N+1, d_model]
        x = x + self.pos_embed[:, : x.size(1)]

        x = self.transformer(x)
        x = self.norm(x)

        if self.pool == "cls":
            x = x[:, 0]
        else:
            x = x[:, 1:].mean(dim=1)

        return self.head(x)


# ---------------------------------------------------------------------------
# ViT-1D – Vision Transformer adapted for 1D CSI signals
# ---------------------------------------------------------------------------

class ViT_1D(nn.Module):
    """Vision Transformer with 1D patch embedding for CSI data.

    Input : [B, C, K, T]
    Output: [B, embed_dim]
    """

    def __init__(self, input_channels: int, input_subcarriers: int,
                 input_timesteps: int, embed_dim: int = 512,
                 patch_size: int = 4, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        in_chan = input_channels * input_subcarriers
        assert input_timesteps % patch_size == 0, \
            f"input_timesteps ({input_timesteps}) must be divisible by patch_size ({patch_size})"
        num_patches = input_timesteps // patch_size

        self.patch_embed = nn.Conv1d(in_chan, d_model, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, K, T = x.shape
        x = x.reshape(B, C * K, T)

        x = self.patch_embed(x)              # [B, d_model, num_patches]
        x = x.transpose(1, 2)               # [B, num_patches, d_model]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # [B, num_patches+1, d_model]
        x = x + self.pos_embed[:, : x.size(1)]

        x = self.transformer(x)
        x = self.norm(x)

        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CSI_ENCODERS: dict[str, type[nn.Module]] = {
    "resnet18_1d": ResNet18_1D,
    "patchtst": PatchTST_CSI,
    "vit_1d": ViT_1D,
}


def build_csi_encoder(name: str, input_channels: int, input_subcarriers: int,
                      input_timesteps: int, embed_dim: int = 512,
                      **kwargs) -> nn.Module:
    """Instantiate a CSI encoder by name.

    Args:
        name: One of ``'resnet18_1d'``, ``'patchtst'``, ``'vit_1d'``,
              ``'amfm'``.
        input_channels: Number of antenna / Rx-Tx channels (C).
        input_subcarriers: Number of OFDM subcarriers (K).
        input_timesteps: Number of time steps (T).
        embed_dim: Output embedding dimension.
        **kwargs: Extra arguments forwarded to the encoder constructor.
            For ``amfm``, the key ``pretrained_path`` triggers loading of
            pre-trained weights from a local file or S3 URI.

    Returns:
        An ``nn.Module`` mapping ``[B, C, K, T] -> [B, embed_dim]``.
    """
    key = name.lower()

    # AM-FM uses a separate module and supports pre-trained weight loading
    if key == "amfm":
        from .amfm_encoder import AMFM_CSI_Encoder, load_pretrained_weights

        pretrained_path = kwargs.pop("pretrained_path", None)
        encoder = AMFM_CSI_Encoder(
            input_channels=input_channels,
            input_subcarriers=input_subcarriers,
            input_timesteps=input_timesteps,
            embed_dim=embed_dim,
            **kwargs,
        )
        if pretrained_path:
            load_pretrained_weights(encoder, pretrained_path)
        else:
            logger.info("AM-FM encoder created without pre-trained weights")
        return encoder

    if key not in _CSI_ENCODERS:
        raise ValueError(f"Unknown CSI encoder '{name}'. "
                         f"Available: {list(_CSI_ENCODERS.keys()) + ['amfm']}")
    return _CSI_ENCODERS[key](
        input_channels=input_channels,
        input_subcarriers=input_subcarriers,
        input_timesteps=input_timesteps,
        embed_dim=embed_dim,
        **kwargs,
    )
