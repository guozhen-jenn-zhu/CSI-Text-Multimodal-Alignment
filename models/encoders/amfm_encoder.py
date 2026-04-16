"""AM-FM Foundation Model encoder for CSI data.

Adapts the encoder components from the AM-FM pre-trained foundation model
(https://arxiv.org/abs/2602.11200) for use in the CSI-Text multimodal
alignment pipeline.  Only the encoder pathway is included -- reconstruction
decoder, classifier, contrastive head, and ACF head are omitted.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sinusoidal positional encoding for the frequency dimension
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, seq_len_or_x, device=None):
        if isinstance(seq_len_or_x, torch.Tensor):
            seq_len = seq_len_or_x.size(-1)
            device = seq_len_or_x.device
        else:
            seq_len = seq_len_or_x
            if device is None:
                raise ValueError("Device must be specified for integer sequence lengths.")

        pe = torch.zeros(seq_len, self.d_model, device=device)
        position = torch.arange(seq_len, device=device).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


# ---------------------------------------------------------------------------
# Adaptive frequency aggregation via cross-attention
# ---------------------------------------------------------------------------

class _AdaptiveDownsampleEncoder(nn.Module):
    """Cross-attention that compresses F input subcarriers into *freq_out*
    latent frequency channels, weighting subcarriers by information content."""

    def __init__(self, in_channels: int = 1, c_mid: int = 8,
                 c_out: int = 16, freq_out: int = 6):
        super().__init__()
        self.freq_out = freq_out
        self.proj = nn.Linear(in_channels, c_mid)
        self.queries = nn.Parameter(torch.randn(freq_out, c_mid))
        self.fc_out = nn.Linear(c_mid, c_out)
        self.pos_encoder = _PositionalEncoding(d_model=c_mid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_channels, T, F_in)
        B, C, T, F_in = x.shape
        x = x.permute(0, 2, 3, 1)                          # (B, T, F_in, C)
        x_proj = self.proj(x)                               # (B, T, F_in, c_mid)
        pe = self.pos_encoder(x, x.device)                  # (F_in, c_mid)
        x_proj = x_proj + pe.unsqueeze(0).unsqueeze(0)

        attn_scores = torch.einsum("btfc,qc->btqf", x_proj, self.queries)
        attn_scores = attn_scores / math.sqrt(self.queries.shape[-1])
        attn_weights = F.softmax(attn_scores, dim=-1)       # (B, T, freq_out, F_in)

        x_agg = torch.einsum("btqf,btfc->btqc", attn_weights, x_proj)
        x_out = self.fc_out(x_agg)                          # (B, T, freq_out, c_out)
        return x_out.permute(0, 3, 1, 2)                    # (B, c_out, T, freq_out)


# ---------------------------------------------------------------------------
# Learned relative position bias for self-attention
# ---------------------------------------------------------------------------

class _TimeRelativePositionBias(nn.Module):
    """Learns a scalar bias for each relative distance in [-max_len+1, max_len-1]."""

    def __init__(self, max_len: int = 512):
        super().__init__()
        self.max_len = max_len
        self.relative_bias = nn.Embedding(2 * max_len - 1, 1)

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        if device is None:
            device = torch.device("cpu")
        positions = torch.arange(seq_len, device=device)
        distance_mat = positions.unsqueeze(0) - positions.unsqueeze(1)
        distances = distance_mat.clamp(-self.max_len + 1, self.max_len - 1) + (self.max_len - 1)
        return self.relative_bias(distances).squeeze(-1)     # (seq_len, seq_len)


# ---------------------------------------------------------------------------
# Transformer encoder layer with relative position bias
# ---------------------------------------------------------------------------

class _RelativeTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.0, max_len: int = 512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.rel_pos_bias = _TimeRelativePositionBias(max_len=max_len)

    def forward(self, src, src_mask=None, is_causal=False,
                src_key_padding_mask=None):
        B, T, _ = src.shape
        rel_bias = self.rel_pos_bias(T, device=src.device)
        attn_output, _ = self.self_attn(
            query=src.clone(), key=src, value=src,
            attn_mask=rel_bias,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        src = self.norm1(src + self.dropout1(attn_output))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


def _make_relative_transformer_encoder(
    emb_dim: int, num_heads: int, ff_dim: int, dropout: float,
    num_layers: int, max_len: int = 250,
) -> nn.TransformerEncoder:
    layer = _RelativeTransformerEncoderLayer(
        d_model=emb_dim, nhead=num_heads, dim_feedforward=ff_dim,
        dropout=dropout, max_len=max_len,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


# ---------------------------------------------------------------------------
# AMFM_CSI_Encoder – wrapper conforming to the [B, C, K, T] -> [B, embed_dim]
#                     interface used by the rest of the alignment pipeline.
# ---------------------------------------------------------------------------

class AMFM_CSI_Encoder(nn.Module):
    """AM-FM Foundation Model encoder for CSI data.

    Wraps the encoder pathway of the AM-FM ``ViT_MultiTask`` model so that it
    presents the same ``[B, C, K, T] -> [B, embed_dim]`` interface used by the
    other CSI encoders in this project.

    Parameters
    ----------
    input_channels : int
        Number of antenna / Rx-Tx channels (C).
    input_subcarriers : int
        Number of OFDM subcarriers (K) -- used only for interface compat.
    input_timesteps : int
        Number of time steps (T) -- used only for interface compat.
    embed_dim : int
        Output embedding dimension expected by the projection head.
    emb_dim : int
        Internal hidden dimension of the AM-FM transformer.
    encoder_heads : int
        Number of attention heads.
    encoder_layers : int
        Number of transformer encoder layers.
    encoder_ff_dim : int
        Feed-forward dimension.
    encoder_dropout : float
        Dropout in transformer layers.
    c_out : int
        Output channels of the adaptive frequency encoder.
    freq_out : int
        Number of latent frequency bins after cross-attention.
    max_len : int
        Maximum temporal sequence length for relative position bias.
    in_channels : int
        Input channels for the adaptive frequency encoder (typically 1).
    """

    def __init__(
        self,
        input_channels: int,
        input_subcarriers: int,
        input_timesteps: int,
        embed_dim: int = 512,
        *,
        emb_dim: int = 256,
        encoder_heads: int = 8,
        encoder_layers: int = 6,
        encoder_ff_dim: int = 512,
        encoder_dropout: float = 0.1,
        c_out: int = 16,
        freq_out: int = 10,
        max_len: int = 512,
        in_channels: int = 1,
        # these are accepted but unused (interface compat with build_csi_encoder)
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.embed_dim = embed_dim

        self.freq_encoder = _AdaptiveDownsampleEncoder(
            in_channels=in_channels, c_mid=8, c_out=c_out, freq_out=freq_out,
        )
        flat_dim = c_out * freq_out
        self.input_embed = nn.Linear(flat_dim, emb_dim)
        self.encoder = _make_relative_transformer_encoder(
            emb_dim=emb_dim,
            num_heads=encoder_heads,
            ff_dim=encoder_ff_dim,
            dropout=encoder_dropout,
            num_layers=encoder_layers,
            max_len=max_len,
        )

        self.output_proj = nn.Linear(emb_dim, embed_dim)

    # ---- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CSI tensor of shape ``[B, C, K, T]``.

        Returns:
            Embedding of shape ``[B, embed_dim]``.
        """
        B, C, K, T = x.shape
        # Reshape to AM-FM layout: [B, C, T, K]  (channels, time, freq)
        x = x.permute(0, 1, 3, 2)

        # Adaptive frequency aggregation
        x = self.freq_encoder(x)                    # (B, c_out, T, freq_out)
        _, c_out, T_out, F_out = x.shape

        # Flatten freq dims and project
        x = x.permute(0, 2, 1, 3).contiguous()      # (B, T, c_out, freq_out)
        x = x.view(B, T_out, c_out * F_out)         # (B, T, flat_dim)
        x = self.input_embed(x)                      # (B, T, emb_dim)

        # Transformer encoder
        x = self.encoder(x)                          # (B, T, emb_dim)

        # Mean-pool over time (matches AM-FM contrastive path)
        x = x.mean(dim=1)                            # (B, emb_dim)

        return self.output_proj(x)                   # (B, embed_dim)


# ---------------------------------------------------------------------------
# Pre-trained weight loading
# ---------------------------------------------------------------------------

_ENCODER_KEY_PREFIXES = ("freq_encoder.", "input_embed.", "encoder.")
_SKIP_PREFIXES = (
    "reconstruction_transformer.", "classifier.", "contrastive_head.",
    "decoder_input.", "decoder_cnn.", "freq_decoder.", "freq_cnn.",
    "acf_head.", "norm.",
)


def load_pretrained_weights(
    model: AMFM_CSI_Encoder,
    checkpoint_path: str,
    *,
    cache_dir: Optional[str] = None,
) -> None:
    """Load pre-trained AM-FM encoder weights from a PyTorch Lightning
    checkpoint.

    Handles S3 paths (``s3://...``) by downloading to a local cache first.

    Parameters
    ----------
    model : AMFM_CSI_Encoder
        The encoder wrapper to load weights into.
    checkpoint_path : str
        Local path or ``s3://bucket/key`` to a ``.ckpt`` file.
    cache_dir : str, optional
        Directory for caching S3 downloads.  Defaults to
        ``~/.cache/amfm_checkpoints``.
    """
    local_path = _resolve_checkpoint(checkpoint_path, cache_dir)

    logger.info("Loading AM-FM checkpoint from %s", local_path)
    ckpt = torch.load(local_path, map_location="cpu", weights_only=False)

    # Lightning stores weights under "state_dict"
    state_dict = ckpt.get("state_dict", ckpt)

    mapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in state_dict.items():
        # Strip Lightning "model." prefix
        clean = key
        if clean.startswith("model."):
            clean = clean[len("model."):]

        if any(clean.startswith(p) for p in _ENCODER_KEY_PREFIXES):
            mapped[clean] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(mapped, strict=False)

    # output_proj is not in the pretrained checkpoint -- expected to be missing
    truly_missing = [k for k in missing if not k.startswith("output_proj")]
    if truly_missing:
        logger.warning("Missing encoder keys (unexpected): %s", truly_missing)
    if unexpected:
        logger.warning("Unexpected keys in mapped state dict: %s", unexpected)

    logger.info(
        "Loaded %d tensors from AM-FM checkpoint (%d skipped non-encoder keys, "
        "%d missing in wrapper [output_proj expected])",
        len(mapped), len(skipped), len(missing),
    )


def _resolve_checkpoint(path: str, cache_dir: Optional[str] = None) -> str:
    """Download from S3 if *path* starts with ``s3://``, otherwise return as-is."""
    if not path.startswith("s3://"):
        return path

    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".cache", "amfm_checkpoints")
    os.makedirs(cache_dir, exist_ok=True)

    # s3://bucket/key  ->  bucket, key
    without_scheme = path[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    local_name = key.replace("/", "_")
    local_path = os.path.join(cache_dir, local_name)

    if os.path.isfile(local_path):
        logger.info("Using cached checkpoint: %s", local_path)
        return local_path

    logger.info("Downloading checkpoint from s3://%s/%s ...", bucket, key)
    import boto3
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    logger.info("Downloaded checkpoint to %s", local_path)

    return local_path
