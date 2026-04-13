"""Standalone loss functions for CSI-Text multimodal alignment."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(logits: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE loss.

    Args:
        logits: ``[B, B]`` cosine similarity matrix (raw, unscaled).
        temperature: Softmax temperature scaling.

    Returns:
        Scalar loss.
    """
    logits = logits / temperature
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i2t + loss_t2i)


def sigmoid_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.1,
    bias: float = -10.0,
) -> torch.Tensor:
    """SigLIP sigmoid pairwise loss.

    Args:
        logits: ``[B, B]`` cosine similarity matrix (raw, unscaled).
        positive_mask: ``[B, B]`` binary mask where 1 indicates a positive pair.
        temperature: Sigmoid temperature scaling.
        bias: Learnable bias offset for the logits.

    Returns:
        Scalar loss.
    """
    scaled = logits / temperature + bias
    targets = 2.0 * positive_mask - 1.0
    return -F.logsigmoid(targets * scaled).mean()


def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    lambda_coeff: float = 0.005,
) -> torch.Tensor:
    """Barlow Twins redundancy reduction loss.

    Both inputs should already be projected but **not** standardized;
    standardization is performed internally.

    Args:
        z_a: ``[B, D]`` embeddings from modality A.
        z_b: ``[B, D]`` embeddings from modality B.
        lambda_coeff: Weight on the off-diagonal (redundancy) term.

    Returns:
        Scalar loss.
    """
    z_a = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-5)
    z_b = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-5)

    batch_size = z_a.size(0)
    cross_corr = (z_a.t() @ z_b) / batch_size

    invariance = (cross_corr.diagonal() - 1.0).pow(2).sum()

    n = cross_corr.size(0)
    off_diag = cross_corr.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    redundancy = off_diag.pow(2).sum()

    return invariance + lambda_coeff * redundancy


def cosine_reconstruction_loss(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean cosine reconstruction loss: ``1 - cos_sim``, averaged over the batch.

    Args:
        pred: ``[B, D]`` predicted embeddings.
        target: ``[B, D]`` target embeddings.

    Returns:
        Scalar loss in ``[0, 2]``.
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()
