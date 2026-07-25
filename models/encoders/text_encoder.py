from __future__ import annotations

import os

import torch
import torch.nn as nn

# ``transformers`` / ``datasets`` probe for TensorFlow, Flax and JAX backends
# when first imported, and a partially-installed backend in the environment
# makes that probe raise rather than degrade. This project is PyTorch-only, so
# opt out of the other backends before the lazy imports below run. Defaults
# only, so an explicit setting from the caller still wins.
for _backend_flag in ("USE_TF", "USE_FLAX", "USE_JAX"):
    os.environ.setdefault(_backend_flag, "0")


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder wrapper using ``open_clip``."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai"):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._text_model = model
        self._dim = model.text_projection.shape[1] if model.text_projection is not None \
            else model.token_embedding.embedding_dim

        for p in self._text_model.parameters():
            p.requires_grad = False
        self._text_model.eval()

    def train(self, mode: bool = True) -> CLIPTextEncoder:
        super().train(mode)
        self._text_model.eval()
        return self

    @property
    def embed_dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Tokenize and encode a list of strings. Returns ``[N, D]``."""
        device = next(self._text_model.parameters()).device
        tokens = self._tokenizer(texts).to(device)
        return self._text_model.encode_text(tokens).float()

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Pass-through for precomputed embeddings."""
        return text_embeddings


class SentenceTransformerEncoder(nn.Module):
    """Frozen ``sentence-transformers`` encoder wrapper."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

    def train(self, mode: bool = True) -> SentenceTransformerEncoder:
        super().train(mode)
        self._model.eval()
        return self

    @property
    def embed_dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings. Returns ``[N, D]``."""
        return torch.from_numpy(
            self._model.encode(texts, convert_to_numpy=True)
        ).float()

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Pass-through for precomputed embeddings."""
        return text_embeddings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TEXT_ENCODERS: dict[str, tuple[type[nn.Module], dict]] = {
    "clip_vit_b32": (CLIPTextEncoder, {"model_name": "ViT-B-32", "pretrained": "openai"}),
    "clip_vit_l14": (CLIPTextEncoder, {"model_name": "ViT-L-14", "pretrained": "openai"}),
    "sentence_transformer": (SentenceTransformerEncoder, {"model_name": "all-MiniLM-L6-v2"}),
}


def build_text_encoder(name: str, **kwargs) -> nn.Module:
    """Instantiate a frozen text encoder by name.

    Args:
        name: One of ``'clip_vit_b32'``, ``'clip_vit_l14'``,
              ``'sentence_transformer'``.
        **kwargs: Overrides forwarded to the encoder constructor.

    Returns:
        A frozen ``nn.Module`` with an ``.encode(texts)`` method and
        an ``.embed_dim`` property.
    """
    key = name.lower()
    if key not in _TEXT_ENCODERS:
        raise ValueError(f"Unknown text encoder '{name}'. "
                         f"Available: {list(_TEXT_ENCODERS.keys())}")
    cls, defaults = _TEXT_ENCODERS[key]
    merged = {**defaults, **kwargs}
    return cls(**merged)
