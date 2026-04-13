from .barlow_twins import BarlowTwins
from .centrobind import CentroBind
from .clip_infonce import CLIPInfoNCE
from .language_bind import LanguageBind
from .masked_predict import MaskedCrossModalPrediction
from .siglip import SigLIP

__all__ = [
    "BarlowTwins",
    "CentroBind",
    "CLIPInfoNCE",
    "LanguageBind",
    "MaskedCrossModalPrediction",
    "SigLIP",
    "build_alignment_model",
]

_REGISTRY: dict[str, type] = {
    "clip_infonce": CLIPInfoNCE,
    "siglip": SigLIP,
    "language_bind": LanguageBind,
    "centrobind": CentroBind,
    "barlow_twins": BarlowTwins,
    "masked_predict": MaskedCrossModalPrediction,
}


def build_alignment_model(
    method_name: str,
    csi_encoder,
    csi_projection,
    text_projection=None,
    **kwargs,
):
    """Instantiate an alignment model by config name.

    Args:
        method_name: One of the registered method keys (e.g. ``"clip_infonce"``).
        csi_encoder: CSI backbone encoder module.
        csi_projection: Projection head for CSI embeddings.
        text_projection: Projection head for text embeddings.
            Not required for ``language_bind`` (which uses raw text embeddings).
        **kwargs: Additional keyword arguments forwarded to the constructor.

    Returns:
        An ``nn.Module`` implementing the requested alignment strategy.

    Raises:
        ValueError: If *method_name* is not in the registry.
    """
    if method_name not in _REGISTRY:
        raise ValueError(
            f"Unknown alignment method '{method_name}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    cls = _REGISTRY[method_name]

    if method_name == "language_bind":
        return cls(
            csi_encoder=csi_encoder,
            csi_projection=csi_projection,
            **kwargs,
        )

    return cls(
        csi_encoder=csi_encoder,
        text_projection=text_projection,
        csi_projection=csi_projection,
        **kwargs,
    )
