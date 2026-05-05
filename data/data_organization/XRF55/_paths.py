"""Shared path resolution for the XRF55 data-organization scripts.

All eight ``step*.py`` scripts in this folder produce or consume a layout
rooted at the same XRF55 task directory. By default the root is

    <repo_root>/data/Dataset/xrf55/action_recognition/

so the produced layout drops directly into the location the trainer reads
from (see ``configs/dataset/xrf55.yaml`` and the ``xrf55/action_recognition``
``task_root`` entries in ``configs/experiment/multitask_*.yaml``).

Override order (highest precedence first):
    1. ``--root <path>`` on the script's command line (when supported)
    2. ``XRF55_ROOT`` environment variable
    3. ``<repo_root>/data/Dataset/xrf55/action_recognition``

Conventional staging layout under the root::

    <xrf55_root>/
    ├── raw/
    │   └── Scene<N>/WiFi/<user>_<act>_<repeat>.npy   # upstream archive (user-provided)
    ├── organized/
    │   └── user_<XX>/act_<XX>/env_<N>/...            # step1 (.npy) + step2 (.h5)
    │                                                 # step3 renames to userXX_actXX_repeatXX_envN.<ext>
    ├── organized_npy/                                # step4 archives the .npy half here
    │   └── user_<XX>/act_<XX>/env_<N>/<file>.npy
    ├── user_<XX>/act_<XX>/env_<N>/<file>.h5          # step5 (final canonical .h5 layout)
    ├── metadata/
    │   └── sample_metadata.csv                       # step6
    └── splits/                                       # step7 + step8
        ├── train_id.json
        ├── val_id.json
        └── test_id.json

The released archive ships a fully-built version of this layout, so most users
do not need to run these scripts; they're kept for transparency and to allow
re-building from a fresh upstream pull.
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives at <repo_root>/data/data_organization/XRF55/_paths.py,
# so the repo root is three directories up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_XRF55_ROOT = _REPO_ROOT / "data" / "Dataset" / "xrf55" / "action_recognition"

RAW_DIRNAME = "raw"
ORGANIZED_DIRNAME = "organized"
ORGANIZED_NPY_DIRNAME = "organized_npy"
METADATA_DIRNAME = "metadata"
SPLITS_DIRNAME = "splits"
METADATA_FILENAME = "sample_metadata.csv"


def xrf55_root(cli_root: str | os.PathLike | None = None) -> Path:
    """Resolve the XRF55 root directory.

    Order of precedence: ``cli_root`` > ``$XRF55_ROOT`` > the repo default
    (``<repo_root>/data/Dataset/xrf55/action_recognition``).
    """
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env = os.environ.get("XRF55_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_XRF55_ROOT


def raw_dir(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<xrf55_root>/raw`` -- upstream Scene<N>/WiFi/*.npy archives live here."""
    return xrf55_root(cli_root) / RAW_DIRNAME


def organized_dir(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<xrf55_root>/organized`` -- staging layout produced by step1 and consumed by steps 2-5."""
    return xrf55_root(cli_root) / ORGANIZED_DIRNAME


def organized_npy_dir(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<xrf55_root>/organized_npy`` -- archive of .npy files separated by step4."""
    return xrf55_root(cli_root) / ORGANIZED_NPY_DIRNAME


def metadata_csv_path(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<xrf55_root>/metadata/sample_metadata.csv``."""
    return xrf55_root(cli_root) / METADATA_DIRNAME / METADATA_FILENAME


def splits_dir(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<xrf55_root>/splits``."""
    return xrf55_root(cli_root) / SPLITS_DIRNAME
