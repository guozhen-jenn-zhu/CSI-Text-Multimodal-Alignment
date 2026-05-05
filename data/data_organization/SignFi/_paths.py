"""Shared path resolution for the SignFi data-organization scripts.

All five `step*.py` scripts in this folder produce or consume the same SignFi
root directory. By default the root is

    <repo_root>/data/Dataset/signfi/sign_recognition/

so the produced layout drops directly into the location the trainer reads
from (see ``configs/dataset/signfi.yaml`` and the ``signfi/sign_recognition``
``task_root`` entries in ``configs/experiment/multitask_*.yaml``).

Override order (highest precedence first):
    1. ``--root <path>`` on the script's command line (when supported)
    2. ``SIGNFI_ROOT`` environment variable
    3. ``<repo_root>/data/Dataset/signfi/sign_recognition``

The released layout under the root is::

    <signfi_root>/
    ├── sub_Human/
    │   └── <dataset_name>/
    │       └── sign{NNN}/
    │           └── <sample>.h5
    ├── metadata/
    │   └── sample_metadata.csv
    └── splits/
        ├── train_id.json
        ├── val_id.json
        └── test_id.json
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives at <repo_root>/data/data_organization/SignFi/_paths.py,
# so the repo root is three directories up.
_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SIGNFI_ROOT = _REPO_ROOT / "data" / "Dataset" / "signfi" / "sign_recognition"

SUB_HUMAN_DIRNAME = "sub_Human"
METADATA_DIRNAME = "metadata"
SPLITS_DIRNAME = "splits"


def signfi_root(cli_root: str | os.PathLike | None = None) -> Path:
    """Resolve the SignFi root directory.

    Order of precedence: ``cli_root`` > ``$SIGNFI_ROOT`` > the repo default
    (``<repo_root>/data/Dataset/signfi/sign_recognition``).
    """
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    env = os.environ.get("SIGNFI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_SIGNFI_ROOT


def sub_human_dir(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<signfi_root>/sub_Human``."""
    return signfi_root(cli_root) / SUB_HUMAN_DIRNAME


def metadata_csv_path(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<signfi_root>/metadata/sample_metadata.csv``."""
    return signfi_root(cli_root) / METADATA_DIRNAME / "sample_metadata.csv"


def splits_dir(cli_root: str | os.PathLike | None = None) -> Path:
    """Return ``<signfi_root>/splits``."""
    return signfi_root(cli_root) / SPLITS_DIRNAME
