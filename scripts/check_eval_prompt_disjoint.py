"""Verify open-vocabulary HAR eval prompts are held out from training text.

For an open-vocabulary evaluation to be valid, the eval prompt families must
not appear in any of the training pair registries.  This script loads one or
more ``pair_registry*.json`` files and checks that none of the prompts in the
new held-out freeform families (and, optionally, all novel families) overlap
with the registries' training texts.

Usage
-----
PYTHONPATH=. python scripts/check_eval_prompt_disjoint.py \
    --registries /path/to/pair_registry.json /path/to/pair_registry_freeform.json

By default only the held-out freeform families are checked.  Pass ``--all``
to check every family in ``OPEN_VOCAB_HAR_FAMILIES`` (note: some existing
families may legitimately overlap with colloquial freeform training text).

Exit code is non-zero if any overlap is found, so it can gate a runbook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.text_generation.templates.open_vocab_har_templates import (
    OPEN_VOCAB_HAR_FAMILIES,
)

# Families that MUST be disjoint from every training registry.
HELD_OUT_FAMILIES = ["freeform_colloquial", "freeform_colloquial_ensemble"]


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _load_registry_texts(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = data.get("texts", {})
    return {_norm(t) for t in texts.values()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check open-vocab eval prompts are held out from training registries",
    )
    parser.add_argument(
        "--registries", nargs="+", required=True,
        help="One or more pair_registry*.json files to check against.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Check every family in OPEN_VOCAB_HAR_FAMILIES, not just the held-out freeform families.",
    )
    args = parser.parse_args()

    families = (
        list(OPEN_VOCAB_HAR_FAMILIES.keys()) if args.all else HELD_OUT_FAMILIES
    )

    # Collect unique eval prompts (normalised) -> family for reporting.
    prompt_to_family: dict[str, str] = {}
    for fam in families:
        if fam not in OPEN_VOCAB_HAR_FAMILIES:
            print(f"WARNING: family '{fam}' not found in OPEN_VOCAB_HAR_FAMILIES")
            continue
        for _cid, texts in OPEN_VOCAB_HAR_FAMILIES[fam].items():
            for t in texts:
                prompt_to_family.setdefault(_norm(t), fam)

    print(f"Checking {len(prompt_to_family)} unique eval prompts "
          f"from {len(families)} families against {len(args.registries)} registries.\n")

    total_overlaps = 0
    for reg_path in args.registries:
        if not Path(reg_path).is_file():
            print(f"ERROR: registry not found: {reg_path}")
            sys.exit(2)
        reg_texts = _load_registry_texts(reg_path)
        overlaps = [
            (prompt_to_family[p], p)
            for p in prompt_to_family
            if p in reg_texts
        ]
        name = Path(reg_path).name
        if overlaps:
            total_overlaps += len(overlaps)
            print(f"[FAIL] {name}: {len(overlaps)} eval prompt(s) overlap with training texts:")
            for fam, p in overlaps:
                print(f"    ({fam}) {p}")
        else:
            print(f"[ OK ] {name}: {len(reg_texts)} training texts, no overlap.")

    print()
    if total_overlaps:
        print(f"DISJOINTNESS CHECK FAILED: {total_overlaps} overlap(s) found.")
        sys.exit(1)
    print("DISJOINTNESS CHECK PASSED: all eval prompts are held out.")


if __name__ == "__main__":
    main()
