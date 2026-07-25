#!/usr/bin/env python3
"""Pick the checkpoint file in a run directory that matches a target architecture.

Several checkpoint directories hold weights from more than one training run.
Priority 0 trained resnet18_1d, vit_1d and patchtst sequentially into
``csibench-har-clip_infonce/``, and the Priority 5B sentence-transformer
ablation later wrote its own ``best.pt`` there (see
``paper_results/RESULTS_STATUS.md`` and ``scripts/finish_missing_evals.sh``).
``best.pt`` in that directory is a sentence-transformer model scoring 90.2%
zero-shot on HAR, while the published 98.7% comes from ``epoch_150_p0_clip.pt``
(resnet18_1d + clip_vit_b32).  Any later evaluation that just globs ``best.pt``
silently reports numbers for the wrong model.

Every checkpoint embeds its training config, so the intended file can be found
by reading that config back rather than trusting the filename.

Usage::

    # Resolve within a directory (prints the chosen path on stdout)
    python -m scripts.resolve_checkpoint \\
        --checkpoint_dir "$MODEL_ROOT/checkpoints/csibench-har-clip_infonce" \\
        --encoder resnet18_1d --text_encoder clip_vit_b32 --method clip_infonce

    # Verify one specific file
    python -m scripts.resolve_checkpoint \\
        --checkpoint "$MODEL_ROOT/checkpoints/csibench-har-clip_infonce/best.pt" \\
        --encoder resnet18_1d --text_encoder clip_vit_b32

    # List every candidate with its embedded config
    python -m scripts.resolve_checkpoint --checkpoint_dir "$CKPT_DIR" --info

Exit codes: 0 resolved, 1 bad arguments / missing path, 2 nothing matched.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Data-variant retrains keep the task suffix out of the class-text vocabulary
# (proximity_fixed is still the proximity task, on corrected 4-class metadata).
TASK_SUFFIXES = ("_fixed", "_fourclass")


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_meta(path: Path) -> Dict[str, Any]:
    """Read architecture provenance out of a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError):
        # mmap needs the zipfile format and torch >= 2.1; fall back to a full read.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "config" not in ckpt:
        # A bare state_dict carries no provenance at all.
        return {"has_config": False}

    cfg = ckpt.get("config") or {}
    task = cfg.get("task")
    if not task and cfg.get("tasks"):
        task = ",".join(str(t) for t in cfg["tasks"])
    return {
        "has_config": bool(cfg),
        "encoder": (cfg.get("csi_encoder") or {}).get("name"),
        "text_encoder": (cfg.get("text_encoder") or {}).get("name"),
        "method": cfg.get("method"),
        "task": task,
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
    }


def _normalize_task(task: Optional[str]) -> Optional[str]:
    if not task:
        return None
    for suffix in TASK_SUFFIXES:
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


def _mismatches(meta: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    """Return the hard mismatches between a checkpoint and the requested model."""
    problems: List[str] = []
    checks = (
        ("encoder", args.encoder),
        ("text_encoder", args.text_encoder),
        ("method", args.method),
    )
    for key, want in checks:
        if not want:
            continue
        got = meta.get(key)
        if got is not None and str(got) != want:
            problems.append(f"{key}={got} (want {want})")
    return problems


def _describe(path: Path, meta: Dict[str, Any]) -> str:
    if not meta.get("has_config"):
        return f"{path.name}: no embedded config"
    val_loss = meta.get("val_loss")
    val_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else "?"
    return (
        f"{path.name}: encoder={meta.get('encoder')} "
        f"text_encoder={meta.get('text_encoder')} "
        f"method={meta.get('method')} task={meta.get('task')} "
        f"epoch={meta.get('epoch')} val_loss={val_str}"
    )


def _candidates(ckpt_dir: Path, prefer: List[str], limit: int) -> List[Path]:
    """Order the .pt files so explicit preferences and best.pt are read first."""
    ordered: List[Path] = []
    for name in prefer:
        path = ckpt_dir / name
        if path.is_file() and path not in ordered:
            ordered.append(path)

    best = ckpt_dir / "best.pt"
    if best.is_file() and best not in ordered:
        ordered.append(best)

    for path in sorted(ckpt_dir.glob("*.pt")):
        if path not in ordered:
            ordered.append(path)

    return ordered[:limit]


def _resolve(paths: List[Path], args: argparse.Namespace) -> Optional[Path]:
    """Read each candidate and pick the one matching the requested architecture."""
    matches: List[tuple] = []
    unverifiable: List[Path] = []

    for rank, path in enumerate(paths):
        try:
            meta = _load_meta(path)
        except Exception as e:  # unreadable / truncated download
            _log(f"  skip {path.name}: cannot read ({e})")
            continue

        if not meta.get("has_config"):
            _log(f"  {_describe(path, meta)} -> unverifiable")
            unverifiable.append(path)
            continue

        problems = _mismatches(meta, args)
        want_task = _normalize_task(args.task)
        got_task = _normalize_task(meta.get("task"))
        if want_task and got_task and want_task != got_task and not problems:
            _log(f"  {_describe(path, meta)} -> task differs (want {want_task}); still eligible")

        if problems:
            _log(f"  {_describe(path, meta)} -> MISMATCH: {', '.join(problems)}")
            continue

        _log(f"  {_describe(path, meta)} -> match")
        val_loss = meta.get("val_loss")
        sort_key = val_loss if isinstance(val_loss, (int, float)) else math.inf
        # Explicit --prefer entries win outright; otherwise the lowest val_loss
        # among matching files is the one training would have called "best".
        is_preferred = path.name in args.prefer
        matches.append((0 if is_preferred else 1, rank if is_preferred else sort_key, rank, path))

    if matches:
        matches.sort(key=lambda m: (m[0], m[1], m[2]))
        return matches[0][3]

    if unverifiable and args.allow_unverified:
        _log(f"  no verifiable match; falling back to {unverifiable[0].name}")
        return unverifiable[0]

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint_dir", help="Directory of .pt files to search")
    src.add_argument("--checkpoint", help="Verify this single checkpoint file")
    parser.add_argument("--encoder", default=None, help="Required csi_encoder.name")
    parser.add_argument("--text_encoder", default=None, help="Required text_encoder.name")
    parser.add_argument("--method", default=None, help="Required alignment method")
    parser.add_argument(
        "--task",
        default=None,
        help="Expected task; a differing task is reported but not rejected "
             "(data-variant retrains such as proximity_fixed keep their own tag).",
    )
    parser.add_argument(
        "--prefer",
        action="append",
        default=[],
        help="Filename to try before best.pt (repeatable)",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=12,
        help="Stop after reading this many files (default: 12)",
    )
    parser.add_argument(
        "--allow_unverified",
        action="store_true",
        help="Accept a checkpoint without an embedded config if nothing matches",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print the embedded config of every candidate and exit",
    )
    args = parser.parse_args()

    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            _log(f"ERROR: checkpoint not found: {path}")
            return 1
        paths = [path]
        ckpt_dir = path.parent
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_dir():
            _log(f"ERROR: checkpoint directory not found: {ckpt_dir}")
            return 1
        paths = _candidates(ckpt_dir, args.prefer, args.max_candidates)
        if not paths:
            _log(f"ERROR: no .pt files in {ckpt_dir}")
            return 2

    if args.info:
        for path in paths:
            try:
                _log(_describe(path, _load_meta(path)))
            except Exception as e:
                _log(f"{path.name}: cannot read ({e})")
        return 0

    if args.checkpoint:
        _log(f"Verifying {paths[0]}")
    else:
        _log(f"Resolving checkpoint in {ckpt_dir}")

    resolved = _resolve(paths, args)
    if resolved is None:
        wanted = ", ".join(
            f"{k}={v}" for k, v in (
                ("encoder", args.encoder),
                ("text_encoder", args.text_encoder),
                ("method", args.method),
            ) if v
        )
        if args.checkpoint:
            _log(f"ERROR: {paths[0]} was not trained with {wanted}.")
        else:
            _log(
                f"ERROR: no checkpoint in {ckpt_dir} matches {wanted}.\n"
                "       The directory may hold weights from a later run that "
                "overwrote best.pt."
            )
        _log(
            "       List every candidate with --info, or restore the original "
            "weights from S3 (see scripts/finish_missing_evals.sh)."
        )
        return 2

    _log(f"Using {resolved}")
    print(resolved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
