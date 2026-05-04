"""
Remove the label_index dataset from all .h5 files under F:\\SignFi\\SignFi\\sub_Human.

Rewrites each file that contains label_index by copying all other datasets into
a temp file, then replacing the original. Keeps "amplitude" and any other keys.
"""

import os
import tempfile

import h5py

SUB_HUMAN = r"F:\SignFi\SignFi\sub_Human"
KEY_TO_REMOVE = "label_index"


def iter_h5_paths(root):
    """Yield (dataset_dir, sign_dir, filepath) for each .h5 under root."""
    if not os.path.isdir(root):
        return
    for d in sorted(os.listdir(root)):
        dp = os.path.join(root, d)
        if not os.path.isdir(dp) or d.startswith("sign"):
            continue
        for s in sorted(os.listdir(dp)):
            sp = os.path.join(dp, s)
            if not os.path.isdir(sp) or not s.startswith("sign"):
                continue
            for f in sorted(os.listdir(sp)):
                if not f.endswith(".h5"):
                    continue
                yield os.path.join(sp, f)


def remove_label_index(path):
    """Rewrite file at path without label_index. Return True if changed."""
    with h5py.File(path, "r") as fin:
        if KEY_TO_REMOVE not in fin:
            return False
        keys_to_keep = [k for k in fin.keys() if k != KEY_TO_REMOVE and not k.startswith("__")]
        if not keys_to_keep:
            return False

    fd, tmp = tempfile.mkstemp(suffix=".h5", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with h5py.File(path, "r") as fin, h5py.File(tmp, "w") as fout:
            for k in keys_to_keep:
                fin.copy(k, fout)
        os.replace(tmp, path)
        return True
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def main():
    if not os.path.isdir(SUB_HUMAN):
        raise SystemExit(f"Not found: {SUB_HUMAN}")

    total = 0
    changed = 0
    for path in iter_h5_paths(SUB_HUMAN):
        total += 1
        try:
            if remove_label_index(path):
                changed += 1
        except Exception as e:
            print(f"  Error {path}: {e}")
    print(f"Processed {total:,} .h5 files, removed label_index from {changed:,}.")

if __name__ == "__main__":
    main()
