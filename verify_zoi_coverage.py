"""
ZOI label coverage guard — the consistency check between sample_zoi_dataset.py
(which frames training will load) and the label generator (which frames got labeled).

The two are linked only by the _manifest/*.txt frame list. Nothing forces the
generator to have actually produced a well-formed .npy for every manifest frame, and
data.py silently treats a missing label as all-zero + mask-off (no crash, just no ZOI
supervision for that frame). This script makes that gap loud: run it after labeling
and BEFORE training to confirm every selected frame is covered and valid.

Checks, per frame listed in the manifest(s):
  - the label .npy exists at <labels_root>/<route_rel>/<out_subdir>/<stem>.npy
  - it is a float [M,4] array, all finite
  - importance column in [0,1]; class_id column in {0,1,2,3} (car/walker/light/stop)
  - M==0 (frame had no kept GT boxes) is VALID but counted separately, so you know how
    many selected frames carry zero supervision.

Exit code is nonzero if coverage < --min_coverage or any label is malformed, so it can
gate a training launch (`python verify_zoi_coverage.py ... && torchrun train.py ...`).

Usage:
  python verify_zoi_coverage.py \
      --manifest <out_root>/_manifest/train_labels.txt <out_root>/_manifest/eval_labels.txt \
      --labels_root <out_root>        # where labels live; default = manifest's <out_root>
"""

import argparse
import os
import sys

import numpy as np

VALID_CLASS_IDS = {0, 1, 2, 3}   # car, walker, traffic_light, stop_sign (zoi_projection.CLASS_TO_ID)


def validate_npy(path):
    """Return (ok, n_rows, reason). ok=False with reason on malformed; n_rows may be 0."""
    try:
        a = np.load(path)
    except Exception as e:
        return False, 0, f"load failed: {e}"
    if a.ndim != 2 or a.shape[1] != 4:
        return False, 0, f"shape {a.shape} != [M,4]"
    if a.shape[0] == 0:
        return True, 0, "empty"
    if not np.isfinite(a).all():
        return False, a.shape[0], "non-finite values"
    imp = a[:, 2]
    if imp.min() < -1e-6 or imp.max() > 1.0 + 1e-6:
        return False, a.shape[0], f"importance out of [0,1]: [{imp.min():.3f},{imp.max():.3f}]"
    cls = set(int(round(c)) for c in a[:, 3].tolist())
    if not cls.issubset(VALID_CLASS_IDS):
        return False, a.shape[0], f"class_id(s) {cls - VALID_CLASS_IDS} not in {VALID_CLASS_IDS}"
    return True, a.shape[0], "ok"


def read_manifest(path):
    """Yield (route_rel, stem) from a sample_zoi_dataset manifest line '<scn>/<route> <stem>'."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                route_rel, stem = line.rsplit(" ", 1)
                yield route_rel, stem


def check_manifest(manifest, labels_root, out_subdir, show):
    total = present = empty = 0
    missing, malformed = [], []
    for route_rel, stem in read_manifest(manifest):
        npy = os.path.join(labels_root, route_rel, out_subdir, stem + ".npy")
        total += 1
        if not os.path.isfile(npy):
            missing.append(f"{route_rel} {stem}")
            continue
        ok, n_rows, reason = validate_npy(npy)
        if not ok:
            malformed.append(f"{route_rel} {stem}: {reason}")
            continue
        present += 1
        if n_rows == 0:
            empty += 1

    name = os.path.basename(manifest)
    cov = present / total if total else 0.0
    print(f"\n=== {name} ===")
    print(f"  frames in manifest : {total}")
    print(f"  valid labels       : {present}  ({cov*100:.1f}% coverage)")
    print(f"     of which empty   : {empty}  (no kept GT boxes -> zero ZOI supervision)")
    print(f"  MISSING label file : {len(missing)}")
    print(f"  MALFORMED label    : {len(malformed)}")
    for tag, items in (("MISSING", missing), ("MALFORMED", malformed)):
        for line in items[:show]:
            print(f"    [{tag}] {line}")
        if len(items) > show:
            print(f"    ... and {len(items) - show} more {tag}")
    return total, present, len(missing), len(malformed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="_manifest/*.txt file(s) from sample_zoi_dataset.py")
    ap.add_argument("--labels_root", default=None,
                    help="root the manifest route paths are relative to (where <route>/zoi_labels/ "
                         "lives). Default = the manifest's parent's parent (<out_root>). Pass the "
                         "--output_dir you gave the generator if labels were mirrored elsewhere.")
    ap.add_argument("--out_subdir", default="zoi_labels")
    ap.add_argument("--min_coverage", type=float, default=1.0,
                    help="fail (exit 1) if any manifest's coverage is below this fraction")
    ap.add_argument("--show", type=int, default=10, help="how many missing/malformed to list each")
    args = ap.parse_args()

    g_total = g_present = g_missing = g_malformed = 0
    worst_cov = 1.0
    for m in args.manifest:
        labels_root = args.labels_root or os.path.dirname(os.path.dirname(os.path.abspath(m)))
        t, p, miss, mal = check_manifest(m, labels_root, args.out_subdir, args.show)
        g_total += t; g_present += p; g_missing += miss; g_malformed += mal
        worst_cov = min(worst_cov, (p / t if t else 0.0))

    print("\n================ OVERALL ================")
    print(f"frames           : {g_total}")
    print(f"valid labels     : {g_present}  ({(g_present/g_total*100 if g_total else 0):.1f}%)")
    print(f"missing          : {g_missing}")
    print(f"malformed        : {g_malformed}")
    print("=========================================")

    fail = (worst_cov < args.min_coverage) or (g_malformed > 0)
    if fail:
        print(f"GUARD FAILED: worst coverage {worst_cov*100:.1f}% "
              f"(min {args.min_coverage*100:.0f}%), malformed {g_malformed}.")
        sys.exit(1)
    print("GUARD PASSED: every selected frame has a valid label.")


if __name__ == "__main__":
    main()
