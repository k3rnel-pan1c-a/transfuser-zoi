#!/usr/bin/env python3
"""
sample_zoi_dataset.py
=====================
Sample a balanced, eval-ready subset of the Kaggle carla_garage dump and lay it
out so that (a) carla_garage's team_code/data.py can load it as-is for training +
eval, and (b) the ZOI label generators have an exact frame manifest to label.

Why this script exists (two format mismatches found in the Kaggle dump):
  1. carla_garage/data.py expects gzipped sidecars: boxes/XXXX.json.gz,
     measurements/XXXX.json.gz, results.json.gz.  The Kaggle dump stores them
     PLAIN (boxes/XXXX.json, measurements/XXXX.json, results.json).
     -> we gzip-convert those three; everything else (rgb*.jpg, lidar*.laz,
        semantics/bev/depth *.png) already matches and is symlinked (free).
  2. carla_garage walks `os.walk(sub_root)[1]` i.e. route dirs must sit DIRECTLY
     under each scenario root.  The Kaggle dump double-nests (<Scn>/<Scn>/<route>).
     -> we flatten to  <out_root>/<Scenario>/<route>.

It also replicates data.py's route filter (perfect expert score, except pure
min-speed infractions) so we never select a route data.py will silently drop, and
it holds out a TOWN for eval via carla_garage's own val_towns mechanism.

Outputs under --out_root:
  <Scenario>/<route>/...            training tree (symlinks + gzipped sidecars)
  _manifest/train_labels.txt        "<scenario>/<route> <stem>" per frame to ZOI-label (train)
  _manifest/eval_labels.txt         same, for the held-out eval town
  _manifest/summary.json            counts, the chosen config, and the exact
                                    carla_garage flags to use.

After running, generate ZOI labels and train:
  # labels (point the generator at the built tree; --output_dir keeps labels separate)
  python generate_zoi_labels.py --root_dir <out_root> --output_dir <out_root>/zoi_labels \
      --skip_first 10 --stride 5            # (add these args per ZOI_CONTEXT TODO)
  # train: --root_dir = the scenario dirs;  set config.val_towns=[13],
  #        config.train_sampling_rate=<stride>, config.skip_first stays 10.
See _manifest/summary.json for the concrete invocation.
"""
import argparse
import glob
import gzip
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Defaults that mirror carla_garage/team_code/config.py
#   skip_first = int(2.5 * carla_fps(20)) // data_save_freq(5) = 10
# ---------------------------------------------------------------------------
SKIP_FIRST_DEFAULT = 10

# Per ZOI_CONTEXT.md section 6: ZOI-relevant scenarios + ParkedObstacle as the
# NEGATIVE / irrelevant-object case (critical for supervised-vs-unsupervised).
# Weights are relative shares of the train frame budget.
DEFAULT_WEIGHTS = {
    "PedestrianCrossing": 1.0,
    "DynamicObjectCrossing": 1.0,
    "HighwayCutIn": 1.0,
    "SignalizedJunctionLeftTurn": 1.0,
    "ParkedObstacleTwoWays": 1.0,   # negatives
    "noScenarios": 0.5,             # easy/background variety
}

# Dirs that already match carla_garage's expected extensions -> symlink as-is.
SYMLINK_DIRS = [
    "rgb", "rgb_augmented", "lidar",
    "semantics", "semantics_augmented",
    "bev_semantics", "bev_semantics_augmented",
    "depth", "depth_augmented",
]


def find_route_dirs(scenario_root):
    """Return route dirs anywhere under a scenario root (handles the double-nest)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(scenario_root):
        base = os.path.basename(dirpath)
        if re.search(r"_Rep\d+", base) and re.search(r"Town\d+", base):
            if os.path.isdir(os.path.join(dirpath, "boxes")):
                out.append(dirpath)
            dirnames[:] = []  # don't descend into a route
    return sorted(out)


def town_of(route_name):
    m = re.search(r"Town(\d+)", route_name)
    return int(m.group(1)) if m else -1


# data.py drops a route ONLY if its status is exactly one of these strings (it does
# NOT require status=="Completed" — "Perfect" and "Completed" are both kept).
FAILED_STATUSES = {
    "Failed - Agent couldn't be set up",
    "Failed",
    "Failed - Simulation crashed",
    "Failed - Agent crashed",
}


def is_trainable(route_dir):
    """Mirror data.py's accept condition EXACTLY: keep the route unless its expert
    run scored <100 with a non-min-speed infraction, or its status is an explicit
    'Failed - ...' that data.py rejects. Returns (ok, reason)."""
    res = os.path.join(route_dir, "results.json")
    if not os.path.isfile(res):
        return False, "no results.json"
    if os.path.basename(route_dir).startswith("FAILED_"):
        return False, "FAILED_ prefix"
    try:
        with open(res, "r", encoding="utf-8") as f:
            r = json.load(f)
    except Exception as e:
        return False, f"results parse: {e}"
    status = r.get("status", "")
    scores = r.get("scores", {})
    composed = scores.get("score_composed", 0.0)
    num_inf = r.get("num_infractions", 0)
    min_speed = len(r.get("infractions", {}).get("min_speed_infractions", []))
    # condition1: drop if score<100 AND not(all infractions are min-speed)
    if composed < 100.0 and not (num_inf == min_speed):
        return False, f"score={composed:.1f} non-minspeed infractions"
    if status in FAILED_STATUSES:
        return False, f"status={status}"
    if not os.path.isdir(os.path.join(route_dir, "lidar")):
        return False, "no lidar"
    if not os.path.isdir(os.path.join(route_dir, "rgb")):
        return False, "no rgb"
    return True, "ok"


def frame_stems(route_dir, skip_first):
    """Sorted 4-digit stems present in boxes/, at or after skip_first."""
    stems = []
    for p in glob.glob(os.path.join(route_dir, "boxes", "*.json")):
        s = os.path.splitext(os.path.basename(p))[0]
        if s.isdigit() and int(s) >= skip_first:
            stems.append(s)
    return sorted(stems)


# ---- tree building (runs in worker processes) -----------------------------
def _gzip_file(src, dst):
    if os.path.exists(dst):
        return
    with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)


def build_route(args):
    """Build one carla_garage-compatible route dir. Returns (dst, n_gz)."""
    src_route, dst_route = args
    os.makedirs(dst_route, exist_ok=True)

    # 1) symlink the dirs whose extensions already match
    for d in SYMLINK_DIRS:
        s = os.path.join(src_route, d)
        if os.path.isdir(s):
            link = os.path.join(dst_route, d)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(s), link)

    # 2) gzip-convert the plain JSON sidecars data.py expects gzipped
    n = 0
    for sub in ("boxes", "measurements"):
        ssub = os.path.join(src_route, sub)
        if not os.path.isdir(ssub):
            continue
        dsub = os.path.join(dst_route, sub)
        os.makedirs(dsub, exist_ok=True)
        for jp in glob.glob(os.path.join(ssub, "*.json")):
            stem = os.path.basename(jp)
            _gzip_file(jp, os.path.join(dsub, stem + ".gz"))
            n += 1

    # 3) results.json -> results.json.gz (route accept-check needs it)
    rj = os.path.join(src_route, "results.json")
    if os.path.isfile(rj):
        _gzip_file(rj, os.path.join(dst_route, "results.json.gz"))
    return dst_route, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_root",
                    default="/kaggle/input/datasets/k3rnelpan1ca/carla-garage",
                    help="root containing the scenario folders")
    ap.add_argument("--out_root", default="/kaggle/working/zoi_data",
                    help="where to build the training/eval tree")
    ap.add_argument("--eval_town", type=int, default=13,
                    help="town held out entirely for eval (-> config.val_towns)")
    ap.add_argument("--target_train_frames", type=int, default=5000,
                    help="ZOI-labeled training frames to aim for (after stride)")
    ap.add_argument("--target_eval_frames", type=int, default=1000,
                    help="ZOI-labeled eval frames (alignment metric; L2 needs none)")
    ap.add_argument("--stride", type=int, default=5,
                    help="temporal subsample: label/train every Nth frame "
                         "(=> set config.train_sampling_rate to this)")
    ap.add_argument("--skip_first", type=int, default=SKIP_FIRST_DEFAULT,
                    help="drop the first N saved frames per route (OOD warm-up)")
    ap.add_argument("--max_routes_per_scenario", type=int, default=0,
                    help="0 = unlimited; cap routes picked per scenario")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="select + write manifests, but don't build the tree")
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    scenarios = [d for d in sorted(os.listdir(args.input_root))
                 if os.path.isdir(os.path.join(args.input_root, d))
                 and d in DEFAULT_WEIGHTS]
    if not scenarios:
        sys.exit(f"No known scenarios under {args.input_root}")

    # ----- 1. enumerate + filter routes, split into train / eval by town -----
    print("Scanning routes (filtering to trainable, splitting by town)...")
    train_routes = defaultdict(list)   # scenario -> [route_dir]
    eval_routes = defaultdict(list)
    stats = defaultdict(lambda: defaultdict(int))
    for scn in scenarios:
        for rd in find_route_dirs(os.path.join(args.input_root, scn)):
            ok, reason = is_trainable(rd)
            stats[scn]["total"] += 1
            if not ok:
                stats[scn]["dropped"] += 1
                continue
            if town_of(os.path.basename(rd)) == args.eval_town:
                eval_routes[scn].append(rd)
            else:
                train_routes[scn].append(rd)
        for d in (train_routes, eval_routes):
            rng.shuffle(d[scn])
        print(f"  {scn:28s} trainable={stats[scn]['total']-stats[scn]['dropped']:4d}"
              f"  (dropped {stats[scn]['dropped']})  "
              f"train={len(train_routes[scn])} eval={len(eval_routes[scn])}")

    # ----- 2. pick routes per scenario to hit the frame budget ---------------
    def select(routes_by_scn, total_budget, weights):
        wsum = sum(weights[s] for s in routes_by_scn if routes_by_scn[s])
        picked = defaultdict(list)   # scenario -> [(route_dir, [stems])]
        frames = 0
        for scn in routes_by_scn:
            if not routes_by_scn[scn]:
                continue
            quota = int(round(total_budget * weights[scn] / wsum)) if wsum else 0
            got = 0
            for rd in routes_by_scn[scn]:
                if args.max_routes_per_scenario and \
                   len(picked[scn]) >= args.max_routes_per_scenario:
                    break
                stems = frame_stems(rd, args.skip_first)[:: args.stride]
                if not stems:
                    continue
                if got >= quota:
                    break
                take = stems[: max(0, quota - got)]
                if not take:
                    continue
                picked[scn].append((rd, take))
                got += len(take)
                frames += len(take)
            stats[scn]["picked_routes"] = len(picked[scn])
            stats[scn]["picked_frames"] = got
        return picked, frames

    print("\nSelecting train routes...")
    train_pick, n_train = select(train_routes, args.target_train_frames, DEFAULT_WEIGHTS)
    print(f"  -> {n_train} train frames across "
          f"{sum(len(v) for v in train_pick.values())} routes")
    print("Selecting eval routes...")
    eval_pick, n_eval = select(eval_routes, args.target_eval_frames, DEFAULT_WEIGHTS)
    print(f"  -> {n_eval} eval frames across "
          f"{sum(len(v) for v in eval_pick.values())} routes")

    # ----- 3. write manifests -------------------------------------------------
    man_dir = os.path.join(args.out_root, "_manifest")
    os.makedirs(man_dir, exist_ok=True)

    def write_manifest(path, pick):
        with open(path, "w") as f:
            for scn in sorted(pick):
                for rd, stems in pick[scn]:
                    route = os.path.basename(rd)
                    for s in stems:
                        f.write(f"{scn}/{route} {s}\n")

    write_manifest(os.path.join(man_dir, "train_labels.txt"), train_pick)
    write_manifest(os.path.join(man_dir, "eval_labels.txt"), eval_pick)

    # ----- 4. build the tree (symlinks + gzip) -------------------------------
    build_jobs = []
    for pick in (train_pick, eval_pick):
        for scn in pick:
            for rd, _ in pick[scn]:
                dst = os.path.join(args.out_root, scn, os.path.basename(rd))
                build_jobs.append((rd, dst))

    if args.dry_run:
        print("\n[dry_run] skipping tree build "
              f"({len(build_jobs)} routes would be built).")
    else:
        print(f"\nBuilding {len(build_jobs)} route dirs "
              f"with {args.workers} workers (symlink + gzip)...")
        total_gz = 0
        with Pool(args.workers) as pool:
            for i, (dst, n) in enumerate(pool.imap_unordered(build_route, build_jobs), 1):
                total_gz += n
                if i % 10 == 0 or i == len(build_jobs):
                    print(f"  [{i}/{len(build_jobs)}] gz so far: {total_gz}")
        print(f"  done. gzipped {total_gz} sidecar files.")

    # ----- 5. summary + ready-to-paste carla_garage flags --------------------
    scenario_roots = [os.path.join(args.out_root, scn)
                      for scn in scenarios
                      if (train_pick.get(scn) or eval_pick.get(scn))]
    summary = {
        "input_root": args.input_root,
        "out_root": args.out_root,
        "eval_town": args.eval_town,
        "stride": args.stride,
        "skip_first": args.skip_first,
        "weights": DEFAULT_WEIGHTS,
        "train_frames": n_train,
        "eval_frames": n_eval,
        "train_routes": sum(len(v) for v in train_pick.values()),
        "eval_routes": sum(len(v) for v in eval_pick.values()),
        "per_scenario": {s: dict(stats[s]) for s in stats},
        "carla_garage": {
            "root_dir": scenario_roots,
            "config.val_towns": [args.eval_town],
            "config.train_sampling_rate": args.stride,
            "config.skip_first": args.skip_first,
            "note": "labels live under <out_root>/zoi_labels/<Scenario>/<route>/zoi_labels/<stem>.npy",
        },
        "label_cmd": (
            f"python generate_zoi_labels.py --root_dir {args.out_root} "
            f"--output_dir {os.path.join(args.out_root, 'zoi_labels')} "
            f"--skip_first {args.skip_first} --stride {args.stride}"
        ),
    }
    with open(os.path.join(man_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"TRAIN: {n_train} frames / {summary['train_routes']} routes")
    print(f"EVAL : {n_eval} frames / {summary['eval_routes']} routes "
          f"(Town{args.eval_town} held out)")
    print(f"Tree : {args.out_root}")
    print(f"Manifests + summary: {man_dir}")
    print("carla_garage root_dir:")
    for r in scenario_roots:
        print(f"   {r}")
    print(f"  set config.val_towns=[{args.eval_town}], "
          f"train_sampling_rate={args.stride}, skip_first={args.skip_first}")
    print("=" * 70)


if __name__ == "__main__":
    main()
