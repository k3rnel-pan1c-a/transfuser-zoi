#!/usr/bin/env python3
"""
eval_bakeoff.py — score and compare teacher_bakeoff.py runs; writes a markdown report.

Reads every <out_root>/<teacher>/rich_records.jsonl produced by teacher_bakeoff.py
and computes, per teacher (VLM-scored car/walker objects only — rule-scored signals
carry no teacher judgment and are excluded, ZOI_CONTEXT.md sec 21.2):

  discriminativeness   spread std, used range, score histogram
  geometry             depth-rho (Spearman of imp vs closeness 1/(1+radial)),
                       top1-closest rate, and the NON-DISTANCE RESIDUAL: 1 - R^2 of
                       imp regressed on [closeness, |lateral|] — the share of the
                       teacher's signal that is NOT already free from geometry
                       (bigger = more to distill; sec 21.2)
  reliability          omission rate after retry, parse-fail frames, dropped objects
  rich schema          tag completeness (path/act/urg present+valid), rationale
                       coverage/length, imp<->tag consistency (mean imp per path tag;
                       contradiction rate: imp>=0.8 with path=no or act=ignore)
  cost                 mean/median VLM seconds per frame

and pairwise between teachers (on the shared objects both scored):
  Spearman/Pearson agreement, mean |delta|, and the top disagreements with each
  teacher's one-sentence rationale so you can eyeball WHO is right.

Usage:
  python eval_bakeoff.py --out_root ./bakeoff_out [--report bakeoff_report.md]

numpy-only (no scipy/pandas) so it runs anywhere the labeling ran.
"""

import argparse
import glob
import json
import os
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------- stats helpers
def _ranks(v):
    v = np.asarray(v, dtype=float)
    order = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(len(v), dtype=float)
    for val in np.unique(v):          # average ties (scores are discrete -> ties matter)
        m = v == val
        r[m] = r[m].mean()
    return r


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(_ranks(a), _ranks(b))


def geometry_residual(imp, radial, lat):
    """1 - R^2 of imp ~ [closeness, |lat|, 1]. The share of the teacher's importance
    signal not explained by free geometry (higher = more worth distilling)."""
    imp = np.asarray(imp, float)
    if len(imp) < 8 or imp.std() < 1e-9:
        return float("nan")
    X = np.stack([1.0 / (1.0 + np.asarray(radial, float)),
                  np.abs(np.asarray(lat, float)),
                  np.ones(len(imp))], axis=1)
    coef, *_ = np.linalg.lstsq(X, imp, rcond=None)
    ss_res = float(((imp - X @ coef) ** 2).sum())
    ss_tot = float(((imp - imp.mean()) ** 2).sum())
    return ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------- data loading
def load_teacher_records(out_root):
    """teacher -> {"frames": [records], "objects": {(frame, oid): obj}} for VLM objects."""
    teachers = {}
    for rec_path in sorted(glob.glob(os.path.join(out_root, "*", "rich_records.jsonl"))):
        name = os.path.basename(os.path.dirname(rec_path))
        frames, objects = [], {}
        with open(rec_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                frames.append(rec)
                for o in rec["objects"]:
                    if o["src"] == "vlm" and o["oid"] is not None:
                        objects[(rec["frame"], o["oid"])] = o
        if frames:
            teachers[name] = {"frames": frames, "objects": objects}
    return teachers


def summarize(name, data):
    frames, objects = data["frames"], data["objects"]
    objs = list(objects.values())
    imp = np.array([o["imp"] for o in objs], float)
    radial = np.array([o["radial"] for o in objs], float)
    lat = np.array([o["y"] for o in objs], float)
    closeness = 1.0 / (1.0 + radial)

    n_marked = sum(r["n_marked"] for r in frames)
    n_missing = sum(len(r["missing_after_retry"]) for r in frames)

    # top1: on frames with >=2 scored objects, does the max-imp object == closest?
    per_frame = {}
    for (fkey, _), o in objects.items():
        per_frame.setdefault(fkey, []).append(o)
    multi = [v for v in per_frame.values() if len(v) >= 2]
    top1 = (np.mean([int(max(v, key=lambda o: o["imp"]) is min(v, key=lambda o: o["radial"]))
                     for v in multi]) if multi else float("nan"))

    hist = {f"{k}": int((np.round(imp * 5) == k).sum()) for k in range(6)}

    tag_complete = (np.mean([all(o[k] is not None for k in ("path", "act", "urg"))
                             for o in objs]) if objs else float("nan"))
    whys = [o["why"] for o in objs if o["why"]]
    imp_by_path = {v: round(float(np.mean([o["imp"] for o in objs if o["path"] == v])), 2)
                   for v in ("yes", "maybe", "no")
                   if any(o["path"] == v for o in objs)}
    contra = (np.mean([int(o["imp"] >= 0.8 and (o["path"] == "no" or o["act"] == "ignore"))
                       for o in objs]) if objs else float("nan"))

    gen = [r["gen_s"] for r in frames if r["gen_s"] > 0]
    return {
        "teacher": name,
        "frames": len(frames),
        "objects_scored": len(objs),
        "marked": n_marked,
        "omission_rate": round(n_missing / max(n_marked, 1), 4),
        "parse_fail_frames": sum(int(r["parse_fail"]) for r in frames),
        "spread_std": round(float(imp.std() * 5), 2) if len(imp) else float("nan"),
        "range": (f"{imp.min() * 5:.0f}-{imp.max() * 5:.0f}" if len(imp) else "-"),
        "hist_0to5": hist if len(imp) else {},
        "depth_rho": round(spearman(imp, closeness), 2) if len(imp) else float("nan"),
        "top1_closest": round(float(top1), 2),
        "nondist_residual": round(geometry_residual(imp, radial, lat), 2),
        "tag_complete": round(float(tag_complete), 2),
        "rationale_cov": round(len(whys) / max(len(objs), 1), 2),
        "rationale_words": round(float(np.mean([len(w.split()) for w in whys])), 1) if whys else 0.0,
        "imp_by_path": imp_by_path,
        "contradiction_rate": round(float(contra), 3),
        "gen_s_mean": round(float(np.mean(gen)), 1) if gen else 0.0,
        "gen_s_median": round(float(np.median(gen)), 1) if gen else 0.0,
    }


def pairwise(name_a, a, name_b, b, top_k=10):
    shared = sorted(set(a["objects"]) & set(b["objects"]))
    if len(shared) < 3:
        return None
    ia = np.array([a["objects"][k]["imp"] for k in shared])
    ib = np.array([b["objects"][k]["imp"] for k in shared])
    deltas = np.abs(ia - ib)
    order = np.argsort(-deltas)
    rows = []
    for idx in order[:top_k]:
        if deltas[idx] < 0.3:
            break
        k = shared[idx]
        oa, ob = a["objects"][k], b["objects"][k]
        rows.append({"frame": k[0], "oid": k[1], "class": oa["class"],
                     "radial": round(oa["radial"], 1),
                     name_a: round(oa["imp"] * 5, 1), name_b: round(ob["imp"] * 5, 1),
                     f"why_{name_a}": oa["why"], f"why_{name_b}": ob["why"]})
    return {"pair": f"{name_a} vs {name_b}", "shared_objects": len(shared),
            "spearman": round(spearman(ia, ib), 2), "pearson": round(pearson(ia, ib), 2),
            "mean_abs_delta_0to5": round(float(deltas.mean() * 5), 2),
            "top_disagreements": rows}


# ---------------------------------------------------------------- report
def fmt_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="./bakeoff_out")
    ap.add_argument("--report", default=None, help="output .md (default <out_root>/bakeoff_report.md)")
    args = ap.parse_args()
    report_path = args.report or os.path.join(args.out_root, "bakeoff_report.md")

    teachers = load_teacher_records(args.out_root)
    if not teachers:
        raise SystemExit(f"no <teacher>/rich_records.jsonl found under {args.out_root}")

    summaries = [summarize(n, d) for n, d in sorted(teachers.items())]
    pairs = [p for (na, da), (nb, db) in
             ((x, y) for x, y in combinations(sorted(teachers.items()), 2))
             if (p := pairwise(na, da, nb, db)) is not None]

    # Ranking heuristic: reliability gate first, then supervision-signal quality.
    def rank_key(s):
        gate = (s["omission_rate"] > 0.1) + (s["parse_fail_frames"] > s["frames"] * 0.05)
        resid = s["nondist_residual"] if np.isfinite(s["nondist_residual"]) else 0.0
        rho = s["depth_rho"] if np.isfinite(s["depth_rho"]) else 0.0
        return (gate, -(resid + 0.5 * rho + 0.1 * s["spread_std"]))
    ranked = sorted(summaries, key=rank_key)

    lines = ["# ZOI teacher bake-off report", "",
             f"Out root: `{args.out_root}` — teachers: {', '.join(sorted(teachers))}", "",
             "## Headline comparison", "",
             fmt_table(summaries, ["teacher", "frames", "objects_scored", "omission_rate",
                                   "parse_fail_frames", "spread_std", "range", "depth_rho",
                                   "top1_closest", "nondist_residual", "gen_s_mean"]), "",
             "- `nondist_residual` = share of the importance signal NOT explained by "
             "closeness+|lateral| (higher = more non-geometric knowledge to distill).",
             "- `depth_rho` = Spearman(imp, closeness); sanity floor, not the target.",
             "- gate: omission_rate > 0.10 or parse fails > 5% of frames disqualifies a "
             "teacher regardless of signal quality.", "",
             "## Rich schema quality", "",
             fmt_table(summaries, ["teacher", "tag_complete", "rationale_cov",
                                   "rationale_words", "imp_by_path", "contradiction_rate"]), "",
             "- `imp_by_path` should be monotone yes > maybe > no; `contradiction_rate` "
             "counts imp>=4/5 objects tagged path=no or act=ignore.", "",
             "## Score histograms (0-5)", ""]
    for s in summaries:
        lines.append(f"- **{s['teacher']}**: {s['hist_0to5']}")
    lines += ["", "## Pairwise agreement", ""]
    for p in pairs:
        lines += [f"### {p['pair']}  (n={p['shared_objects']})",
                  f"Spearman {p['spearman']}, Pearson {p['pearson']}, "
                  f"mean |delta| {p['mean_abs_delta_0to5']} (0-5 scale)", ""]
        if p["top_disagreements"]:
            cols = list(p["top_disagreements"][0].keys())
            lines += [fmt_table(p["top_disagreements"], cols), ""]
    lines += ["## Ranking (reliability gate, then residual + 0.5*rho + 0.1*spread)", ""]
    for i, s in enumerate(ranked, 1):
        gate = " (FAILS reliability gate)" if rank_key(s)[0] else ""
        lines.append(f"{i}. **{s['teacher']}**{gate} — residual {s['nondist_residual']}, "
                     f"rho {s['depth_rho']}, spread {s['spread_std']}, "
                     f"omissions {s['omission_rate']}")
    lines += ["", "Small-n caveat: with a few hundred objects these are directional. "
              "The verdict metric for the PROJECT is the attention-probe residual "
              "correlation on each teacher's labels (attention_probe.py --labels_root "
              "<out_root>/<teacher>/labels), run by us — not part of this kit."]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out_root, "bakeoff_metrics.json"), "w") as f:
        json.dump({"summaries": summaries, "pairwise": pairs}, f, indent=2)

    print("\n".join(lines))
    print(f"\nwrote {report_path} and bakeoff_metrics.json")


if __name__ == "__main__":
    main()
