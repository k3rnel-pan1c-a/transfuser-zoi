"""
Offline ZOI label generator (Set-of-Marks) — Gemma 4 + exemplar prompt.

This is the PRODUCTION label generator chosen by the teacher/prompt comparison
(ZOI_CONTEXT.md sec 17/18/18a): Gemma 4 12B with the exemplar-anchored prompt gave
the best supervision signal (highest spread AND depth-rho 0.86; exemplar HURTS
InternVL, so the prompt is teacher-specific). Output is byte-identical in format to
the Qwen / Gemma 3 generators

    [x, y, importance(0..1), class_id]   one row per filtered GT box   (BEV ego meters)

so the .npy files are drop-in for ZoiModule supervision (data.py loads them, the
Hungarian zoi_loss matches on xy + BCE on importance).

Differences vs the older generators (and WHY):
  - Teacher: Gemma 4 12B (`unsloth/gemma-4-12b-it`, bf16, sharded via device_map="auto";
    needs transformers >= 5.10 for the gemma4_unified arch). Loaded with the generic
    AutoModelForImageTextToText (the Gemma-3 class does NOT load Gemma 4).
  - Prompt: EXEMPLAR_PROMPT from zoi_prompts.py (compact id+importance OUTPUT, so no
    JSON truncation; per-score-level driving anchors that calibrate Gemma 4).
  - Signal handling is the FIXED logic (shared with generate_zoi_labels.py): a
    traffic_light / stop_sign GT `position` is the road-level stop-line trigger, NOT the
    visible fixture, so marking it misleads the VLM. Those classes are scored BY RULE
    (rule_importance) and never shown to the model; only car/walker get a Set-of-Marks
    mark. keep_box already restricts kept signals to red+affects_ego / affects_ego.
  - Object lines include depth ("~12 m"), matching how the exemplar prompt was evaluated
    in run_vlm_multiframe.py (the comparison that selected it).

All delicate geometry (projection convention + z-sign, box filtering, mark drawing) is
imported from the single source of truth (zoi_projection.py) so it can never drift.

Frame selection mirrors training (ZOI_CONTEXT.md sec 7): --skip_first 10 drops the OOD
warm-up, --stride matches config.train_sampling_rate. Defaults are set for training parity.

The run is timed (per-frame VLM latency + per-route + total wall) so you can budget a
full-dataset labeling pass.
"""

import argparse
import glob
import os
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# Single source of truth for geometry + filtering + signal rules (verified z-sign).
from zoi_projection import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CLASS_TO_ID,
    VLM_MARK_CLASSES, RULE_CLASSES, rule_importance,
    project_ego_to_image, keep_box, draw_marks,
)
# Shared, model-agnostic helpers (gz-aware box loader + tolerant JSON parser).
from generate_zoi_labels import load_boxes, extract_json
# Teacher-specific prompts (exemplar is the chosen default; others for ablation rows).
from zoi_prompts import EXEMPLAR_PROMPT, EGOPATHCONFLICT_PROMPT
from generate_zoi_labels import PROMPT_TEMPLATE as FULL_PROMPT_TEMPLATE

DEFAULT_MODEL = "unsloth/gemma-4-12b-it"
PROMPTS = {
    "exemplar": EXEMPLAR_PROMPT,
    "egopathconflict": EGOPATHCONFLICT_PROMPT,
    "full": FULL_PROMPT_TEMPLATE,
}


def load_model(model_id):
    """Load Gemma 4 (bf16, sharded across visible GPUs). The generic
    AutoModelForImageTextToText handles the gemma4_unified arch; the Gemma-3
    class does not. Pre-quantized *-4bit checkpoints carry their own config."""
    kwargs = {"device_map": "auto"}
    if "4bit" not in model_id.lower():
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def gemma_score(pil, prompt, model, processor, max_new_tokens=256):
    """Run the Set-of-Marks scoring prompt through Gemma 4. Returns
    (raw_text, gen_seconds) — gen_seconds is pure model.generate() wall time."""
    messages = [{"role": "user", "content": [{"type": "image", "image": pil},
                                             {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    out = processor.decode(gen[0][input_len:], skip_special_tokens=True)
    return out, gen_s


def process_frame(rgb_path, boxes_path, model, processor, prompt_template,
                  debug_dir=None, max_new_tokens=256):
    """Mirror of generate_zoi_labels.process_frame (signal-fixed) with the Gemma 4 call.
    Returns (targets[M,4], gen_seconds); gen_seconds is 0.0 when no marks are visible."""
    boxes = load_boxes(boxes_path)

    kept = [b for b in boxes if keep_box(b)]               # filtered GT set (matches the planner)
    n = len(kept)
    targets = np.zeros((n, 4), dtype=np.float32)           # [x, y, importance, class_id]
    for i, b in enumerate(kept):
        x, y, _ = b["position"]
        targets[i, 0], targets[i, 1] = x, y
        targets[i, 3] = CLASS_TO_ID[b["class"]]
        # Signals: GT position is the road-level stop-line trigger, not the fixture, so a
        # mark would mislead the VLM -> score by rule and DON'T show them to the model.
        if b["class"] in RULE_CLASSES:
            targets[i, 2] = rule_importance(b)

    # Only car/walker get a Set-of-Marks mark (their GT position lands on the object).
    marks, id_to_row, object_lines = [], {}, []
    for i, b in enumerate(kept):
        if b["class"] not in VLM_MARK_CLASSES:
            continue
        proj = project_ego_to_image(b["position"])
        if proj is None:
            continue
        u, v, depth = proj
        if not (0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT):
            continue
        oid = len(marks)
        marks.append((oid, u, v))
        id_to_row[oid] = i
        # Include depth to match how the exemplar prompt was evaluated (run_vlm_multiframe).
        object_lines.append(f'  id {oid}: a {b["class"]} at ~{depth:.0f} m')

    img = cv2.imread(str(rgb_path))
    if not marks:                                          # nothing visible -> all-zero (+ rule) targets
        return targets, 0.0

    marked = draw_marks(img, marks)
    pil = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    prompt = prompt_template.format(object_list="\n".join(object_lines))

    out, gen_s = gemma_score(pil, prompt, model, processor, max_new_tokens=max_new_tokens)

    try:
        parsed = extract_json(out)
        for s in parsed.get("scores", []):
            oid = int(s["id"])
            if oid in id_to_row:
                imp = float(s.get("importance", 0))
                targets[id_to_row[oid], 2] = np.clip(imp / 5.0, 0.0, 1.0)   # -> [0,1]
    except Exception as e:
        print(f"  [warn] parse failed for {rgb_path}: {e}")

    if debug_dir is not None:
        for oid, u, v in marks:
            score = targets[id_to_row[oid], 2]
            cv2.putText(marked, f"{score:.1f}", (int(u) + 6, int(v) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, Path(rgb_path).stem + ".jpg"), marked)

    return targets, gen_s


def discover_routes(args):
    """Either a single explicit --route, or every route folder under --root_dir
    that has a boxes/ directory (boxes may be .json or .json.gz)."""
    if args.route:
        return [os.path.abspath(args.route)]
    box_glob = (glob.glob(os.path.join(args.root_dir, "**", "boxes", "*.json"), recursive=True) +
                glob.glob(os.path.join(args.root_dir, "**", "boxes", "*.json.gz"), recursive=True))
    return sorted({str(Path(p).parents[1]) for p in box_glob})


def resolve_boxes(route_dir, stem):
    """Return the existing boxes path for a frame (.json.gz in the built tree,
    plain .json in the raw dump), or None."""
    for ext in (".json.gz", ".json"):
        p = os.path.join(route_dir, "boxes", stem + ext)
        if os.path.isfile(p):
            return p
    return None


def route_stems_from_walk(route_dir, skip_first, stride):
    """Frames in a route selected by the SAME rule training uses (skip_first + stride)."""
    box_files = sorted(glob.glob(os.path.join(route_dir, "boxes", "*.json")) +
                       glob.glob(os.path.join(route_dir, "boxes", "*.json.gz")))
    stems = []
    for bf in box_files:
        stem = os.path.basename(bf).split(".")[0]
        if stem.isdigit():
            fi = int(stem)
            if fi < skip_first or fi % stride != 0:
                continue
        stems.append(stem)
    return stems


def build_worklist(args):
    """Ordered [(route_dir, [stems])]. When --manifest is given, the frame list comes
    STRAIGHT from the sampler's manifest (single source of truth -> no skip_first/stride
    drift between sampling, labeling, and training). Otherwise frames are re-derived by
    walking each route with --skip_first/--stride (must be set to match training)."""
    if args.manifest:
        from collections import OrderedDict
        routes = OrderedDict()
        for mpath in args.manifest:
            with open(mpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    route_rel, stem = line.rsplit(" ", 1)   # "<scn>/<route> <stem>"
                    rd = os.path.join(args.root_dir, route_rel)
                    routes.setdefault(rd, []).append(stem)
        return list(routes.items())
    return [(rd, route_stems_from_walk(rd, args.skip_first, args.stride))
            for rd in discover_routes(args)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", help="dataset root (contains route folders)")
    ap.add_argument("--route", help="process exactly ONE route folder (rgb/ + boxes/); "
                                     "use for a clean single-route timing run")
    ap.add_argument("--out_subdir", default="zoi_labels", help="subfolder for the written .npy files")
    ap.add_argument("--output_dir", default=None,
                    help="mirror each route's path (relative to --root_dir) under this writable "
                         "dir instead of writing inside the route folder (use for read-only inputs)")
    ap.add_argument("--manifest", nargs="+", default=None,
                    help="one or more _manifest/*.txt from sample_zoi_dataset.py. Labels EXACTLY "
                         "those frames (single source of truth; ignores --skip_first/--stride/--route). "
                         "Needs --root_dir = the built tree the manifest paths are relative to.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="exemplar",
                    help="exemplar = chosen default (sec 18a); egopathconflict/full = ablation rows")
    ap.add_argument("--debug_dir", default=None, help="if set, save the first N marked overlays here")
    ap.add_argument("--debug_n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="cap total frames (0 = all) for a quick test")
    ap.add_argument("--skip_first", type=int, default=10,
                    help="skip frames with index < N per route (OOD warm-up; mirror config.skip_first)")
    ap.add_argument("--stride", type=int, default=1,
                    help="label every Nth frame per route (match config.train_sampling_rate)")
    ap.add_argument("--max_new_tokens", type=int, default=256,
                    help="generation cap; compact JSON output needs few tokens")
    args = ap.parse_args()

    if not args.route and not args.root_dir and not args.manifest:
        ap.error("provide --manifest, --route (single route), or --root_dir (all routes)")
    if args.manifest and not args.root_dir:
        ap.error("--manifest needs --root_dir (the built tree the manifest paths are relative to)")
    if args.route and args.output_dir and not args.root_dir:
        ap.error("--output_dir needs --root_dir to compute the relative mirror path")

    prompt_template = PROMPTS[args.prompt]
    worklist = build_worklist(args)
    n_frames = sum(len(s) for _, s in worklist)
    src = "manifest" if args.manifest else "walk(skip_first/stride)"
    print(f"frame source: {src} -> {len(worklist)} routes, {n_frames} frames selected")

    print(f"loading {args.model} (prompt={args.prompt}) ...")
    t_load0 = time.perf_counter()
    model, processor = load_model(args.model)
    print(f"model loaded in {time.perf_counter() - t_load0:.1f}s")

    done = 0
    missing = 0
    all_gen_times = []
    run_t0 = time.perf_counter()

    for route, stems in worklist:
        if args.output_dir:
            base = args.root_dir or os.path.dirname(route)
            route_rel = os.path.relpath(route, base)
            out_dir = os.path.join(args.output_dir, route_rel, args.out_subdir)
        else:
            out_dir = os.path.join(route, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)

        route_t0 = time.perf_counter()
        route_frames = 0
        route_gen_times = []
        for stem in stems:
            bf = resolve_boxes(route, stem)
            rgb = os.path.join(route, "rgb", stem + ".jpg")
            if bf is None or not os.path.isfile(rgb):
                missing += 1                 # manifest frame whose files aren't in the tree
                continue
            dbg = args.debug_dir if (args.debug_dir and done < args.debug_n) else None
            tgt, gen_s = process_frame(rgb, bf, model, processor, prompt_template,
                                       debug_dir=dbg, max_new_tokens=args.max_new_tokens)
            np.save(os.path.join(out_dir, stem + ".npy"), tgt)
            done += 1
            route_frames += 1
            if gen_s > 0:
                route_gen_times.append(gen_s)
                all_gen_times.append(gen_s)
            if done % 50 == 0:
                rate = done / (time.perf_counter() - run_t0)
                print(f"  {done} frames ({rate:.2f} frames/s)")
            if args.limit and done >= args.limit:
                print("hit --limit, stopping early")
                break

        route_wall = time.perf_counter() - route_t0
        print(f"\n[route] {route}")
        print(f"  frames processed : {route_frames}")
        print(f"  frames VLM-scored: {len(route_gen_times)} (rest had no camera-visible marks)")
        print(f"  route wall time  : {route_wall:.1f}s ({route_wall/60:.2f} min)")
        if route_gen_times:
            print(f"  VLM gen/frame    : mean {statistics.mean(route_gen_times):.2f}s, "
                  f"median {statistics.median(route_gen_times):.2f}s, max {max(route_gen_times):.2f}s")
        if args.limit and done >= args.limit:
            break

    total_wall = time.perf_counter() - run_t0
    print("\n================ SUMMARY ================")
    print(f"teacher / prompt     : {args.model} / {args.prompt}")
    print(f"routes               : {len(worklist)}")
    print(f"frames labeled       : {done}")
    if missing:
        print(f"frames MISSING       : {missing}  (selected but rgb/boxes not found in the tree)")
    print(f"total wall time      : {total_wall:.1f}s ({total_wall/60:.2f} min)")
    if done:
        print(f"avg per-frame (wall) : {total_wall/done:.2f}s => {done/total_wall*60:.1f} frames/min")
    if all_gen_times:
        print(f"VLM gen/frame        : mean {statistics.mean(all_gen_times):.2f}s, "
              f"median {statistics.median(all_gen_times):.2f}s "
              f"(first frame {all_gen_times[0]:.2f}s incl. warmup)")
    print("=========================================")


if __name__ == "__main__":
    main()
