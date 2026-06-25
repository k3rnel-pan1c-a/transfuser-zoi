"""
Offline ZOI label generator (Set-of-Marks approach).

Instead of asking the VLM to DETECT planning-relevant objects (off-task, noisy
grounding + fragile matching), we:
  1. Load the CARLA GT boxes for a frame (boxes/XXXX.json).
  2. Filter them to the SAME set carla_garage's parse_bounding_boxes keeps,
     so our targets align with what the detector/planner actually sees.
  3. Project each GT box center into the front camera image (repo convention).
  4. Draw a numbered marker per visible object (Set-of-Marks).
  5. Ask Qwen2.5-VL to return an importance score (0-5) + planning_role per ID.
  6. Save a per-frame target array in BEV ego meters:
        [x, y, importance(0..1), class_id]   one row per filtered GT box.
     (importance 0 for objects the VLM didn't score / not camera-visible.)

The saved .npy is what ZoiModule is supervised against (Hungarian matching on
xy + BCE on importance). Locations come from GT, so they are exact.

NOTE ON PROJECTION: based on create_projection_grid()'s convention from transfuser_utils.py
(CARLA x-front/y-right/z-up -> pinhole, camera at camera_pos, zero rotation), but with the
height (z) term NEGATED relative to that function -- verified empirically against real
--debug_dir overlays (un-negated, markers float well above the actual objects). ALWAYS
eyeball a few overlays on a new dataset/camera config before trusting a full run.
"""

import argparse
import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import ujson
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# qwen_vl_utils is imported lazily inside process_frame so that other tools can
# reuse PROMPT_TEMPLATE / extract_json without having that package installed.

# Projection + Set-of-Marks geometry now live in ONE shared, torch-free module
# (zoi_projection.py) so the preview/overlay tools and this generator can never
# drift apart. The z-sign there is verified empirically (verify_projection.py).
from zoi_projection import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CLASS_TO_ID,
    VLM_MARK_CLASSES, RULE_CLASSES, rule_importance,
    project_ego_to_image, keep_box, draw_marks,
)


PROMPT_TEMPLATE = """You are a planning-aware zone-of-interest (ZOI) annotator for an autonomous driving system.

The image is the EGO vehicle's front camera view. Numbered green markers have
already been placed on objects detected by the simulator's ground truth — you
are NOT detecting objects. Your only task is to judge, for EACH numbered
marker, how planning-relevant that object is: whether it may directly affect
the ego vehicle's near-future driving decisions such as braking, steering,
yielding, turning, lane changes, or obstacle avoidance over the next few
seconds.

SCORE HIGH (4-5) when the marked object:
- Is directly ahead in the ego lane or actively merging/cutting into it.
- Is a stopped or slow vehicle ahead that may require braking or a lane change.
- Is a pedestrian near a crosswalk, at a road edge, or visibly about to step onto the road.
- Is a cyclist on or crossing the ego vehicle's path.
- Is a traffic light or stop sign governing the ego lane or the upcoming intersection AND currently red/stop-relevant.
- Is an obstacle blocking or narrowing the ego vehicle's current drivable lane.

SCORE MEDIUM (2-3) when the marked object:
- Is near the ego path but not yet requiring an active response (should be monitored).
- Could plausibly interact with the ego vehicle if its motion or the ego's changes (e.g. a vehicle at a side street, a pedestrian on a sidewalk near the road).

SCORE LOW OR ZERO (0-1) when the marked object:
- Is parked far away or on the opposite side of the road and does not interact with the ego path.
- Is far ahead or to the side and clearly does not affect the ego vehicle's current trajectory.
- Is a pedestrian on a distant sidewalk, far from any road crossing.
- Is a traffic light or sign that clearly faces opposing traffic or controls a different lane/direction.
- Is occluded, barely visible, or has no plausible effect on ego in the next few seconds.

IMPORTANCE SCALE (0-5):
  5 = critical (immediate collision risk or hard constraint on the ego path)
  4 = high (requires an active response such as braking or yielding)
  3 = medium (should be monitored and may influence speed or steering)
  2 = low (minor influence on planning)
  1 = marginal (barely relevant)
  0 = irrelevant (no plausible effect on ego's near-future driving)

The numbered objects are:
{object_list}

Return ONLY valid JSON, exactly this format:
{{
  "scores": [
    {{"id": 0, "importance": 3, "planning_role": "dynamic_agent | traffic_rule | static_obstacle | other", "reason": "one sentence: spatial relationship to ego and what decision it affects"}}
  ]
}}
Score every id listed, even if importance is 0. No text outside the JSON. No markdown. No code fences.
"""


def extract_json(text):
    text = re.sub(r"^```json|^```|```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    raise ValueError("could not parse VLM JSON")


def load_boxes(boxes_path):
    """Read a boxes file, transparently handling .json and gzipped .json.gz
    (the built training tree gzips sidecars; the raw Kaggle dump does not)."""
    if boxes_path.endswith(".gz"):
        import gzip
        with gzip.open(boxes_path, "rt", encoding="utf-8") as f:
            return ujson.load(f)
    with open(boxes_path, "r", encoding="utf-8") as f:
        return ujson.load(f)


def process_frame(rgb_path, boxes_path, model, processor, debug_dir=None):
    boxes = load_boxes(boxes_path)

    kept = [b for b in boxes if keep_box(b)]               # filtered GT set
    n = len(kept)
    # target rows: [x, y, importance, class_id]; importance defaults to 0.
    targets = np.zeros((n, 4), dtype=np.float32)
    for i, b in enumerate(kept):
        x, y, _ = b["position"]
        targets[i, 0], targets[i, 1] = x, y
        targets[i, 3] = CLASS_TO_ID[b["class"]]
        # Signals (traffic_light / stop_sign): GT position is a road-level stop-line
        # trigger, NOT the visible fixture, so a Set-of-Marks dot would land on the
        # road or another vehicle and mislead the VLM. keep_box already restricts
        # them to red+affects_ego / affects_ego, so they ARE planning-relevant ->
        # score by rule and DON'T show them to the VLM.
        if b["class"] in RULE_CLASSES:
            targets[i, 2] = rule_importance(b)

    # Only cars/walkers get a VLM mark (their GT position lands on the object).
    marks, id_to_row, object_lines = [], {}, []
    for i, b in enumerate(kept):
        if b["class"] not in VLM_MARK_CLASSES:
            continue
        proj = project_ego_to_image(b["position"])
        if proj is None:
            continue
        u, v, _ = proj
        if not (0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT):
            continue
        oid = len(marks)
        marks.append((oid, u, v))
        id_to_row[oid] = i
        object_lines.append(f'  id {oid}: a {b["class"]}')

    img = cv2.imread(str(rgb_path))
    if not marks:                                          # nothing visible -> all-zero targets
        return targets

    marked = draw_marks(img, marks)
    pil = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    prompt = PROMPT_TEMPLATE.format(object_list="\n".join(object_lines))

    messages = [{"role": "user", "content": [{"type": "image", "image": pil},
                                             {"type": "text", "text": prompt}]}]
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512)
    gen = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]

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

    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="dataset root (contains route folders)")
    ap.add_argument("--out_subdir", default="zoi_labels", help="subfolder name for the written .npy files")
    ap.add_argument("--output_dir", default=None,
                    help="if set, mirror each route's path (relative to --root_dir) under this writable "
                         "directory instead of writing inside the route folder itself. Use this when "
                         "--root_dir is read-only (e.g. a mounted Kaggle input dataset).")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--debug_dir", default=None, help="if set, save a few marked overlays here")
    ap.add_argument("--debug_n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="cap frames (0 = all) for a quick test")
    ap.add_argument("--skip_first", type=int, default=10,
                    help="skip the first N saved frames per route (OOD warm-up; "
                         "mirrors config.skip_first). Match what training uses.")
    ap.add_argument("--stride", type=int, default=1,
                    help="label every Nth frame per route (temporal subsample; "
                         "match config.train_sampling_rate used in training).")
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    # route folders that have both rgb/ and boxes/ (boxes may be .json or .json.gz)
    box_glob = (glob.glob(os.path.join(args.root_dir, "**", "boxes", "*.json"), recursive=True) +
                glob.glob(os.path.join(args.root_dir, "**", "boxes", "*.json.gz"), recursive=True))
    routes = sorted({str(Path(p).parents[1]) for p in box_glob})
    print(f"found {len(routes)} route folders")

    done = 0
    for route in routes:
        box_files = sorted(glob.glob(os.path.join(route, "boxes", "*.json")) +
                           glob.glob(os.path.join(route, "boxes", "*.json.gz")))
        if args.output_dir:
            route_rel = os.path.relpath(route, args.root_dir)
            out_dir = os.path.join(args.output_dir, route_rel, args.out_subdir)
        else:
            out_dir = os.path.join(route, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        for bf in box_files:
            stem = os.path.basename(bf).split(".")[0]   # 0010.json / 0010.json.gz -> 0010
            # mirror training's frame selection: drop warm-up frames + subsample
            if stem.isdigit():
                fi = int(stem)
                if fi < args.skip_first or fi % args.stride != 0:
                    continue
            rgb = os.path.join(route, "rgb", stem + ".jpg")
            if not os.path.isfile(rgb):
                continue
            dbg = args.debug_dir if (args.debug_dir and done < args.debug_n) else None
            tgt = process_frame(rgb, bf, model, processor, debug_dir=dbg)
            np.save(os.path.join(out_dir, stem + ".npy"), tgt)
            done += 1
            if done % 100 == 0:
                print(f"  {done} frames")
            if args.limit and done >= args.limit:
                print("hit --limit, stopping"); return
    print(f"done: {done} frames labeled")


if __name__ == "__main__":
    main()
