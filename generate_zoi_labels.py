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

NOTE ON PROJECTION: this replicates create_projection_grid()'s convention from
transfuser_utils.py (CARLA x-front/y-right/z-up -> pinhole, camera at camera_pos,
zero rotation). The vertical (z) sign in that convention is subtle; ALWAYS eyeball
a few --debug_dir overlays before trusting a full run.
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
from qwen_vl_utils import process_vision_info


# --- must mirror carla_garage/team_code/config.py -----------------------------
CAMERA_POS = [-1.5, 0.0, 2.0]      # x, y, z mounting position of the camera
CAMERA_FOV = 110
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 512
MIN_X, MAX_X = -32.0, 32.0
MIN_Y, MAX_Y = -32.0, 32.0
MIN_Z, MAX_Z = -3.0, 3.0           # height filter; check config.min_z/max_z
# Same classes / order as parse_bounding_boxes + visualize_dataset color map.
CLASS_TO_ID = {"car": 0, "walker": 1, "traffic_light": 2, "stop_sign": 3}
# LiDAR-hit thresholds from config.py (verify exact values in your config).
NUM_LIDAR_HITS_CAR = 1
NUM_LIDAR_HITS_WALKER = 1
# ------------------------------------------------------------------------------


def intrinsic_matrix(fov, height, width):
    f = width / (2.0 * np.tan(fov * np.pi / 360.0))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


K = intrinsic_matrix(CAMERA_FOV, CAMERA_HEIGHT, CAMERA_WIDTH)


def project_ego_to_image(pos_xyz):
    """Ego/vehicle frame (CARLA x-front, y-right, z-up) -> image (u, v, depth).
    Returns None if behind the camera. Mirrors create_projection_grid()."""
    p = np.asarray(pos_xyz, dtype=np.float64) - np.asarray(CAMERA_POS)
    # CARLA (x front, y right, z up) -> pinhole (x right, y down, z front)
    cam = np.array([p[1], p[2], p[0]])          # [y, z, x]
    depth = cam[2]
    if depth <= 0.1:
        return None                              # behind / at camera plane
    uv = K @ cam
    return uv[0] / depth, uv[1] / depth, depth


def keep_box(b):
    """Replicates parse_bounding_boxes filtering so targets match the model's set."""
    if b.get("class") not in CLASS_TO_ID:
        return False
    if "num_points" in b:
        if b["class"] == "walker" and b["num_points"] <= NUM_LIDAR_HITS_WALKER:
            return False
        if b["class"] == "car" and b["num_points"] <= NUM_LIDAR_HITS_CAR:
            return False
    if b["class"] == "traffic_light":
        if not b.get("affects_ego") or b.get("state") == "Green":
            return False
    if b["class"] == "stop_sign" and not b.get("affects_ego"):
        return False
    x, y, z = b["position"]
    if not (MIN_X < x < MAX_X and MIN_Y < y < MAX_Y and MIN_Z < z < MAX_Z):
        return False
    return True


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


def draw_marks(img, marks):
    """marks: list of (id, u, v). Draws a labeled dot per object."""
    out = img.copy()
    for oid, u, v in marks:
        u, v = int(round(u)), int(round(v))
        cv2.circle(out, (u, v), 6, (0, 255, 0), -1)
        cv2.putText(out, str(oid), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def process_frame(rgb_path, boxes_path, model, processor, debug_dir=None):
    with open(boxes_path, "r", encoding="utf-8") as f:
        boxes = ujson.load(f)

    kept = [b for b in boxes if keep_box(b)]               # filtered GT set
    n = len(kept)
    # target rows: [x, y, importance, class_id]; importance defaults to 0.
    targets = np.zeros((n, 4), dtype=np.float32)
    for i, b in enumerate(kept):
        x, y, _ = b["position"]
        targets[i, 0], targets[i, 1] = x, y
        targets[i, 3] = CLASS_TO_ID[b["class"]]

    # Which kept objects are visible in the front camera -> get a mark id.
    marks, id_to_row, object_lines = [], {}, []
    for i, b in enumerate(kept):
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
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    # route folders that have both rgb/ and boxes/
    routes = sorted({str(Path(p).parents[1]) for p in
                     glob.glob(os.path.join(args.root_dir, "**", "boxes", "*.json"), recursive=True)})
    print(f"found {len(routes)} route folders")

    done = 0
    for route in routes:
        box_files = sorted(glob.glob(os.path.join(route, "boxes", "*.json")))
        if args.output_dir:
            route_rel = os.path.relpath(route, args.root_dir)
            out_dir = os.path.join(args.output_dir, route_rel, args.out_subdir)
        else:
            out_dir = os.path.join(route, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        for bf in box_files:
            stem = Path(bf).stem
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
