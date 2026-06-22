"""
Offline ZOI label generator (VLM-DETECTS variant).

Unlike generate_zoi_labels.py (Set-of-Marks), here the VLM actually predicts
planning-relevant boxes itself, then we MATCH those predictions to the CARLA GT
boxes:

  1. VLM (Qwen2.5-VL) -> 2D boxes in image space, each with an importance 1-5.
  2. Load + filter GT boxes (same set as parse_bounding_boxes).
  3. Project each GT 3D box -> 2D image AABB (repo projection convention).
  4. IoU-match VLM boxes to GT AABBs (Hungarian, with an IoU gate).
  5. For each matched GT box: importance = matched VLM importance (->[0,1]).
     Unmatched GT boxes -> importance 0. Unmatched VLM boxes -> discarded.
  6. Save per-frame target [x, y, importance, class_id] in BEV ego meters,
     ONE ROW PER FILTERED GT BOX  ==  identical format to the Set-of-Marks
     generator, so the supervision / dataloader code is unchanged.

The position always comes from GT (exact); only importance comes from the VLM.
This is the "VLM detects, then match" design. It is noisier than Set-of-Marks
(grounding error + IoU mismatch), so eyeball --debug_dir overlays.
"""

import argparse
import glob
import gzip
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import ujson
from PIL import Image
from scipy.optimize import linear_sum_assignment
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# --- must mirror carla_garage/team_code/config.py -----------------------------
CAMERA_POS = [-1.5, 0.0, 2.0]
CAMERA_FOV = 110
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 512
MIN_X, MAX_X = -32.0, 32.0
MIN_Y, MAX_Y = -32.0, 32.0
MIN_Z, MAX_Z = -3.0, 3.0
CLASS_TO_ID = {"car": 0, "walker": 1, "traffic_light": 2, "stop_sign": 3}
NUM_LIDAR_HITS_CAR = 1
NUM_LIDAR_HITS_WALKER = 1
IOU_GATE = 0.1          # minimum IoU for a VLM<->GT match to count
# ------------------------------------------------------------------------------


def intrinsic_matrix(fov, height, width):
    f = width / (2.0 * np.tan(fov * np.pi / 360.0))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


K = intrinsic_matrix(CAMERA_FOV, CAMERA_HEIGHT, CAMERA_WIDTH)


def project_point(p_ego):
    """Ego (CARLA x-front,y-right,z-up) -> (u, v, depth) or None if behind."""
    p = np.asarray(p_ego, dtype=np.float64) - np.asarray(CAMERA_POS)
    cam = np.array([p[1], p[2], p[0]])          # -> pinhole (x right, y down, z front)
    depth = cam[2]
    if depth <= 0.1:
        return None
    uv = K @ cam
    return uv[0] / depth, uv[1] / depth, depth


def gt_box_to_aabb(b):
    """Project a GT 3D box (position, extent, yaw) to a 2D image AABB.
    Returns [x1,y1,x2,y2] clamped to image, or None if not in front."""
    cx, cy, cz = b["position"]
    ex, ey, ez = b.get("extent", [0.5, 0.5, 0.75])   # CARLA extents are half-sizes
    yaw = b.get("yaw", 0.0)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    us, vs = [], []
    for dx in (-ex, ex):
        for dy in (-ey, ey):
            for dz in (-ez, ez):
                corner = np.array([cx, cy, cz]) + R @ np.array([dx, dy, dz])
                proj = project_point(corner)
                if proj is None:
                    return None                  # any corner behind -> skip (POC)
                u, v, _ = proj
                us.append(u); vs.append(v)
    x1, y1 = max(0, min(us)), max(0, min(vs))
    x2, y2 = min(CAMERA_WIDTH - 1, max(us)), min(CAMERA_HEIGHT - 1, max(vs))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def keep_box(b):
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
    return MIN_X < x < MAX_X and MIN_Y < y < MAX_Y and MIN_Z < z < MAX_Z


# Detection prompt (same spirit as test_zoi_qwen.py).
PROMPT = """You are evaluating autonomous-driving planning relevance.
Identify regions in the image that can affect the EGO vehicle's near-future path,
speed, safety, or decisions (interacting vehicles, pedestrians, cyclists, red
traffic lights facing ego, relevant signs, obstacles).

Return ONLY valid JSON:
{
  "boxes": [
    {"label": "short", "planning_role": "dynamic_agent|traffic_rule|static_obstacle|other",
     "bbox_2d": [x1, y1, x2, y2], "importance": 3}
  ]
}
Rules: bbox_2d in pixel coords, x1<x2, y1<y2; importance integer 1-5;
at most 10 boxes; prefer few high-confidence boxes; no text/markdown outside JSON.
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


def run_vlm(pil, model, processor):
    messages = [{"role": "user", "content": [{"type": "image", "image": pil},
                                             {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512)
    gen = gen[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(gen, skip_special_tokens=True)[0]
    boxes = []
    try:
        for it in extract_json(out).get("boxes", []):
            bb = it.get("bbox_2d")
            if isinstance(bb, list) and len(bb) == 4:
                x1, y1, x2, y2 = map(float, bb)
                if x2 > x1 and y2 > y1:
                    boxes.append({"bbox": [x1, y1, x2, y2],
                                  "imp": float(it.get("importance", 0))})
    except Exception as e:
        print(f"  [warn] VLM parse failed: {e}")
    return boxes


def process_frame(rgb_path, boxes_path, model, processor, debug_dir=None):
    with gzip.open(boxes_path, "rt", encoding="utf-8") as f:
        gt = [b for b in ujson.load(f) if keep_box(b)]

    targets = np.zeros((len(gt), 4), dtype=np.float32)
    for i, b in enumerate(gt):
        targets[i, 0], targets[i, 1] = b["position"][0], b["position"][1]
        targets[i, 3] = CLASS_TO_ID[b["class"]]

    img = cv2.imread(str(rgb_path))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    vlm_boxes = run_vlm(pil, model, processor)

    # project GT to 2D AABBs (only camera-visible ones can be matched)
    gt_aabb = [(i, gt_box_to_aabb(b)) for i, b in enumerate(gt)]
    gt_aabb = [(i, a) for i, a in gt_aabb if a is not None]

    if vlm_boxes and gt_aabb:
        cost = np.ones((len(vlm_boxes), len(gt_aabb)), dtype=np.float64)
        for vi, vb in enumerate(vlm_boxes):
            for gi, (_, a) in enumerate(gt_aabb):
                cost[vi, gi] = 1.0 - iou(vb["bbox"], a)     # minimize 1-IoU
        rows, cols = linear_sum_assignment(cost)
        for vi, gi in zip(rows, cols):
            if (1.0 - cost[vi, gi]) >= IOU_GATE:            # passes IoU gate
                gt_row = gt_aabb[gi][0]
                targets[gt_row, 2] = np.clip(vlm_boxes[vi]["imp"] / 5.0, 0.0, 1.0)

    if debug_dir is not None:
        dbg = img.copy()
        for _, a in gt_aabb:
            cv2.rectangle(dbg, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 0), 1)
        for vb in vlm_boxes:
            x1, y1, x2, y2 = map(int, vb["bbox"])
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(dbg, f'{vb["imp"]:.0f}', (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, Path(rgb_path).stem + ".jpg"), dbg)

    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--out_subdir", default="zoi_labels")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--debug_dir", default=None)
    ap.add_argument("--debug_n", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    model.eval()

    routes = sorted({str(Path(p).parents[1]) for p in
                     glob.glob(os.path.join(args.root_dir, "**", "boxes", "*.json.gz"), recursive=True)})
    print(f"found {len(routes)} route folders")

    done = 0
    for route in routes:
        out_dir = os.path.join(route, args.out_subdir)
        os.makedirs(out_dir, exist_ok=True)
        for bf in sorted(glob.glob(os.path.join(route, "boxes", "*.json.gz"))):
            stem = Path(bf).stem.replace(".json", "")
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
