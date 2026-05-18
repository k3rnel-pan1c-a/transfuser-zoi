import argparse
import gc
import json
import os
import re
from pathlib import Path

import cv2
import torch
from PIL import Image

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


# ---------------------------------------------------------------------------
# Prompt variants
# Use --prompt_mode full | short | strict  (default: full)
#
# Evaluation tips:
#   Good output:  2-5 boxes, each with a spatial reason tied to the ego lane.
#   Bad output:   >8 boxes, boxes on buildings/sky, boxes on every parked car,
#                 identical boxes across very different scenes.
#
# Common failure modes and fixes:
#   1. Model marks every visible car.
#      Fix: strengthen the exclusion list; add "not every parked or distant car"
#           explicitly in the INCLUDE section.
#   2. Model marks background (sky, buildings, trees).
#      Fix: the EXCLUDE section already lists these; if still happening, move
#           exclusions to the top of the prompt.
#   3. Model hallucinates objects not in the image.
#      Fix: add "only annotate what you can clearly see; do not guess."
#   4. Model outputs >10 boxes on a simple highway scene.  #      Fix: lower the box cap; add "prefer 2-4 boxes over 8 marginal ones."
#   5. Model marks traffic lights for opposing lanes.
#      Fix: prompt already says "governs the ego lane"; reinforce with
#           "ignore traffic lights clearly facing away from the ego vehicle."
# ---------------------------------------------------------------------------

# Full prompt — use this for all real experiments.
PROMPT_FULL = """You are a planning-aware zone-of-interest (ZOI) annotator for an autonomous driving system.

Your task is to identify ONLY the spatial regions in this driving image that are planning-relevant — meaning they may directly affect the ego vehicle's near-future driving decisions such as braking, steering, yielding, turning, lane changes, or obstacle avoidance.

This is NOT general object detection. Do NOT annotate every visible object. Select only what matters for the immediate driving decision.

TASK FRAMING
The ego vehicle is the camera vehicle. The image is a front-facing or multi-camera driving view. You must identify spatial regions that constrain or influence what the ego vehicle should do in the next few seconds.

SELECT THESE — but only if spatially relevant to the ego vehicle:
- A vehicle directly ahead in the ego lane or actively merging into it.
- A vehicle cutting into the ego lane or approaching from the side at an intersection.
- A stopped or slow vehicle ahead that may require braking or a lane change.
- A pedestrian near a crosswalk, at a road edge, or visibly about to step onto the road.
- A cyclist on or crossing the ego vehicle's path.
- A traffic light or traffic sign that governs the ego lane or the upcoming intersection.
- An obstacle (debris, construction cone, barrier) blocking or narrowing the current drivable lane.
- An intersection, merge zone, or turn area that the ego vehicle is actively approaching.
- A road boundary, guardrail, or lane marking that constrains the drivable corridor immediately ahead.
- An occlusion zone near a junction or crosswalk where hidden agents may suddenly emerge.

DO NOT SELECT THESE — strict exclusion rules:
- Parked cars far away or on the opposite side of the road that do not interact with the ego path.
- Vehicles far ahead or to the side that clearly do not affect the ego vehicle's current trajectory.
- Pedestrians on a distant sidewalk or far from any road crossing, not near the ego lane.
- Buildings, sky, trees, poles, walls, fences, sidewalks, and general background scenery.
- Traffic lights or signs that clearly face opposing traffic or control a different lane or direction.
- Every car in a parking lot — only annotate the specific one requiring braking, yielding, or lane adjustment.
- Lane markings in the far distance that do not constrain the current maneuver.
- Road surface texture, grass, or static objects behind the ego vehicle.

OUTPUT FORMAT
Return ONLY valid JSON. No markdown. No explanation outside the JSON block.

{
  "boxes": [
    {
      "label": "short object label",
      "planning_role": "dynamic_agent | traffic_rule | lane_boundary | navigation_corridor | static_obstacle | occlusion_risk | crosswalk | other",
      "reason": "one sentence explaining the spatial relationship to the ego vehicle and what decision it affects",
      "bbox_2d": [x1, y1, x2, y2],
      "importance": 3
    }
  ]
}

FIELD DEFINITIONS
- label: short name such as vehicle, pedestrian, cyclist, traffic_light, intersection, obstacle, cyclist, crosswalk.
- planning_role: the functional role this region plays in the driving decision.
- reason: must describe the spatial relationship to the ego vehicle and which decision it affects (e.g. braking, yielding, turning).
- bbox_2d: integer pixel coordinates [x1, y1, x2, y2] within the image bounds, x1 < x2, y1 < y2.
- importance: 5 = critical (immediate collision risk or hard constraint on the ego path), 4 = high (requires an active response such as braking or yielding), 3 = medium (should be monitored and may influence speed or steering), 2 = low (minor influence on planning), 1 = marginal.

RULES
- If no planning-relevant region is visible, return exactly: {"boxes": []}
- Do not include text, markdown, code fences, or any content outside the JSON object.
- Do not annotate objects that are not clearly visible in the image.
- Coordinates must be integers within the image pixel bounds."""

# Short prompt — use for quick iteration and sanity checks.
PROMPT_SHORT = """You are a planning-aware zone annotator for autonomous driving.

Select ONLY the regions that will affect the ego vehicle's near-future braking, steering, turning, or yielding.

INCLUDE (only if spatially relevant to ego): vehicles in or merging into the ego lane, pedestrians near the road or crosswalk, cyclists on the path, traffic lights governing the ego lane, obstacles blocking the lane, nearby intersections, occlusion zones near crossings.

EXCLUDE: parked cars far away or on the opposite side, vehicles clearly not on the ego path, pedestrians far from the road, buildings, sky, trees, traffic lights for other lanes, parking lot cars that do not interact with ego, distant lane markings, background scenery.

Return ONLY valid JSON. No markdown. No text outside the JSON.

{
  "boxes": [
    {
      "label": "short label",
      "planning_role": "dynamic_agent | traffic_rule | lane_boundary | navigation_corridor | static_obstacle | occlusion_risk | crosswalk | other",
      "reason": "why this affects the ego vehicle's next action",
      "bbox_2d": [x1, y1, x2, y2],
      "importance": 3
    }
  ]
}

Return {"boxes": []} if nothing is planning-relevant. Max 6 boxes. Prefer fewer high-confidence boxes."""

# Strict JSON-only prompt — use when the model keeps leaking text outside the JSON.
PROMPT_STRICT = """Task: planning-aware zone-of-interest annotation for autonomous driving.

Output format: a single JSON object. Nothing else. No markdown. No explanation. No code fences.

Schema:
{"boxes":[{"label":string,"planning_role":"dynamic_agent|traffic_rule|lane_boundary|navigation_corridor|static_obstacle|occlusion_risk|crosswalk|other","reason":string,"bbox_2d":[x1,y1,x2,y2],"importance":1|2|3|4|5}]}

Selection rule: annotate ONLY regions that directly affect the ego vehicle's near-future braking, steering, yielding, or turning. Ignore parked cars on the opposite side, background, scenery, pedestrians far from the road, traffic signals for other lanes, and any object not clearly visible.

Empty response when nothing is relevant: {"boxes":[]}

Max boxes: 6. Output the JSON object now."""


PROMPT = PROMPT_FULL  # active prompt — swap to PROMPT_SHORT or PROMPT_STRICT as needed


def extract_json(text: str):
    """
    Extract JSON even if the model accidentally adds text or markdown.
    """
    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON from model output.")


def sanitize_boxes(data, image_w, image_h):
    """
    Clamp boxes to image bounds and remove invalid boxes.
    """
    if isinstance(data, list):
        boxes = data
    else:
        boxes = data.get("boxes", [])

    clean = []

    for item in boxes:
        if "bbox_2d" not in item:
            continue

        box = item["bbox_2d"]

        if not isinstance(box, list) or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = map(float, box)
        except Exception:
            continue

        x1 = int(round(max(0, min(x1, image_w - 1))))
        y1 = int(round(max(0, min(y1, image_h - 1))))
        x2 = int(round(max(0, min(x2, image_w - 1))))
        y2 = int(round(max(0, min(y2, image_h - 1))))

        if x2 <= x1 or y2 <= y1:
            continue

        clean.append(
            {
                "label": str(item.get("label", "unknown")),
                "planning_role": str(item.get("planning_role", "other")),
                "reason": str(item.get("reason", "")),
                "bbox_2d": [x1, y1, x2, y2],
                "importance": int(item.get("importance", 1)),
            }
        )

    return {"boxes": clean}


def draw_boxes(image_path, data, save_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    for item in data["boxes"]:
        x1, y1, x2, y2 = item["bbox_2d"]
        label = item["label"]
        role = item["planning_role"]
        importance = item["importance"]

        thickness = max(1, min(4, importance))

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        text = f"{label} | {role} | {importance}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, text_thickness)

        # label background
        cv2.rectangle(
            img,
            (x1, max(0, y1 - th - 8)),
            (min(img.shape[1] - 1, x1 + tw + 4), y1),
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            img,
            text,
            (x1 + 2, max(12, y1 - 5)),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(save_path), img)


def run_single_model(model_id, image_paths, output_dir, max_new_tokens=768):
    model_name_safe = model_id.replace("/", "__")
    model_out = output_dir / model_name_safe
    json_out = model_out / "json"
    vis_out = model_out / "vis"
    raw_out = model_out / "raw"

    json_out.mkdir(parents=True, exist_ok=True)
    vis_out.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {model_id}")

    if "Qwen2.5-VL" in model_id:
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model: {model_id}")

    model = model_cls.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_id)

    model.eval()

    for image_path in image_paths:
        print(f"  Processing {image_path.name}")

        pil_img = Image.open(image_path).convert("RGB")
        image_w, image_h = pil_img.size

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        stem = image_path.stem

        raw_path = raw_out / f"{stem}.txt"
        raw_path.write_text(output_text, encoding="utf-8")

        try:
            parsed = extract_json(output_text)
            clean = sanitize_boxes(parsed, image_w, image_h)

            json_path = json_out / f"{stem}.json"
            json_path.write_text(
                json.dumps(clean, indent=2),
                encoding="utf-8",
            )

            vis_path = vis_out / f"{stem}.jpg"
            draw_boxes(image_path, clean, vis_path)

        except Exception as e:
            print(f"    Failed to parse/draw for {image_path.name}: {e}")

            fail_path = json_out / f"{stem}_FAILED.txt"
            fail_path.write_text(
                f"ERROR:\n{e}\n\nRAW OUTPUT:\n{output_text}",
                encoding="utf-8",
            )

    del model
    del processor
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing test images.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_zoi",
        help="Directory where JSON and visualization outputs are saved.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
        help="Hugging Face model IDs to test.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    parser.add_argument(
        "--prompt_mode",
        choices=["full", "short", "strict"],
        default="full",
        help="Which prompt variant to use (full | short | strict).",
    )

    args = parser.parse_args()

    global PROMPT
    if args.prompt_mode == "short":
        PROMPT = PROMPT_SHORT
    elif args.prompt_mode == "strict":
        PROMPT = PROMPT_STRICT
    else:
        PROMPT = PROMPT_FULL

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    image_paths = []
    for ext in exts:
        image_paths.extend(sorted(image_dir.glob(ext)))

    image_paths = sorted(image_paths)

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"Found {len(image_paths)} images.")

    for model_id in args.models:
        run_single_model(
            model_id=model_id,
            image_paths=image_paths,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
