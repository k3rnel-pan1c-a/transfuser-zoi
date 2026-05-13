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


PROMPT = """
You are evaluating autonomous-driving planning relevance.

Given the image, identify the regions that are important for near-future driving and motion planning.

Mark only regions that can affect the ego vehicle's path, speed, safety, or decision making.

Important examples:
- vehicles that may interact with ego
- pedestrians
- cyclists
- traffic lights
- traffic signs
- lane-relevant road boundaries
- intersections
- crosswalks
- road obstacles
- construction zones
- occlusion regions where hidden agents may appear
- the intended drivable corridor

Return ONLY valid JSON.

Use exactly this format:
{
  "boxes": [
    {
      "label": "short label",
      "planning_role": "dynamic_agent | traffic_rule | lane_boundary | navigation_corridor | static_obstacle | occlusion_risk | crosswalk | other",
      "reason": "short reason",
      "bbox_2d": [x1, y1, x2, y2],
      "importance": 1
    }
  ]
}

Rules:
- bbox_2d must be [x1, y1, x2, y2].
- Coordinates must be pixel coordinates in the input image.
- x1 < x2 and y1 < y2.
- importance is an integer from 1 to 5.
- Return at most 10 boxes.
- Prefer fewer high-confidence planning-relevant boxes.
- Do not include markdown.
- Do not include text outside the JSON.
"""


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

    args = parser.parse_args()

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
