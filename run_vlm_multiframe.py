"""
Run the Set-of-Marks ZOI scoring on SEVERAL frames from DIFFERENT routes/scenes,
with one or more VLM backends, and save everything under zoi_preview/ for review.

For each frame it:
  1. projects the filtered GT boxes with the verified projection (zoi_projection),
     draws numbered marks, brightens night frames, saves <tag>_marked.jpg
  2. asks each model to score every mark (0-5), saves <tag>__<model>.json + .txt
  3. writes multiframe_scores.json (all models x all frames) + a markdown table.

Backends (pick with --models) — all full-precision bf16 for an apples-to-apples comparison:
  gemma3   : unsloth/gemma-3-12b-it       (bf16, sharded across both T4s)
  gemma4   : unsloth/gemma-4-12b-it       (bf16, sharded across both T4s)
  internvl : OpenGVLab/InternVL3_5-8B-HF  (bf16, sharded across both T4s)
  qwen     : Qwen/Qwen2.5-VL-7B-Instruct  (bf16; needs qwen_vl_utils)

Usage (frames come from _preview_frames.json by default, written by the selector):
  python run_vlm_multiframe.py --models gemma3 gemma4 qwen internvl \
      --prompt compact --out_dir /kaggle/working/zoi_preview_compare

--prompt options (all use compact JSON output to avoid token-cap truncation):
  compact          — bare 0-5 scale only (original)
  exemplar         — same as compact + one concrete driving scenario per score level
  egopathconflict  — alternative framing: "does this object conflict with the ego path?"
  full             — adds per-id reason/role text (verbose; can truncate on many marks)
"""
import argparse
import gc
import json
import os
import time

import cv2
import numpy as np
import ujson
from PIL import Image
import torch

from zoi_projection import visible_marks, draw_marks
from generate_zoi_labels import PROMPT_TEMPLATE, extract_json
from zoi_prompts import EXEMPLAR_PROMPT, EGOPATHCONFLICT_PROMPT

OUT_DIR = "/kaggle/working/zoi_preview"
# All full-precision (bf16), no 4-bit, so the comparison is apples-to-apples. The 12B
# Gemmas don't fit one 15GB T4 in bf16 (~24GB) -> device_map="auto" shards across both T4s.
MODEL_IDS = {
    "gemma3": "unsloth/gemma-3-12b-it",      # bf16 (was the -bnb-4bit mirror)
    "gemma4": "unsloth/gemma-4-12b-it",      # bf16
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl": "OpenGVLab/InternVL3_5-8B-HF",  # bf16 (no 12B variant; 8B is the safe fit)
}

# Compact prompt: only id+importance -> short JSON that won't hit the token cap.
COMPACT_PROMPT_TEMPLATE = """You are a planning-relevance annotator for an autonomous driving system.
The image is the EGO vehicle's front camera. Numbered green markers are already placed on
objects detected by the simulator's ground truth — you are NOT detecting objects. For EACH
numbered marker, rate how planning-relevant that object is to the ego's near-future driving
decisions (braking, steering, yielding, lane changes, obstacle avoidance) over the next few
seconds.

IMPORTANCE SCALE (0-5):
  5 = critical  — object is IN the ego path right now; immediate braking or swerving required
  4 = high      — object will enter ego path within 1-2 s; ego must respond now
  3 = medium    — object is near the ego path and worth monitoring; may require speed/steering adjustment
  2 = low       — object is in an adjacent lane with no sign of merging, or is >25 m away in a non-threatening position
  1 = marginal  — object is clearly in a different lane and far away (>25 m), or moving away from ego
  0 = irrelevant — object is parked, stationary far off-path, in an opposing lane going the other way,
                   or so far away (>40 m) that no action is needed regardless of what it does

DEFAULT RULE: if an object is in a DIFFERENT lane from the ego and shows NO sign of merging or cutting in,
score it 0-1 regardless of distance. Reserve 2+ only for objects on or converging toward the ego lane.

The numbered objects are:
{object_list}

Return ONLY compact JSON, no prose, no markdown, no code fences:
{{"scores": [{{"id": 0, "importance": 3}}]}}
Score every id listed."""


def brighten(img, gamma=0.5):
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype("uint8")
    return cv2.LUT(img, table)


def frame_tag(route_dir):
    return os.path.basename(route_dir.rstrip("/"))


def build_marked(route_dir, stem):
    """Return (marked_BGR, object_lines, marks) or None if no visible marks."""
    bf = os.path.join(route_dir, "boxes", f"{stem}.json")
    rgb = os.path.join(route_dir, "rgb", f"{stem}.jpg")
    if not (os.path.isfile(bf) and os.path.isfile(rgb)):
        return None
    boxes = ujson.load(open(bf))
    marks = visible_marks(boxes)                      # (id,u,v,depth,class)
    if not marks:
        return None
    img = brighten(cv2.imread(rgb))
    marked = draw_marks(img, marks)
    object_lines = [f'  id {m[0]}: a {m[4]} at ~{m[3]:.0f} m' for m in marks]
    return marked, object_lines, marks


# ----------------------------- model backends ------------------------------
def load_model(kind):
    mid = MODEL_IDS[kind]
    print(f"[load] {kind}: {mid}")
    if kind == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map="auto")
        proc = AutoProcessor.from_pretrained(mid)
    else:  # gemma3 (4-bit mirror) / gemma4 (bf16 sharded)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map="auto")
        proc = AutoProcessor.from_pretrained(mid)
    model.eval()
    return model, proc


def generate(kind, model, proc, pil, prompt, max_new_tokens):
    if kind == "qwen":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, padding=True,
                      return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = gen[:, inputs.input_ids.shape[1]:]
        return proc.batch_decode(gen, skip_special_tokens=True)[0]
    # gemma
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
    inputs = proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    ilen = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(gen[0][ilen:], skip_special_tokens=True)


def scores_from_raw(raw, marks):
    """Parse -> {id: importance(0-5)}; missing ids default to None."""
    out = {m[0]: None for m in marks}
    try:
        parsed = extract_json(raw)
        for s in parsed.get("scores", []):
            oid = int(s["id"])
            if oid in out:
                out[oid] = float(s.get("importance", 0))
    except Exception as e:
        print(f"    [warn] parse failed: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_json", default="_preview_frames.json")
    ap.add_argument("--models", nargs="+", default=["gemma3"],
                    choices=list(MODEL_IDS))
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--prompt",
                    choices=["compact", "full", "exemplar", "egopathconflict"],
                    default="compact",
                    help="compact = id+importance only; full = +reason/role text; "
                         "exemplar = compact + per-score driving examples; "
                         "egopathconflict = alternative spatial-conflict framing")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    prompt_template = {
        "compact": COMPACT_PROMPT_TEMPLATE,
        "full": PROMPT_TEMPLATE,
        "exemplar": EXEMPLAR_PROMPT,
        "egopathconflict": EGOPATHCONFLICT_PROMPT,
    }[args.prompt]

    picks = json.load(open(args.frames_json))
    # build all marked images first (no model needed)
    frames = []   # (tag, stem, marked_pil, object_lines, marks)
    for route_dir, stem in picks:
        built = build_marked(route_dir, stem)
        if built is None:
            print(f"[skip] no visible marks: {route_dir} {stem}")
            continue
        marked, object_lines, marks = built
        tag = f"{frame_tag(route_dir)}_{stem}"
        cv2.imwrite(os.path.join(args.out_dir, f"{tag}_marked.jpg"), marked)
        pil = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
        frames.append((tag, route_dir, stem, pil, object_lines, marks))
    print(f"prepared {len(frames)} marked frames")

    results = {}   # tag -> {meta, scores:{model:{id:imp}}}
    for tag, route_dir, stem, pil, object_lines, marks in frames:
        results[tag] = {
            "route": route_dir, "stem": stem,
            "classes": {m[0]: m[4] for m in marks},
            "depth_m": {m[0]: round(m[3], 1) for m in marks},
            "scores": {},
        }

    for kind in args.models:
        model, proc = load_model(kind)
        for tag, route_dir, stem, pil, object_lines, marks in frames:
            prompt = prompt_template.format(object_list="\n".join(object_lines))
            t0 = time.time()
            raw = generate(kind, model, proc, pil, prompt, args.max_new_tokens)
            dt = time.time() - t0
            sc = scores_from_raw(raw, marks)
            results[tag]["scores"][kind] = sc
            with open(os.path.join(args.out_dir, f"{tag}__{kind}.txt"), "w") as f:
                f.write(raw)
            with open(os.path.join(args.out_dir, f"{tag}__{kind}.json"), "w") as f:
                json.dump(sc, f, indent=2)
            shown = {k: (round(v, 1) if v is not None else None) for k, v in sc.items()}
            print(f"  [{kind}] {tag} ({dt:4.1f}s): {shown}")
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "multiframe_scores.json"), "w") as f:
        json.dump(results, f, indent=2)

    # markdown comparison
    lines = ["# Multi-frame ZOI VLM comparison\n",
             f"prompt mode: **{args.prompt}**, max_new_tokens={args.max_new_tokens}\n",
             "Models:\n"]
    for m in args.models:
        lines.append(f"- `{m}` = {MODEL_IDS[m]}")
    lines.append("\nimportance 0-5 (blank = the model didn't return / parse that id)\n")
    for tag, r in results.items():
        lines.append(f"\n## {tag}\nscene: `{os.path.basename(r['route'])}` frame {r['stem']}\n")
        ids = sorted(r["classes"])
        head = "| id | class | depth |" + "".join(f" {m} |" for m in args.models)
        sep = "|---|---|---|" + "---|" * len(args.models)
        lines += [head, sep]
        for i in ids:
            row = f"| {i} | {r['classes'][i]} | {r['depth_m'][i]}m |"
            for m in args.models:
                v = r["scores"].get(m, {}).get(i)
                row += f" {('' if v is None else round(v,1))} |"
            lines.append(row)
    with open(os.path.join(args.out_dir, "multiframe_compare.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote multiframe_scores.json + multiframe_compare.md to {args.out_dir}")


if __name__ == "__main__":
    main()
