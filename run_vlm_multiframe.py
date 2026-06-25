"""
Run the Set-of-Marks ZOI scoring on SEVERAL frames from DIFFERENT routes/scenes,
with one or more VLM backends, and save everything under zoi_preview/ for review.

For each frame it:
  1. projects the filtered GT boxes with the verified projection (zoi_projection),
     draws numbered marks, brightens night frames, saves <tag>_marked.jpg
  2. asks each model to score every mark (0-5), saves <tag>__<model>.json + .txt
  3. writes multiframe_scores.json (all models x all frames) + a markdown table.

Backends (pick with --models):
  gemma3 : unsloth/gemma-3-12b-it-bnb-4bit      (4-bit, fits one T4; the recommended labeler)
  gemma4 : unsloth/gemma-4-12b-it               (bf16, sharded across both T4s)
  qwen   : Qwen/Qwen2.5-VL-7B-Instruct          (needs qwen_vl_utils)

Usage (frames come from _preview_frames.json by default, written by the selector):
  python run_vlm_multiframe.py --models gemma3 --max_new_tokens 256
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

OUT_DIR = "/kaggle/working/zoi_preview"
MODEL_IDS = {
    "gemma3": "unsloth/gemma-3-12b-it-bnb-4bit",
    "gemma4": "unsloth/gemma-4-12b-it",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}


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
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
            prompt = PROMPT_TEMPLATE.format(object_list="\n".join(object_lines))
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
             f"Models: {', '.join(args.models)}\n"]
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
