#!/usr/bin/env python3
"""
teacher_bakeoff.py — multi-teacher VLM comparison for ZOI supervision (portable kit).

Labels the SAME set of frames with several candidate teacher VLMs using the RICH
schema (importance score + structured tags + one-sentence rationale — ZOI_CONTEXT.md
sec 23.2) so the teachers can be compared apples-to-apples by eval_bakeoff.py, and so
the per-teacher .npy trees are drop-in for attention_probe.py (--labels_root).

Designed to run on a machine that has ONLY:
    teacher_bakeoff.py  zoi_projection.py  zoi_prompts.py  eval_bakeoff.py
plus the raw carla_garage data dump (double-nested <Scenario>/<Scenario>/<route>/
with rgb/*.jpg and boxes/*.json — gzipped .json.gz also handled). No carla, no
carla_garage checkout, no torch model code from the repo is needed.

Teachers (pick with --teachers, comma-separated; run any subset):
    gemma4     unsloth/gemma-4-12b-it                current production teacher (sec 18b)
    cosmos8b   nvidia/Cosmos-Reason2-8B              NVIDIA physical-AI reasoning VLM
    cosmos32b  nvidia/Cosmos-Reason2-32B             bigger Cosmos (needs ~66 GB bf16
                                                     or --load_4bit)
    cosmos2b   nvidia/Cosmos-Reason2-2B              cheap sanity point
    qwen3vl8b  Qwen/Qwen3-VL-8B-Instruct             Cosmos2-8B's BASE model — isolates
                                                     what NVIDIA's post-training adds
    internvl   OpenGVLab/InternVL3_5-8B-HF           known reference point (sec 17)
    mock       no GPU / no model — deterministic geometry-based scores; use it to
               verify the whole pipeline end-to-end before burning GPU time

Cosmos models are reasoning VLMs (Qwen3-VL based): they get the <think>/<answer>
system instruction from the model card, a 4096-token budget, and the parser strips
the reasoning trace and takes the LAST parseable {"scores": ...} JSON object.

Every teacher labels the IDENTICAL frame list: the first run writes
<out_root>/frames.json and later runs (any teacher) reuse it.

Label-quality guards from ZOI_CONTEXT.md sec 22.4 are all active: exposure
normalization of the VLM input, rich object lines (longitudinal + lateral + radial +
lane hint), and the completeness guard (targeted retry for omitted ids; objects still
missing after retry are DROPPED from the .npy, never silently 0.0, and logged).

Outputs per teacher under <out_root>/<teacher>/:
    labels/<route_rel>/zoi_labels/<stem>.npy   [M,4] = [x, y, imp 0..1, class_id]
                                               (same schema as training labels ->
                                               feed to attention_probe.py --labels_root)
    rich_records.jsonl                         one JSON per frame: per-object imp /
                                               imp_std / path / act / urg / why / src,
                                               omissions, timing, parse status
    overlays/<n>__<route>__<stem>.jpg          first --debug_n marked+scored frames
    raw/<n>__<route>__<stem>.txt               first --debug_n raw model outputs
    summary.json                               counts + timing + omission/parse rates

Typical run on the friend's box (see README_BAKEOFF.md):
    python teacher_bakeoff.py --root_dir /path/to/carla-garage-dump \\
        --out_root ./bakeoff_out --teachers mock --per_scenario 4        # pipeline check
    python teacher_bakeoff.py --root_dir /path/to/carla-garage-dump \\
        --out_root ./bakeoff_out --teachers gemma4,cosmos8b --per_scenario 40
    python teacher_bakeoff.py --root_dir ... --out_root ./bakeoff_out \\
        --teachers cosmos32b --load_4bit                                 # if <70 GB VRAM
    python eval_bakeoff.py --out_root ./bakeoff_out
"""

import argparse
import gc
import glob
import gzip
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import cv2
from PIL import Image

# Single source of truth for geometry + filtering + signal rules (verified z-sign).
from zoi_projection import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CLASS_TO_ID,
    VLM_MARK_CLASSES, RULE_CLASSES, rule_importance,
    project_ego_to_image, keep_box, draw_marks, describe_object_line,
)
from zoi_prompts import RICH_PROMPT

# ------------------------------------------------------------------------------
# Teacher registry
# ------------------------------------------------------------------------------
TEACHERS = {
    "gemma4":    dict(model_id="unsloth/gemma-4-12b-it",          family="imagetext", reasoning=False),
    "cosmos8b":  dict(model_id="nvidia/Cosmos-Reason2-8B",        family="qwen3vl",   reasoning=True),
    "cosmos32b": dict(model_id="nvidia/Cosmos-Reason2-32B",       family="qwen3vl",   reasoning=True),
    "cosmos2b":  dict(model_id="nvidia/Cosmos-Reason2-2B",        family="qwen3vl",   reasoning=True),
    "qwen3vl8b": dict(model_id="Qwen/Qwen3-VL-8B-Instruct",       family="qwen3vl",   reasoning=False),
    "internvl":  dict(model_id="OpenGVLab/InternVL3_5-8B-HF",     family="imagetext", reasoning=False),
    "mock":      dict(model_id=None,                              family="mock",      reasoning=False),
}

# Model-card instruction for the Cosmos reasoning format; the parser strips the
# <think> block and reads the JSON from the <answer> (or trailing) text.
THINK_SYS = ("Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n"
             "<answer>\nyour answer\n</answer>")

TAG_VOCAB = {
    "path": {"yes", "no", "maybe"},
    "act": {"brake", "yield", "monitor", "ignore"},
    "urg": {"now", "soon", "none"},
}


# ------------------------------------------------------------------------------
# Small shared helpers (inlined from generate_zoi_labels*.py so this kit is
# self-contained — the friend's machine does not need the other generators).
# ------------------------------------------------------------------------------
def load_boxes(boxes_path):
    if str(boxes_path).endswith(".gz"):
        with gzip.open(boxes_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(boxes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_exposure(img_bgr, target=120.0, gamma_floor=0.4, clahe_clip=2.0):
    """Sec 22.2: dark (night/dusk/rain) frames made Gemma silently drop visible
    objects; adaptive gamma lift + light CLAHE recovers them. Never darkens."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    out = img_bgr
    if 1.0 < mean < target:
        gamma = np.log(target / 255.0) / np.log(mean / 255.0)
        gamma = float(np.clip(gamma, gamma_floor, 1.0))
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).astype(np.uint8)
        out = cv2.LUT(img_bgr, lut)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def strip_reasoning(text):
    """Drop a reasoning model's <think> trace; prefer the <answer> block if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = re.search(r"<answer>(.*?)(</answer>|$)", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def extract_scores_json(text):
    """Return the LAST parseable JSON object containing a "scores" key. Tolerates
    code fences, prose around the JSON, and leftover reasoning text."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    dec = json.JSONDecoder()
    best = None
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            break
        try:
            obj, consumed = dec.raw_decode(text[i:])
            if isinstance(obj, dict) and "scores" in obj:
                best = obj
            i += max(consumed, 1)
        except Exception:
            i += 1
    if best is None:
        raise ValueError("no parseable {'scores': ...} JSON in model output")
    return best


def parse_rich_entries(parsed):
    """{"scores":[{id, imp|importance, path, act, urg, why}, ...]} -> {oid: fields}.
    Out-of-vocab tags become None (counted by eval as incomplete, never guessed)."""
    per_id = {}
    for s in parsed.get("scores", []):
        try:
            oid = int(s["id"])
        except (KeyError, TypeError, ValueError):
            continue
        imp_raw = s.get("imp", s.get("importance"))
        if imp_raw is None:
            continue
        try:
            imp = float(np.clip(float(imp_raw) / 5.0, 0.0, 1.0))
        except (TypeError, ValueError):
            continue
        rec = {"imp": imp}
        for k in TAG_VOCAB:
            v = s.get(k)
            v = v.strip().lower() if isinstance(v, str) else None
            rec[k] = v if v in TAG_VOCAB[k] else None
        why = s.get("why")
        rec["why"] = why.strip()[:200] if isinstance(why, str) and why.strip() else None
        per_id[oid] = rec
    return per_id


# ------------------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------------------
class MockVLM:
    """Deterministic geometry-based scorer for pipeline testing without a GPU.
    Deliberately omits some ids on the first pass so the retry guard is exercised."""

    def __init__(self):
        self.ctx = {}          # oid -> (x, y, radial)
        self.frame_key = ""

    def set_context(self, frame_key, ctx):
        self.frame_key, self.ctx = frame_key, ctx

    def generate(self, prompt, sample=False):
        is_retry = "missed scoring" in prompt
        entries = []
        for oid, (x, y, radial) in self.ctx.items():
            if not is_retry:
                h = int(hashlib.md5(f"{self.frame_key}:{oid}".encode()).hexdigest(), 16)
                if h % 6 == 0:          # simulate an omission -> retry path
                    continue
            in_lane = abs(y) < 1.75 and x > 0
            imp = 5 if radial < 8 else 4 if radial < 15 else 3 if radial < 22 else 2 if radial < 30 else 1
            if not in_lane:
                imp = max(0, imp - 2)
            entries.append({"id": oid, "imp": imp,
                            "path": "yes" if in_lane else "no",
                            "act": "brake" if imp >= 4 else ("monitor" if imp >= 2 else "ignore"),
                            "urg": "now" if imp == 5 else ("soon" if imp >= 3 else "none"),
                            "why": f"mock: {radial:.0f} m away, {'ego lane' if in_lane else 'off lane'}"})
        return json.dumps({"scores": entries}), 0.0


def load_teacher(name, load_4bit=False):
    spec = TEACHERS[name]
    if spec["family"] == "mock":
        return MockVLM(), None
    import torch
    import transformers
    kwargs = {"device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kwargs["dtype"] = torch.bfloat16
    cls = transformers.AutoModelForImageTextToText
    if spec["family"] == "qwen3vl":
        # Cosmos-Reason2 / Qwen3-VL native class (model card); Auto* as fallback.
        cls = getattr(transformers, "Qwen3VLForConditionalGeneration", cls)
    model = cls.from_pretrained(spec["model_id"], **kwargs)
    processor = transformers.AutoProcessor.from_pretrained(spec["model_id"])
    model.eval()
    return model, processor


def vlm_generate(name, model, processor, pil, prompt, max_new_tokens, sample=False):
    """One VLM call. Returns (raw_text, pure_generate_seconds)."""
    if isinstance(model, MockVLM):
        return model.generate(prompt, sample=sample)
    import torch
    spec = TEACHERS[name]
    messages = []
    if spec["reasoning"]:
        messages.append({"role": "system", "content": [{"type": "text", "text": THINK_SYS}]})
    messages.append({"role": "user", "content": [{"type": "image", "image": pil},
                                                 {"type": "text", "text": prompt}]})
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if sample:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen = model.generate(**inputs, **gen_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    out = processor.decode(gen[0][input_len:], skip_special_tokens=True)
    return out, gen_s


# ------------------------------------------------------------------------------
# Frame sampling (balanced across scenarios, identical for every teacher)
# ------------------------------------------------------------------------------
def discover_routes(root_dir):
    """route_dir -> scenario name. Scenario = first component of the route's path
    relative to root (works for both the double-nested raw dump and built trees).
    Targeted depth-1..3 globs instead of a recursive `**` walk: the dump has
    millions of rgb/lidar files and a full recursion takes minutes on slow/network
    storage (route layouts: <Scn>/<Scn>/<route>/boxes, <Scn>/<route>/boxes,
    <route>/boxes)."""
    box_dirs = set()
    for depth in ("*", os.path.join("*", "*"), os.path.join("*", "*", "*")):
        box_dirs |= {p for p in glob.glob(os.path.join(root_dir, depth, "boxes"))
                     if os.path.isdir(p)}
    box_dirs = {str(Path(p).parent) for p in box_dirs}
    routes = {}
    for bd in sorted(box_dirs):
        route_dir = str(Path(bd).parent)
        rel = os.path.relpath(route_dir, root_dir)
        routes[route_dir] = rel.split(os.sep)[0]
    return routes


def route_stems(route_dir, skip_first, stride):
    stems = []
    for bf in sorted(glob.glob(os.path.join(route_dir, "boxes", "*.json")) +
                     glob.glob(os.path.join(route_dir, "boxes", "*.json.gz"))):
        stem = os.path.basename(bf).split(".")[0]
        if stem.isdigit():
            fi = int(stem)
            if fi < skip_first or fi % stride != 0:
                continue
        # frame must also have an rgb image
        if os.path.isfile(os.path.join(route_dir, "rgb", stem + ".jpg")):
            stems.append(stem)
    return stems


def sample_frames(root_dir, per_scenario, skip_first, stride, seed):
    """Balanced sample: up to `per_scenario` frames per scenario, drawn round-robin
    across that scenario's routes (route/town diversity > frames-per-route)."""
    rng = random.Random(seed)
    routes = discover_routes(root_dir)
    by_scenario = {}
    for rd, scn in routes.items():
        by_scenario.setdefault(scn, []).append(rd)

    frames = []
    for scn in sorted(by_scenario):
        rds = by_scenario[scn]
        rng.shuffle(rds)
        pools = []
        for rd in rds:
            stems = route_stems(rd, skip_first, stride)
            rng.shuffle(stems)
            if stems:
                pools.append((os.path.relpath(rd, root_dir), stems))
        picked, i = 0, 0
        while picked < per_scenario and pools:
            route_rel, stems = pools[i % len(pools)]
            if stems:
                frames.append({"scenario": scn, "route": route_rel, "stem": stems.pop()})
                picked += 1
                i += 1
            else:
                pools.pop(i % len(pools))
        print(f"  {scn}: {picked} frames from {len(rds)} routes")
    return frames


def resolve_boxes(route_dir, stem):
    for ext in (".json", ".json.gz"):
        p = os.path.join(route_dir, "boxes", stem + ext)
        if os.path.isfile(p):
            return p
    return None


# ------------------------------------------------------------------------------
# Per-frame labeling with the rich schema + completeness guard
# ------------------------------------------------------------------------------
def process_frame(name, model, processor, rgb_path, boxes_path, frame_key,
                  passes, max_new_tokens):
    """Returns (targets[M,4], record dict). Rows for VLM-marked objects that stay
    unscored after the retry are DROPPED from targets (sec 22.1: never silent 0.0)."""
    boxes = load_boxes(boxes_path)
    kept = [b for b in boxes if keep_box(b)]
    n = len(kept)
    targets = np.zeros((n, 4), dtype=np.float32)
    obj_records = []
    for i, b in enumerate(kept):
        x, y, _ = b["position"]
        targets[i, 0], targets[i, 1] = x, y
        targets[i, 3] = CLASS_TO_ID[b["class"]]
        rec = {"row": i, "oid": None, "class": b["class"],
               "x": float(x), "y": float(y), "radial": float(np.hypot(x, y)),
               "imp": None, "imp_std": None, "path": None, "act": None,
               "urg": None, "why": None, "src": None}
        if b["class"] in RULE_CLASSES:
            targets[i, 2] = rule_importance(b)
            rec["imp"], rec["src"] = float(targets[i, 2]), "rule"
        obj_records.append(rec)

    marks, id_to_row, object_lines, mock_ctx = [], {}, [], {}
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
        obj_records[i]["oid"] = oid
        object_lines.append(describe_object_line(oid, b["class"], b["position"]))
        x, y, _ = b["position"]
        mock_ctx[oid] = (x, y, float(np.hypot(x, y)))

    record = {"frame": frame_key, "n_kept": n, "n_marked": len(marks),
              "objects": obj_records, "missing_after_retry": [],
              "parse_fail": False, "gen_s": 0.0, "passes": passes, "raw": None}
    if not marks:
        return targets, record, None

    img = normalize_exposure(cv2.imread(str(rgb_path)))
    marked = draw_marks(img, marks)
    pil = Image.fromarray(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    prompt = RICH_PROMPT.format(object_list="\n".join(object_lines))
    if isinstance(model, MockVLM):
        model.set_context(frame_key, mock_ctx)

    # --- main passes (pass 0 greedy; extra passes sampled for self-consistency) ---
    per_pass = []
    raw0 = None
    for p in range(passes):
        out, gen_s = vlm_generate(name, model, processor, pil, prompt,
                                  max_new_tokens, sample=(p > 0))
        record["gen_s"] += gen_s
        if p == 0:
            raw0 = out
        try:
            per_pass.append(parse_rich_entries(extract_scores_json(strip_reasoning(out))))
        except Exception as e:
            record["parse_fail"] = True
            print(f"  [warn] parse failed ({name}, pass {p}) {frame_key}: {e}")

    merged = {}
    for oid in id_to_row:
        imps = [pp[oid]["imp"] for pp in per_pass if oid in pp]
        if not imps:
            continue
        m = {"imp": float(np.mean(imps)),
             "imp_std": float(np.std(imps)) if len(imps) > 1 else 0.0}
        for k in TAG_VOCAB:
            vals = [pp[oid][k] for pp in per_pass if oid in pp and pp[oid][k]]
            m[k] = Counter(vals).most_common(1)[0][0] if vals else None
        whys = [pp[oid]["why"] for pp in per_pass if oid in pp and pp[oid]["why"]]
        m["why"] = whys[0] if whys else None
        merged[oid] = m

    # --- completeness guard: targeted greedy retry for omitted ids (sec 22.1) ---
    missing = [oid for oid in id_to_row if oid not in merged]
    if missing:
        retry_prompt = (
            f"You missed scoring these ids: {missing}.\n"
            "For EACH of them output imp (0-5), path (yes/maybe/no), act "
            "(brake/yield/monitor/ignore), urg (now/soon/none), why (max 15 words). Objects:\n"
            + "\n".join(object_lines[oid] for oid in missing)
            + '\nReply ONLY with JSON: {"scores":[{"id":N,"imp":K,"path":"...","act":"...",'
              '"urg":"...","why":"..."}]}')
        out2, gen_s2 = vlm_generate(name, model, processor, pil, retry_prompt,
                                    max_new_tokens, sample=False)
        record["gen_s"] += gen_s2
        try:
            for oid, m in parse_rich_entries(extract_scores_json(strip_reasoning(out2))).items():
                if oid in id_to_row and oid not in merged:
                    m["imp_std"] = 0.0
                    merged[oid] = m
        except Exception as e:
            print(f"  [warn] retry parse failed ({name}) {frame_key}: {e}")

    still_missing = [oid for oid in id_to_row if oid not in merged]
    record["missing_after_retry"] = still_missing

    for oid, m in merged.items():
        row = id_to_row[oid]
        targets[row, 2] = m["imp"]
        obj_records[row].update(m)
        obj_records[row]["src"] = "vlm"
    dropped_rows = set()
    for oid in still_missing:
        obj_records[id_to_row[oid]]["src"] = "dropped"
        dropped_rows.add(id_to_row[oid])
    if dropped_rows:
        keep_rows = [i for i in range(n) if i not in dropped_rows]
        targets = targets[keep_rows] if keep_rows else np.zeros((0, 4), dtype=np.float32)

    # scored overlay for eyeballing
    for oid, u, v in marks:
        m = merged.get(oid)
        txt = f"{m['imp']:.1f}" if m else "MISS"
        cv2.putText(marked, txt, (int(u) + 6, int(v) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    record["raw"] = raw0
    return targets, record, marked


# ------------------------------------------------------------------------------
def run_teacher(name, frames, args):
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)

    spec = TEACHERS[name]
    max_new = args.max_new_tokens or (4096 if spec["reasoning"] else 768)
    print(f"\n===== teacher {name} ({spec['model_id']}) max_new_tokens={max_new} =====")
    t0 = time.perf_counter()
    model, processor = load_teacher(name, load_4bit=args.load_4bit)
    print(f"loaded in {time.perf_counter() - t0:.1f}s")

    rec_path = os.path.join(out_dir, "rich_records.jsonl")
    n_done = n_marked = n_missing = n_parse_fail = 0
    gen_times = []
    run_t0 = time.perf_counter()

    with open(rec_path, "w") as rec_f:
        for fi, fr in enumerate(frames):
            route_dir = os.path.join(args.root_dir, fr["route"])
            rgb = os.path.join(route_dir, "rgb", fr["stem"] + ".jpg")
            bf = resolve_boxes(route_dir, fr["stem"])
            if bf is None or not os.path.isfile(rgb):
                print(f"  [warn] files missing for {fr['route']}/{fr['stem']}, skipped")
                continue
            frame_key = f"{fr['route']}/{fr['stem']}"
            targets, record, marked = process_frame(
                name, model, processor, rgb, bf, frame_key, args.passes, max_new)
            record.update(scenario=fr["scenario"], route=fr["route"], stem=fr["stem"],
                          teacher=name)

            lbl_dir = os.path.join(out_dir, "labels", fr["route"], "zoi_labels")
            os.makedirs(lbl_dir, exist_ok=True)
            np.save(os.path.join(lbl_dir, fr["stem"] + ".npy"), targets)

            tag = f"{fi:03d}__{fr['route'].replace(os.sep, '_')}__{fr['stem']}"
            if marked is not None and fi < args.debug_n:
                cv2.imwrite(os.path.join(out_dir, "overlays", tag + ".jpg"), marked)
                if record["raw"]:
                    with open(os.path.join(out_dir, "raw", tag + ".txt"), "w") as f:
                        f.write(record["raw"])
            raw = record.pop("raw", None)
            record["raw_len"] = len(raw) if raw else 0
            rec_f.write(json.dumps(record) + "\n")

            n_done += 1
            n_marked += record["n_marked"]
            n_missing += len(record["missing_after_retry"])
            n_parse_fail += int(record["parse_fail"])
            if record["gen_s"] > 0:
                gen_times.append(record["gen_s"])
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(frames)} frames "
                      f"({(time.perf_counter() - run_t0) / n_done:.1f}s/frame)")
            if args.limit and n_done >= args.limit:
                print("hit --limit")
                break

    wall = time.perf_counter() - run_t0
    summary = {
        "teacher": name, "model_id": spec["model_id"], "passes": args.passes,
        "load_4bit": bool(args.load_4bit), "frames": n_done, "marked_objects": n_marked,
        "missing_after_retry": n_missing,
        "omission_rate": round(n_missing / max(n_marked, 1), 4),
        "parse_fail_frames": n_parse_fail, "wall_s": round(wall, 1),
        "gen_s_mean": round(statistics.mean(gen_times), 2) if gen_times else 0.0,
        "gen_s_median": round(statistics.median(gen_times), 2) if gen_times else 0.0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{name}] {n_done} frames, omission_rate={summary['omission_rate']}, "
          f"parse_fails={n_parse_fail}, {summary['gen_s_mean']}s/frame gen "
          f"({wall / 60:.1f} min total)")

    del model, processor
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root_dir", required=True, help="raw carla_garage dump root")
    ap.add_argument("--out_root", default="./bakeoff_out")
    ap.add_argument("--teachers", default="mock",
                    help=f"comma-separated subset of {sorted(TEACHERS)}")
    ap.add_argument("--per_scenario", type=int, default=40,
                    help="frames sampled per scenario (round-robin across routes)")
    ap.add_argument("--skip_first", type=int, default=10, help="training-parity warm-up skip")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--passes", type=int, default=1,
                    help="self-consistency passes (1 = greedy only; 3 recommended if compute allows)")
    ap.add_argument("--load_4bit", action="store_true",
                    help="bitsandbytes 4-bit quantization (use for cosmos32b on <70 GB VRAM)")
    ap.add_argument("--max_new_tokens", type=int, default=0,
                    help="override generation cap (default: 4096 reasoning / 768 others)")
    ap.add_argument("--limit", type=int, default=0, help="cap frames per teacher (smoke test)")
    ap.add_argument("--debug_n", type=int, default=12,
                    help="save this many scored overlays + raw outputs per teacher")
    args = ap.parse_args()

    teachers = [t.strip() for t in args.teachers.split(",") if t.strip()]
    unknown = [t for t in teachers if t not in TEACHERS]
    if unknown:
        ap.error(f"unknown teacher(s) {unknown}; choose from {sorted(TEACHERS)}")

    os.makedirs(args.out_root, exist_ok=True)
    frames_path = os.path.join(args.out_root, "frames.json")
    if os.path.isfile(frames_path):
        with open(frames_path) as f:
            frames = json.load(f)
        print(f"reusing existing frame list: {len(frames)} frames from {frames_path}")
    else:
        print(f"sampling frames (per_scenario={args.per_scenario}, skip_first={args.skip_first}, "
              f"stride={args.stride}, seed={args.seed})")
        frames = sample_frames(args.root_dir, args.per_scenario, args.skip_first,
                               args.stride, args.seed)
        if not frames:
            sys.exit(f"no frames found under {args.root_dir} — is this the dump root?")
        with open(frames_path, "w") as f:
            json.dump(frames, f, indent=1)
        print(f"wrote {len(frames)} frames to {frames_path} (all teachers will reuse it)")

    for t in teachers:
        run_teacher(t, frames, args)

    print("\nDone. Compare teachers with:\n"
          f"  python eval_bakeoff.py --out_root {args.out_root}")


if __name__ == "__main__":
    main()
