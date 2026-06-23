"""
One-off: run the Set-of-Marks VLM scoring step (same subroutines as
generate_zoi_labels.process_frame) on an already-marked preview image,
since we don't have the original boxes.json for that frame on disk.
"""
import json
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from generate_zoi_labels import PROMPT_TEMPLATE, extract_json

IMAGE_PATH = "/kaggle/working/zoi_preview/0035_fixed.jpg"
# Marks visible in the image (ids 0-6); ground-truth classes aren't available
# for this frame (boxes.json not on disk), so label them generically.
OBJECT_IDS = list(range(7))

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
print(f"loading {model_id} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)
model.eval()

pil = Image.open(IMAGE_PATH).convert("RGB")
# All 7 marked objects are cars (confirmed visually), so feed the real class
# exactly like generate_zoi_labels.process_frame does (`id N: a car`).
object_lines = [f"  id {i}: a car" for i in OBJECT_IDS]
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

print("\n=== RAW MODEL OUTPUT ===")
print(out)

with open("/kaggle/working/zoi_preview/0035_vlm_raw.txt", "w") as f:
    f.write(out)

try:
    parsed = extract_json(out)
    with open("/kaggle/working/zoi_preview/0035_vlm_scores.json", "w") as f:
        json.dump(parsed, f, indent=2)
    print("\n=== PARSED ===")
    print(json.dumps(parsed, indent=2))
except Exception as e:
    print(f"\n[warn] parse failed: {e}", file=sys.stderr)
