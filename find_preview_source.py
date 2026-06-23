"""Find which route's boxes/0035.json produced the 7-mark preview, by replaying
keep_box + project_ego_to_image (same subroutines) and matching mark count/positions."""
import glob, os
from pathlib import Path
import ujson
from generate_zoi_labels import keep_box, project_ego_to_image, CAMERA_WIDTH, CAMERA_HEIGHT

# Approx pixel locations of the 7 marks read off 0035_fixed.jpg (u, v):
OBSERVED = [(190,285),(700,280),(655,295),(660,268),(575,285),(465,305),(975,300)]

def visible_marks(boxes):
    out = []
    for b in boxes:
        if not keep_box(b):
            continue
        proj = project_ego_to_image(b["position"])
        if proj is None:
            continue
        u, v, depth = proj
        if 0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT:
            out.append((u, v, b.get("class"), depth))
    return out

candidates = []
for bf in glob.glob("/kaggle/input/**/boxes/0035.json", recursive=True):
    try:
        with open(bf) as f:
            boxes = ujson.load(f)
    except Exception:
        continue
    marks = visible_marks(boxes)
    if len(marks) == 7:
        candidates.append((bf, marks))

print(f"routes with exactly 7 visible marks at frame 0035: {len(candidates)}")
for bf, marks in candidates:
    route = str(Path(bf).parents[1])
    print("\n", route)
    for i, (u, v, cls, d) in enumerate(sorted(marks, key=lambda m: m[0])):
        print(f"   ({u:6.1f},{v:6.1f})  {cls:14s} depth={d:5.1f}")
