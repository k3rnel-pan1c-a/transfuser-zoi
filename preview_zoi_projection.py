"""
Standalone preview of the Set-of-Marks projection used by generate_zoi_labels.py --
no VLM is loaded. Projects the filtered GT boxes for one frame into the front camera
image and draws the same numbered markers the VLM would see, so you can eyeball the
projection (especially the vertical/z sign) before running a full labeling pass.

Usage:
  python preview_zoi_projection.py --rgb path/to/rgb/0030.jpg --boxes path/to/boxes/0030.json --out preview.jpg
"""
import argparse

import cv2

from generate_zoi_labels import keep_box, project_ego_to_image, draw_marks, CAMERA_WIDTH, CAMERA_HEIGHT
import ujson


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--out", default="preview.jpg")
    args = ap.parse_args()

    with open(args.boxes, "r", encoding="utf-8") as f:
        boxes = ujson.load(f)

    kept = [b for b in boxes if keep_box(b)]
    print(f"{len(boxes)} GT boxes total, {len(kept)} kept after filtering")

    marks, object_lines = [], []
    for b in kept:
        proj = project_ego_to_image(b["position"])
        if proj is None:
            continue
        u, v, depth = proj
        if not (0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT):
            continue
        oid = len(marks)
        marks.append((oid, u, v))
        object_lines.append(f"  id {oid}: a {b['class']} at depth {depth:.1f}m, pos {b['position']}")

    print(f"{len(marks)} kept boxes are camera-visible (this is what the VLM is asked to score):")
    print("\n".join(object_lines) if object_lines else "  (none)")

    img = cv2.imread(args.rgb)
    if img is None:
        raise RuntimeError(f"could not read {args.rgb}")
    marked = draw_marks(img, marks)
    cv2.imwrite(args.out, marked)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
