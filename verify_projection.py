"""
Render the Set-of-Marks projection for given frames using BOTH z-sign conventions:
  GREEN = z_sign=-1 (corrected; should land ON objects)
  RED   = z_sign=+1 (old bug; floats above objects)
Brightens dark night frames so the cars are visible in the overlay.
Saves <out_dir>/<tag>_projcheck.jpg per frame.

Usage:
  python verify_projection.py --frames \
     "DynamicObjectCrossing/.../route 0035" "PedestrianCrossing/.../route 0050" \
     --out_dir /kaggle/working/zoi_preview
"""
import argparse
import os
import cv2
import numpy as np
import ujson

from zoi_projection import visible_marks, draw_marks, CAMERA_WIDTH, CAMERA_HEIGHT


def brighten(img, gamma=0.5):
    inv = 1.0 / gamma
    table = ((np.arange(256) / 255.0) ** inv * 255).astype("uint8")
    return cv2.LUT(img, table)


def render(route_dir, stem, out_path):
    bf = os.path.join(route_dir, "boxes", f"{stem}.json")
    rgb = os.path.join(route_dir, "rgb", f"{stem}.jpg")
    if not (os.path.isfile(bf) and os.path.isfile(rgb)):
        print(f"  [skip] missing {bf} or {rgb}")
        return None
    with open(bf) as f:
        boxes = ujson.load(f)
    good = visible_marks(boxes, z_sign=-1.0)   # correct
    bad = visible_marks(boxes, z_sign=+1.0)    # old bug
    img = brighten(cv2.imread(rgb))
    img = draw_marks(img, bad, color=(0, 0, 255), txt_color=(0, 0, 255))      # red
    img = draw_marks(img, good, color=(0, 255, 0), txt_color=(0, 255, 255))   # green
    cv2.imwrite(out_path, img)
    dv = np.mean([g[2] - b[2] for g, b in zip(good, bad)]) if good else 0.0
    print(f"  {os.path.basename(out_path)}: {len(good)} marks; "
          f"green is on avg {dv:+.0f}px BELOW red (should be >0 = lower/on cars)")
    return len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", required=True,
                    help='each "ROUTE_DIR STEM" (space-separated inside the quotes)')
    ap.add_argument("--out_dir", default="/kaggle/working/zoi_preview")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for spec in args.frames:
        route_dir, stem = spec.rsplit(" ", 1)
        tag = os.path.basename(route_dir.rstrip("/")) + "_" + stem
        render(route_dir, stem, os.path.join(args.out_dir, tag + "_projcheck.jpg"))


if __name__ == "__main__":
    main()
