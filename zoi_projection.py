"""
Shared, dependency-light ZOI projection + Set-of-Marks geometry.
No torch / transformers import, so preview/overlay tools and the VLM runners can
all share ONE definition of keep_box + project_ego_to_image (avoids the previous
drift where the projection lived inside the VLM-heavy generate_zoi_labels.py).

Projection convention (verified empirically with verify_projection.py overlays):
  CARLA ego frame: x-front, y-right, z-up. Camera at CAMERA_POS, zero rotation.
  OpenCV pinhole camera: x-right, y-down, z-forward.
  => cam = [ p_y, -p_z, p_x ]   with p = world_pos - CAMERA_POS
  The -p_z term (CARLA z-up -> image y-down) is what makes marks land ON objects;
  using +p_z makes them float above (the old bug). See verify_projection.py.
"""
import numpy as np

# --- must mirror carla_garage/team_code/config.py -----------------------------
CAMERA_POS = [-1.5, 0.0, 2.0]      # x, y, z mounting position of the camera
CAMERA_FOV = 110
CAMERA_WIDTH = 1024
CAMERA_HEIGHT = 512
MIN_X, MAX_X = -32.0, 32.0
MIN_Y, MAX_Y = -32.0, 32.0
MIN_Z, MAX_Z = -3.0, 3.0           # height filter; check config.min_z/max_z
CLASS_TO_ID = {"car": 0, "walker": 1, "traffic_light": 2, "stop_sign": 3}
NUM_LIDAR_HITS_CAR = 1
NUM_LIDAR_HITS_WALKER = 1

# Classes whose GT `position` actually lands ON the visible object, so a
# Set-of-Marks dot is meaningful and worth a VLM relevance judgment.
VLM_MARK_CLASSES = {"car", "walker"}
# Classes whose GT `position` is a road-level trigger / stop line (NOT the visible
# fixture) -> a projected dot lands on the road or even on another vehicle and is
# misleading. We score these by RULE instead of asking the VLM (see rule_importance).
RULE_CLASSES = {"traffic_light", "stop_sign"}
# ------------------------------------------------------------------------------


def intrinsic_matrix(fov, height, width):
    f = width / (2.0 * np.tan(fov * np.pi / 360.0))
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


K = intrinsic_matrix(CAMERA_FOV, CAMERA_HEIGHT, CAMERA_WIDTH)


def project_ego_to_image(pos_xyz, z_sign=-1.0):
    """Ego frame (CARLA x-front, y-right, z-up) -> image (u, v, depth).
    Returns None if behind the camera. z_sign=-1.0 is the correct/verified value;
    z_sign=+1.0 reproduces the old floats-above-objects bug (kept for the overlay).
    """
    p = np.asarray(pos_xyz, dtype=np.float64) - np.asarray(CAMERA_POS)
    cam = np.array([p[1], z_sign * p[2], p[0]])   # [y, -z, x]
    depth = cam[2]
    if depth <= 0.1:
        return None
    uv = K @ cam
    return uv[0] / depth, uv[1] / depth, depth


def keep_box(b):
    """Replicates parse_bounding_boxes filtering so targets match the model's set."""
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
    if not (MIN_X < x < MAX_X and MIN_Y < y < MAX_Y and MIN_Z < z < MAX_Z):
        return False
    return True


def rule_importance(b):
    """Rule-based importance (0..1) for classes we do NOT show to the VLM.
    keep_box already restricts traffic lights to red + affects_ego and stop signs
    to affects_ego, so a kept signal is by definition planning-relevant."""
    c = b.get("class")
    if c == "traffic_light":
        return 1.0          # red light governing the ego lane -> must stop
    if c == "stop_sign":
        return 0.8          # stop sign governing the ego lane
    return 0.0


def visible_marks(boxes, z_sign=-1.0, classes=VLM_MARK_CLASSES):
    """Return [(id, u, v, depth, class)] for kept, camera-visible boxes whose class
    is in `classes`. Defaults to VLM_MARK_CLASSES (car/walker) so signals — whose
    GT position is a road-level trigger, not the visible fixture — are NOT drawn."""
    out = []
    for b in boxes:
        if not keep_box(b):
            continue
        if classes is not None and b.get("class") not in classes:
            continue
        proj = project_ego_to_image(b["position"], z_sign=z_sign)
        if proj is None:
            continue
        u, v, depth = proj
        if 0 <= u < CAMERA_WIDTH and 0 <= v < CAMERA_HEIGHT:
            out.append((len(out), u, v, depth, b.get("class")))
    return out


def draw_marks(img, marks, color=(0, 255, 0), txt_color=(0, 0, 255)):
    """marks: list whose first 3 elements are (id, u, v). Draws a labeled dot."""
    import cv2
    out = img.copy()
    for m in marks:
        oid, u, v = m[0], m[1], m[2]
        u, v = int(round(u)), int(round(v))
        cv2.circle(out, (u, v), 6, color, -1)
        cv2.putText(out, str(oid), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, txt_color, 2, cv2.LINE_AA)
    return out
