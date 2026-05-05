"""
Part 3 — Gripper (Pinch Detection)
------------------------------------
Builds on Part 2. Adds:
  - Distance-invariant pinch detection (thumb tip ↔ index tip)
  - OPEN / CLOSED shown on HUD and via colour change on fingertips
  - Gripper state added to the status bar

Still no robot commands — just verify all three values look right.

Press 'q' to quit.

Requirements:
    pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DEAD_ZONE = 0.10
GRIPPER_THRESHOLD = 0.4   # pinch_dist / hand_ref; below this = closed

# ---------------------------------------------------------------------------
# Landmark indices
# ---------------------------------------------------------------------------
WRIST      = 0
MIDDLE_MCP = 9
THUMB_TIP  = 4
INDEX_TIP  = 8

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def hand_centre(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return float(np.mean(xs)), float(np.mean(ys))


def position_to_angles(cx: float, cy: float) -> tuple[float, float]:
    dx = cx - 0.5
    dy = cy - 0.5

    def apply_dead_zone(v, dz):
        if abs(v) < dz:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - dz) / (0.5 - dz)

    pan_deg  = float(np.clip(-apply_dead_zone(dx, DEAD_ZONE) * 90.0, -90.0, 90.0))
    lift_deg = float(np.clip(-apply_dead_zone(dy, DEAD_ZONE) * 90.0, -90.0, 90.0))
    return pan_deg, lift_deg


def lm_dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def hand_to_gripper(landmarks) -> float:
    """
    Returns 1.0 (open) or 0.0 (closed).
    Normalised by wrist-to-middle-MCP span so distance from camera doesn't matter.
    """
    hand_ref   = lm_dist(landmarks[WRIST], landmarks[MIDDLE_MCP]) + 1e-6
    pinch_dist = lm_dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    return 1.0 if (pinch_dist / hand_ref) > GRIPPER_THRESHOLD else 0.0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_hand(frame, landmarks, img_w, img_h, gripper_open: float):
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]

    # Skeleton
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 80), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 200, 80), -1)

    # Pinch tips — cyan = open, red = closed
    pinch_color = (0, 200, 255) if gripper_open > 0.5 else (0, 80, 255)
    for idx in [THUMB_TIP, INDEX_TIP]:
        cv2.circle(frame, pts[idx], 10, pinch_color, -1)
    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], pinch_color, 2)


def draw_crosshair(frame, img_w, img_h):
    cx, cy = img_w // 2, img_h // 2
    dz_radius = int(DEAD_ZONE * img_w / 2)
    cv2.circle(frame, (cx, cy), dz_radius, (60, 60, 60), 1)
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (80, 80, 80), 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (80, 80, 80), 1)


def draw_centroid(frame, cx_norm, cy_norm, img_w, img_h):
    px, py = int(cx_norm * img_w), int(cy_norm * img_h)
    cv2.line(frame, (img_w // 2, img_h // 2), (px, py), (255, 200, 0), 1)
    cv2.circle(frame, (px, py), 10, (255, 200, 0), -1)


def draw_hud(frame, pan, lift, gripper):
    lines = [
        f"Pan  (L/R): {pan:+6.1f} deg",
        f"Lift (U/D): {lift:+6.1f} deg",
        f"Gripper:    {'OPEN' if gripper > 0.5 else 'CLOSED'}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 40 + i * 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)


def draw_bars(frame, pan, lift, img_w, img_h):
    bar_w, bar_h, margin = 200, 14, 20

    def draw_bar(label, value, y0):
        x0 = margin
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
        frac = (value + 90.0) / 180.0
        cv2.rectangle(frame, (x0, y0),
                      (x0 + int(np.clip(frac, 0, 1) * bar_w), y0 + bar_h),
                      (0, 180, 255), -1)
        mid_x = x0 + bar_w // 2
        cv2.line(frame, (mid_x, y0), (mid_x, y0 + bar_h), (200, 200, 200), 1)
        cv2.putText(frame, label, (x0, y0 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    draw_bar("Lift", lift, img_h - margin - bar_h * 2 - 18)
    draw_bar("Pan",  pan,  img_h - margin - bar_h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        result = landmarker.detect(mp_image)
        draw_crosshair(frame, img_w, img_h)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            gripper = hand_to_gripper(landmarks)
            draw_hand(frame, landmarks, img_w, img_h, gripper)

            cx, cy = hand_centre(landmarks)
            draw_centroid(frame, cx, cy, img_w, img_h)

            pan, lift = position_to_angles(cx, cy)
            draw_hud(frame, pan, lift, gripper)
            draw_bars(frame, pan, lift, img_w, img_h)

            label = f"Pan {pan:+.0f}deg  Lift {lift:+.0f}deg  {'OPEN' if gripper > 0.5 else 'CLOSED'}"
        else:
            label = "No hand detected — show your hand"

        cv2.putText(frame, label, (20, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Part 3: Gripper Detection  [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()