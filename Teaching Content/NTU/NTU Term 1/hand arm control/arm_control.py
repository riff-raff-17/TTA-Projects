"""
Robot Arm Controller — Hand Position Mapping
---------------------------------------------
Hold your hand in front of the webcam to control the robot arm:

    Hand position (relative to screen centre):
        Up    → Lift  +90°   (arm raises)
        Down  → Lift  -90°   (arm lowers)
        Left  → Pan   +90°   (arm pans left)
        Right → Pan   -90°   (arm pans right)

    Pinch (thumb tip ↔ index tip close together) → Gripper CLOSED
    No pinch                                      → Gripper OPEN

Press 'q' to quit.
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
# Hand landmark indices
# ---------------------------------------------------------------------------
WRIST = 0
MIDDLE_MCP = 9  # stable reference for hand scale
THUMB_TIP = 4
INDEX_TIP = 8

# Fingertip / PIP pairs for finger counting (index → pinky, thumb excluded)
FINGERTIP_IDS = [8, 12, 16, 20]
FINGER_PIP_IDS = [6, 10, 14, 18]

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
# Dead-zone: fraction of half-screen within which output is zero
DEAD_ZONE = 0.10

# Gripper pinch threshold (pinch_dist / hand_ref)
GRIPPER_THRESHOLD = 0.4

# EMA smoothing factor (lower = smoother but laggier)
EMA_ALPHA = 0.20

# Only send a new robot command when the value changes by more than this
PAN_THRESHOLD = 1.0  # degrees
LIFT_THRESHOLD = 1.0  # degrees
GRIPPER_THRESHOLD_CMD = 0.5  # 0–1


# ---------------------------------------------------------------------------
# Robot command stubs — replace with your real robot API
# ---------------------------------------------------------------------------
def set_pan(angle: float):
    """Pan joint. Left = +90°, Right = -90°."""
    print(f"Robot: PAN  -> {angle:6.1f}°")


def set_lift(angle: float):
    """Lift joint. Up = +90°, Down = -90°."""
    print(f"Robot: LIFT -> {angle:6.1f}°")


def set_gripper(open_pct: float):
    """Gripper. 0.0 = fully closed, 1.0 = fully open."""
    print(f"Robot: GRIPPER -> {'OPEN' if open_pct > 0.5 else 'CLOSED'}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def lm_dist(a, b):
    """Euclidean distance between two landmarks in normalised coords."""
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def hand_centre(landmarks):
    """
    Return the (x, y) centroid of all 21 hand landmarks in normalised [0,1] coords.
    Using all landmarks gives a stable centroid that isn't thrown off by a single
    extended or curled finger.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return float(np.mean(xs)), float(np.mean(ys))


def position_to_angles(cx: float, cy: float) -> tuple[float, float]:
    """
    Map a normalised hand centre (cx, cy) → (pan_deg, lift_deg).

    Screen coords: cx=0 left, cx=1 right; cy=0 top, cy=1 bottom.
    Desired output:
        cx=0.5 (centre) → pan=0
        cx=0.0 (left)   → pan=+90
        cx=1.0 (right)  → pan=-90
        cy=0.5 (centre) → lift=0
        cy=0.0 (top)    → lift=+90
        cy=1.0 (bottom) → lift=-90
    """
    # Offset from centre, range [-0.5, +0.5]
    dx = cx - 0.5  # positive = hand is to the right
    dy = cy - 0.5  # positive = hand is below centre

    # Apply dead zone (relative to half-screen = 0.5 units)
    def apply_dead_zone(v: float, dz: float) -> float:
        if abs(v) < dz:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        # Rescale so edge of dead zone maps to 0 and ±0.5 maps to ±1
        return sign * (abs(v) - dz) / (0.5 - dz)

    dx_norm = apply_dead_zone(dx, DEAD_ZONE)  # -1 … +1
    dy_norm = apply_dead_zone(dy, DEAD_ZONE)  # -1 … +1

    pan_deg = float(np.clip(-dx_norm * 90.0, -90.0, 90.0))  # right = negative
    lift_deg = float(np.clip(-dy_norm * 90.0, -90.0, 90.0))  # down  = negative

    return pan_deg, lift_deg


def hand_to_gripper(landmarks) -> float:
    """
    Distance-invariant pinch detection.
    Returns 1.0 (open) or 0.0 (closed).
    """
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_TIP]
    wrist = landmarks[WRIST]
    mid_mcp = landmarks[MIDDLE_MCP]

    hand_ref = lm_dist(wrist, mid_mcp) + 1e-6
    pinch_ratio = lm_dist(thumb_tip, index_tip) / hand_ref

    return 1.0 if pinch_ratio > GRIPPER_THRESHOLD else 0.0


# ---------------------------------------------------------------------------
# Smoothing — exponential moving average
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm base
]


def draw_hand(frame, landmarks, img_w, img_h, gripper_open: float):
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
    skeleton_color = (0, 200, 80)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], skeleton_color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, skeleton_color, -1)

    # Highlight pinch tips
    pinch_color = (0, 200, 255) if gripper_open > 0.5 else (0, 80, 255)
    for idx in [THUMB_TIP, INDEX_TIP]:
        cv2.circle(frame, pts[idx], 9, pinch_color, -1)
    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], pinch_color, 2)


def draw_crosshair(frame, img_w, img_h):
    """Draw screen centre reference."""
    cx, cy = img_w // 2, img_h // 2
    color = (80, 80, 80)
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), color, 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), color, 1)
    cv2.circle(frame, (cx, cy), 6, color, 1)


def draw_hand_dot(frame, cx_norm, cy_norm, img_w, img_h):
    """Draw a dot at the hand centroid."""
    px = int(cx_norm * img_w)
    py = int(cy_norm * img_h)
    cv2.circle(frame, (px, py), 8, (255, 200, 0), -1)
    # Line from screen centre to hand
    cv2.line(frame, (img_w // 2, img_h // 2), (px, py), (255, 200, 0), 1)


def draw_hud(frame, pan, lift, gripper):
    lines = [
        f"Pan  (L/R): {pan:6.1f} deg",
        f"Lift (U/D): {lift:6.1f} deg",
        f"Gripper:    {'OPEN' if gripper > 0.5 else 'CLOSED'}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 40 + i * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )


def draw_axis_bars(frame, pan, lift, img_w, img_h):
    """
    Draw two small progress bars (pan and lift) near the screen edges
    so you can see the current output at a glance.
    """
    bar_w = 200
    bar_h = 14
    margin = 20

    def draw_bar(label, value, x0, y0):
        # Background
        cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
        # Fill: value in [-90, 90] → fraction in [0, 1]
        frac = (value + 90.0) / 180.0
        fill_w = int(frac * bar_w)
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + bar_h), (0, 180, 255), -1)
        # Centre mark
        mid_x = x0 + bar_w // 2
        cv2.line(frame, (mid_x, y0), (mid_x, y0 + bar_h), (200, 200, 200), 1)
        cv2.putText(
            frame,
            label,
            (x0, y0 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

    # Pan bar — bottom left
    draw_bar("Pan", pan, margin, img_h - margin - bar_h)
    # Lift bar — bottom left, above pan
    draw_bar("Lift", lift, margin, img_h - margin - bar_h * 2 - 18)


# ---------------------------------------------------------------------------
# Main loop
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

    pan_ema = EMA()
    lift_ema = EMA()

    prev_pan = None
    prev_lift = None
    prev_gripper = None

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

        label = "No hand detected — show your hand"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # --- Gripper ---
            gripper = hand_to_gripper(landmarks)

            # --- Position → angles ---
            cx, cy = hand_centre(landmarks)
            pan_raw, lift_raw = position_to_angles(cx, cy)
            pan = pan_ema.update(pan_raw)
            lift = lift_ema.update(lift_raw)

            # --- Drawing ---
            draw_hand(frame, landmarks, img_w, img_h, gripper)
            draw_hand_dot(frame, cx, cy, img_w, img_h)
            draw_hud(frame, pan, lift, gripper)
            draw_axis_bars(frame, pan, lift, img_w, img_h)

            # --- Send commands only when values change enough ---
            pan_changed = prev_pan is None or abs(pan - prev_pan) > PAN_THRESHOLD
            lift_changed = prev_lift is None or abs(lift - prev_lift) > LIFT_THRESHOLD
            grip_changed = (
                prev_gripper is None
                or abs(gripper - prev_gripper) > GRIPPER_THRESHOLD_CMD
            )

            if pan_changed:
                set_pan(pan)
                prev_pan = pan
            if lift_changed:
                set_lift(lift)
                prev_lift = lift
            if grip_changed:
                set_gripper(gripper)
                prev_gripper = gripper

            label = f"Pan {pan:+.0f}°  Lift {lift:+.0f}°  {'OPEN' if gripper > 0.5 else 'CLOSED'}"

        cv2.putText(
            frame,
            label,
            (20, img_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        cv2.imshow("Hand Position Robot Control  [q to quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
