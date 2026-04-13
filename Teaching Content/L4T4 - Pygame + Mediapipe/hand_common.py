"""
=============================================================
hand_common.py — Shared foundation for all parts
=============================================================
This module is imported by every part in the series.
It contains everything that never changes:
  - Model download + detector setup
  - Landmark index constants
  - Hand skeleton connections
  - Pure-OpenCV drawing helpers
  - FPSCounter

Nothing in here should ever need editing as the project grows.
If you find yourself wanting to change something here, it
probably belongs in a higher-level file instead.
=============================================================
"""

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import time

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print(f"[hand_common] Downloading model to '{MODEL_PATH}'...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[hand_common] Download complete.\n")

# ---------------------------------------------------------------------------
# Landmark index constants
# ---------------------------------------------------------------------------
# 21 landmarks per hand, (x, y, z) each, normalised 0-1.
# x=0 left edge, x=1 right edge, y=0 top, y=1 bottom.
# z is depth relative to wrist (negative = closer to camera).

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [2, INDEX_MCP, 9, 13, 17]  # base knuckles, one per finger

LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_MCP",
    6: "INDEX_PIP",
    7: "INDEX_DIP",
    8: "INDEX_TIP",
    9: "MIDDLE_MCP",
    10: "MIDDLE_PIP",
    11: "MIDDLE_DIP",
    12: "MIDDLE_TIP",
    13: "RING_MCP",
    14: "RING_PIP",
    15: "RING_DIP",
    16: "RING_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

# ---------------------------------------------------------------------------
# Hand skeleton connections
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def make_detector(num_hands=2):
    """Create and return a HandLandmarker detector."""
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Drawing helpers — pure OpenCV
# ---------------------------------------------------------------------------


def lm_px(lm, img_w, img_h):
    """Normalised landmark -> pixel (x, y) tuple."""
    return (int(lm.x * img_w), int(lm.y * img_h))


def draw_hand(image, landmark_list, img_w, img_h):
    """Draw hand skeleton (bones + joints) onto an OpenCV image."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(
            image,
            lm_px(landmark_list[a], img_w, img_h),
            lm_px(landmark_list[b], img_w, img_h),
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    for idx, lm in enumerate(landmark_list):
        pt = lm_px(lm, img_w, img_h)
        is_tip = idx in FINGER_TIPS
        color = (0, 255, 150) if is_tip else (255, 255, 255)
        radius = 7 if is_tip else 4
        cv2.circle(image, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, pt, radius, (0, 0, 0), 1, cv2.LINE_AA)


def draw_landmark_indices(image, landmark_list, img_w, img_h):
    """Draw the index number next to every landmark point."""
    for idx, lm in enumerate(landmark_list):
        cx, cy = lm_px(lm, img_w, img_h)
        cv2.putText(
            image,
            str(idx),
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def draw_data_panel(image, landmark_list, hand_label, img_w, img_h):
    """Draw a live (x, y) coordinate table for wrist + fingertips."""
    SHOWN = [0, 4, 8, 12, 16, 20]
    panel_x = 10 if hand_label == "Left" else img_w // 2 + 10
    panel_y = 30
    line_h = 16

    cv2.rectangle(
        image,
        (panel_x - 4, panel_y - 18),
        (panel_x + 230, panel_y + len(SHOWN) * line_h + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        f"{hand_label} hand",
        (panel_x, panel_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 180),
        1,
        cv2.LINE_AA,
    )
    for i, idx in enumerate(SHOWN):
        lm = landmark_list[idx]
        name = LANDMARK_NAMES[idx]
        text = f"{idx:>2} {name:<12}  x={lm.x:.2f} y={lm.y:.2f}"
        cv2.putText(
            image,
            text,
            (panel_x, panel_y + (i + 1) * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------
class FPSCounter:
    def __init__(self, smoothing=20):
        self._times = []
        self._smoothing = smoothing

    def tick(self):
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._smoothing:
            self._times.pop(0)

    def get_fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
