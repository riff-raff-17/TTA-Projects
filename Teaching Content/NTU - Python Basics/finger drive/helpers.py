"""
helpers.py — Shared utilities for robot finger control.
Contains: direction logic, overlay drawing, and model download.
"""

import cv2
import urllib.request
import os

# ── Constants ─────────────────────────────────────────────────────────────────

DEADZONE = 0.15  # 15% on each side → 30% total dead band

DIRECTION_COLORS = {
    "forward":  (0, 200, 0),
    "backward": (0, 0, 200),
    "left":     (200, 150, 0),
    "right":    (0, 150, 200),
    "stop":     (160, 160, 160),
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# ── Direction logic ────────────────────────────────────────────────────────────

def get_direction(x_norm: float, y_norm: float) -> str:
    """
    Map a normalised (0-1) finger position to a robot command.
    x_norm: 0 = left edge, 1 = right edge
    y_norm: 0 = top edge,  1 = bottom edge  (MediaPipe convention)
    Returns one of: 'forward', 'backward', 'left', 'right', 'stop'
    """
    dx = x_norm - 0.5   # signed distance from centre (-0.5 … +0.5)
    dy = y_norm - 0.5

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return "stop"

    # Whichever axis has the larger deviation wins
    if abs(dy) >= abs(dx):
        return "backward" if dy > 0 else "forward"
    else:
        return "right" if dx > 0 else "left"

# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_overlay(frame, direction: str, tip_px):
    h, w = frame.shape[:2]

    # Deadzone rectangle
    dz_x1 = int((0.5 - DEADZONE) * w)
    dz_x2 = int((0.5 + DEADZONE) * w)
    dz_y1 = int((0.5 - DEADZONE) * h)
    dz_y2 = int((0.5 + DEADZONE) * h)
    cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (220, 220, 220), 2)
    cv2.putText(frame, "DEAD", (dz_x1 + 4, (dz_y1 + dz_y2) // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "ZONE", (dz_x1 + 4, (dz_y1 + dz_y2) // 2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Fingertip dot
    color = DIRECTION_COLORS[direction]
    cv2.circle(frame, tip_px, 14, color, -1)
    cv2.circle(frame, tip_px, 14, (255, 255, 255), 2)

    # Direction label
    cv2.putText(frame, direction.upper(), (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Crosshair at centre
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

    # Instructions
    cv2.putText(frame, "q = quit", (w - 100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

# ── Model download ─────────────────────────────────────────────────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.\n")
