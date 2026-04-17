"""
=============================================================
PART 1 — MediaPipe Hello World  (MediaPipe 0.10.x+, no legacy deps)
=============================================================
Goal: Get MediaPipe running, understand the 21-landmark hand
model, and see raw data that will later drive game controls.

Install dependencies (run once in your terminal):
    pip install mediapipe opencv-python

Run:
    python part1_mediapipe_hello.py

Controls (while the OpenCV window is focused):
    Q  — quit
    L  — toggle landmark index labels on/off
    D  — toggle the data panel (landmark coordinates) on/off
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import time

# ---------------------------------------------------------------------------
# Download the hand landmarker model if not already present.
# MediaPipe 0.10.x requires an explicit .task model file.
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading hand landmarker model to '{MODEL_PATH}'...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.\n")

# ---------------------------------------------------------------------------
# Landmark reference
# ---------------------------------------------------------------------------
# MediaPipe gives you 21 landmarks per hand, each with (x, y, z):
#   x, y  – normalised to [0, 1] relative to image width/height
#   z     – depth relative to the wrist (negative = closer to camera)
#
LANDMARK_NAMES = {
    0:  "WRIST",
    1:  "THUMB_CMC",   2:  "THUMB_MCP",   3:  "THUMB_IP",    4:  "THUMB_TIP",
    5:  "INDEX_MCP",   6:  "INDEX_PIP",   7:  "INDEX_DIP",   8:  "INDEX_TIP",
    9:  "MIDDLE_MCP",  10: "MIDDLE_PIP",  11: "MIDDLE_DIP",  12: "MIDDLE_TIP",
    13: "RING_MCP",    14: "RING_PIP",    15: "RING_DIP",    16: "RING_TIP",
    17: "PINKY_MCP",   18: "PINKY_PIP",   19: "PINKY_DIP",   20: "PINKY_TIP",
}

FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_BASES = [2, 5,  9, 13, 17]

# The 21 connections that make up the hand skeleton, as (start_idx, end_idx) pairs.
# This is what mp_hands_legacy.HAND_CONNECTIONS used to provide.
HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),    # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),    # index
    (0, 9),  (9, 10), (10, 11),(11, 12),  # middle
    (0, 13), (13, 14),(14, 15),(15, 16),  # ring
    (0, 17), (17, 18),(18, 19),(19, 20),  # pinky
    (5, 9),  (9, 13), (13, 17),           # palm knuckle bar
]

# ---------------------------------------------------------------------------
# Detector setup
# ---------------------------------------------------------------------------
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,   # we call detect() each frame
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# Drawing helpers — pure OpenCV, no mediapipe.python dependency
# ---------------------------------------------------------------------------

def lm_px(lm, img_w, img_h):
    """Convert a normalised landmark to pixel coords."""
    return (int(lm.x * img_w), int(lm.y * img_h))


def draw_hand(image, landmark_list, img_w, img_h):
    """Draw the hand skeleton (connections then joints) onto image."""
    # 1. Connections
    for (a, b) in HAND_CONNECTIONS:
        pt1 = lm_px(landmark_list[a], img_w, img_h)
        pt2 = lm_px(landmark_list[b], img_w, img_h)
        cv2.line(image, pt1, pt2, (0, 200, 255), 2, cv2.LINE_AA)

    # 2. Joints — fingertips bigger and brighter
    for idx, lm in enumerate(landmark_list):
        pt     = lm_px(lm, img_w, img_h)
        is_tip = idx in FINGER_TIPS
        color  = (0, 255, 150) if is_tip else (255, 255, 255)
        radius = 7 if is_tip else 4
        cv2.circle(image, pt, radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, pt, radius, (0, 0, 0), 1, cv2.LINE_AA)   # outline


def draw_landmark_indices(image, landmark_list, img_w, img_h):
    """Draw the landmark index number next to each point."""
    for idx, lm in enumerate(landmark_list):
        cx, cy = lm_px(lm, img_w, img_h)
        cv2.putText(
            image, str(idx), (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA,
        )


def draw_data_panel(image, landmark_list, hand_label, img_w, img_h):
    """
    Draw a live coordinate table for key landmarks.
    Helps you see what the raw numbers look like in real time —
    important context before we write gesture logic in Part 2.
    """
    SHOWN   = [0, 4, 8, 12, 16, 20]   # wrist + all fingertips
    panel_x = 10 if hand_label == "Left" else img_w // 2 + 10
    panel_y = 30
    line_h  = 16

    cv2.rectangle(
        image,
        (panel_x - 4, panel_y - 18),
        (panel_x + 230, panel_y + len(SHOWN) * line_h + 4),
        (0, 0, 0), -1,
    )
    cv2.putText(
        image, f"{hand_label} hand", (panel_x, panel_y - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA,
    )
    for i, idx in enumerate(SHOWN):
        lm   = landmark_list[idx]
        name = LANDMARK_NAMES[idx]
        text = f"{idx:>2} {name:<12}  x={lm.x:.2f} y={lm.y:.2f}"
        cv2.putText(
            image, text, (panel_x, panel_y + (i + 1) * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA,
        )

# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------
class FPSCounter:
    def __init__(self, smoothing=20):
        self._times     = []
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

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_counter  = FPSCounter()
    show_indices = True
    show_data    = True

    print("=== Part 1: MediaPipe Hello World (0.10.x+) ===")
    print("  Q — quit")
    print("  L — toggle landmark index labels")
    print("  D — toggle data panel")
    print("================================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: empty frame, retrying...")
            continue

        frame = cv2.flip(frame, 1)   # mirror — feels more natural
        img_h, img_w = frame.shape[:2]

        # Wrap numpy array in mp.Image for the Tasks API
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        result = detector.detect(mp_image)

        # result.hand_landmarks — list of 21-landmark lists
        # result.handedness     — parallel list of classification results
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
            ):
                hand_label = handedness[0].display_name  # "Left" or "Right"

                draw_hand(frame, hand_landmarks, img_w, img_h)

                if show_indices:
                    draw_landmark_indices(frame, hand_landmarks, img_w, img_h)

                if show_data:
                    draw_data_panel(frame, hand_landmarks, hand_label, img_w, img_h)

                # Terminal one-liner — watch these numbers as you move your hand
                wrist     = hand_landmarks[0]
                index_tip = hand_landmarks[8]
                print(
                    f"[{hand_label}]  "
                    f"WRIST=({wrist.x:.3f}, {wrist.y:.3f})  "
                    f"INDEX_TIP=({index_tip.x:.3f}, {index_tip.y:.3f})",
                    end="\r",
                )
        else:
            print("No hands detected...                              ", end="\r")

        # HUD
        fps_counter.tick()
        cv2.putText(
            frame, f"FPS: {fps_counter.get_fps():.0f}", (img_w - 90, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, "[Q] quit   [L] landmarks   [D] data",
            (10, img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )

        cv2.imshow("Part 1 - MediaPipe Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            show_indices = not show_indices
            print(f"\nLandmark indices: {'ON' if show_indices else 'OFF'}")
        elif key == ord("d"):
            show_data = not show_data
            print(f"\nData panel: {'ON' if show_data else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nDone.")


if __name__ == "__main__":
    main()