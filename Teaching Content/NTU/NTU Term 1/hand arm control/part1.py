"""
Part 1 — Hand Detection + Centroid Dot
---------------------------------------
Detects your hand and draws:
  - The full hand skeleton
  - A yellow dot at the hand's centroid
  - A crosshair at the screen center

This confirms your camera and MediaPipe are working before adding any logic.

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
# Hand connections for drawing the skeleton
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def hand_center(landmarks):
    """Return the (x, y) centroid of all 21 landmarks in normalised [0,1] coords."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return float(np.mean(xs)), float(np.mean(ys))


def draw_hand(frame, landmarks, img_w, img_h):
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 80), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 200, 80), -1)


def draw_crosshair(frame, img_w, img_h):
    cx, cy = img_w // 2, img_h // 2
    cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (80, 80, 80), 1)
    cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (80, 80, 80), 1)
    cv2.circle(frame, (cx, cy), 6, (80, 80, 80), 1)
    cv2.putText(frame, "Center", (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)


def draw_centroid(frame, cx_norm, cy_norm, img_w, img_h):
    px = int(cx_norm * img_w)
    py = int(cy_norm * img_h)
    # Line from screen center to hand centroid
    cv2.line(frame, (img_w // 2, img_h // 2), (px, py), (255, 200, 0), 1)
    # Centroid dot
    cv2.circle(frame, (px, py), 10, (255, 200, 0), -1)
    cv2.putText(frame, f"Hand ({cx_norm:.2f}, {cy_norm:.2f})",
                (px + 12, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)


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
            draw_hand(frame, landmarks, img_w, img_h)

            cx, cy = hand_center(landmarks)
            draw_centroid(frame, cx, cy, img_w, img_h)

            label = "Hand detected"
        else:
            label = "No hand detected - show your hand"

        cv2.putText(frame, label, (20, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Part 1: Hand Detection [q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()