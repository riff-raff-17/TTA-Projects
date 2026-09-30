"""
PART 3 — Pinch detection (gripper logic)
=========================================
Goal: use the hand skeleton from Part 2 to detect a pinch gesture and
      print OPEN / CLOSED to the terminal. This is the gripper logic in isolation.

New concepts vs Part 2:
  - Accessing specific landmark indices (thumb tip = 4, index tip = 8)
  - Computing a normalised distance between two landmarks
  - Thresholding that distance into a binary state

What you'll see: webcam with a coloured line between thumb and index tip.
                 Terminal prints OPEN or CLOSED each time the state changes.
Press Q to quit.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    print("Done.")

# Landmark indices we care about
THUMB_TIP = 4
INDEX_TIP = 8
WRIST     = 0
MID_MCP   = 9   # middle finger knuckle — used as a stable size reference

def lm_dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def detect_pinch(landmarks):
    """Returns 1.0 (open) or 0.0 (closed/pinching)."""
    hand_ref   = lm_dist(landmarks[WRIST], landmarks[MID_MCP]) + 1e-6
    pinch_dist = lm_dist(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    pinch_ratio = pinch_dist / hand_ref
    THRESHOLD = 0.4
    return 1.0 if pinch_ratio > THRESHOLD else 0.0

def draw_pinch(frame, landmarks, img_w, img_h, gripper_open):
    color = (0, 200, 255) if gripper_open > 0.5 else (0, 60, 255)
    t = landmarks[THUMB_TIP]
    i = landmarks[INDEX_TIP]
    pt_t = (int(t.x * img_w), int(t.y * img_h))
    pt_i = (int(i.x * img_w), int(i.y * img_h))
    cv2.circle(frame, pt_t, 10, color, -1)
    cv2.circle(frame, pt_i, 10, color, -1)
    cv2.line(frame, pt_t, pt_i, color, 3)
    state = "OPEN" if gripper_open > 0.5 else "CLOSED"
    cv2.putText(frame, f"Gripper: {state}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
hand_options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

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
    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        gripper = detect_pinch(lms)
        draw_pinch(frame, lms, img_w, img_h, gripper)

        # Print to terminal only when state changes
        if gripper != prev_gripper:
            print(f"Gripper: {'OPEN' if gripper > 0.5 else 'CLOSED'}")
            prev_gripper = gripper
    else:
        cv2.putText(frame, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    cv2.imshow("Part 3 — Pinch Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()