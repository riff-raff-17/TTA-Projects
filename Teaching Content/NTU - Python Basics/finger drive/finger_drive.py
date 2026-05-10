"""
Robot finger control using MediaPipe hand tracking.
Move your index finger to control the robot:
  - Center (deadzone): STOP
  - Up:    FORWARD
  - Down:  BACKWARD
  - Left:  LEFT
  - Right: RIGHT

Press 'q' to quit.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import urllib.request
import os

# ── Robot helper functions (replace with real hardware calls) ──────────────────


def robot_forward():
    print("FORWARD")


def robot_backward():
    print("BACKWARD")


def robot_left():
    print("LEFT")


def robot_right():
    print("RIGHT")


def robot_stop():
    print("STOP")


# ── Direction logic ────────────────────────────────────────────────────────────

# Deadzone: fraction of screen width/height around the center
DEADZONE = 0.15  # 15% on each side → 30% total dead band


def get_direction(x_norm: float, y_norm: float) -> str:
    """
    Map a normalised (0-1) finger position to a robot command.
    x_norm: 0 = left edge, 1 = right edge
    y_norm: 0 = top edge,  1 = bottom edge  (MediaPipe convention)
    Returns one of: 'forward', 'backward', 'left', 'right', 'stop'
    """
    dx = x_norm - 0.5  # signed distance from centre (-0.5 … +0.5)
    dy = y_norm - 0.5

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return "stop"

    # Whichever axis has the larger deviation wins
    if abs(dy) >= abs(dx):
        return "backward" if dy > 0 else "forward"
    else:
        return "right" if dx > 0 else "left"


def dispatch(direction: str):
    """Call the appropriate robot helper."""
    actions = {
        "forward": robot_forward,
        "backward": robot_backward,
        "left": robot_left,
        "right": robot_right,
        "stop": robot_stop,
    }
    actions[direction]()


# ── Drawing helpers ────────────────────────────────────────────────────────────

DIRECTION_COLORS = {
    "forward": (0, 200, 0),
    "backward": (0, 0, 200),
    "left": (200, 150, 0),
    "right": (0, 150, 200),
    "stop": (160, 160, 160),
}


def draw_overlay(frame, direction: str, tip_px):
    h, w = frame.shape[:2]

    # Deadzone rectangle
    dz_x1 = int((0.5 - DEADZONE) * w)
    dz_x2 = int((0.5 + DEADZONE) * w)
    dz_y1 = int((0.5 - DEADZONE) * h)
    dz_y2 = int((0.5 + DEADZONE) * h)
    cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (220, 220, 220), 2)
    cv2.putText(
        frame,
        "DEAD",
        (dz_x1 + 4, (dz_y1 + dz_y2) // 2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )
    cv2.putText(
        frame,
        "ZONE",
        (dz_x1 + 4, (dz_y1 + dz_y2) // 2 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Fingertip dot
    color = DIRECTION_COLORS[direction]
    cv2.circle(frame, tip_px, 14, color, -1)
    cv2.circle(frame, tip_px, 14, (255, 255, 255), 2)

    # Direction label
    label = direction.upper()
    cv2.putText(
        frame, label, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA
    )

    # Crosshair at centre
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

    # Instructions
    cv2.putText(
        frame,
        "q = quit",
        (w - 100, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

# Index fingertip landmark index (same number as before, just accessed differently)
INDEX_FINGER_TIP = 8

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.\n")


def main():
    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_direction = None

    print("Hand tracking started. Show your index finger to the camera.")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror so left/right feel natural
        h, w = frame.shape[:2]

        # New API uses mp.Image instead of raw numpy arrays
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        direction = "stop"
        tip_px = (w // 2, h // 2)  # default dot position

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            tip = lm[INDEX_FINGER_TIP]  # NormalizedLandmark with .x, .y
            tip_px = (int(tip.x * w), int(tip.y * h))
            direction = get_direction(tip.x, tip.y)

        draw_overlay(frame, direction, tip_px)
        cv2.imshow("Robot Finger Control", frame)

        # Only call the robot when the command changes (avoids spam)
        if direction != last_direction:
            dispatch(direction)
            last_direction = direction

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
