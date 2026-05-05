"""
Robot Controller using MediaPipe Tasks API (compatible with mediapipe 0.10+)
-----------------------------------------------------------------------------
Hold your hand up to the webcam to control the robot:
    1 finger  -> forward()
    2 fingers -> left()
    3 fingers -> right()
    4 fingers -> backwards()
    0 / other -> stop()

Press 'q' to quit.

Requirements:
    pip install mediapipe opencv-python
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import urllib.request
import os

# Download the hand landmark model if not present
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.")

# Fingertip and PIP (middle knuckle) landmark indices
# Order: thumb, index, middle, ring, pinky
# FINGERTIP_IDS = [4, 8, 12, 16, 20]  # includes thumb (index 0)
# FINGER_PIP_IDS = [3, 6, 10, 14, 18]  # includes thumb (index 0)
FINGERTIP_IDS = [8, 12, 16, 20]  # index, middle, ring, pinky
FINGER_PIP_IDS = [6, 10, 14, 18]  # index, middle, ring, pinky


def count_fingers(landmarks: list[NormalizedLandmark], is_right_hand: bool) -> int:
    fingers_up = []

    # Thumb (commented out — re-enable and restore FINGERTIP_IDS/FINGER_PIP_IDS above to use)
    # tip, pip = landmarks[4], landmarks[3]
    # fingers_up.append(tip.x < pip.x if is_right_hand else tip.x > pip.x)

    # Index-pinky: fingertip y above PIP y means finger is extended
    for tip_id, pip_id in zip(FINGERTIP_IDS, FINGER_PIP_IDS):
        fingers_up.append(landmarks[tip_id].y < landmarks[pip_id].y)

    return sum(fingers_up)


def draw_landmarks(frame, landmarks, img_w, img_h):
    """Draw hand skeleton on the frame."""
    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    ]
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 220, 0), -1)


# ---------------------------------------------------------------------------
# Robot commands — replace these with your real robot API calls
# ---------------------------------------------------------------------------
def forward():
    print("Robot: FORWARD")


def left():
    print("Robot: LEFT")


def right():
    print("Robot: RIGHT")


def backwards():
    print("Robot: BACKWARDS")


def stop():
    print("Robot: STOP")


FINGER_COMMANDS = {
    1: (forward, "FORWARD"),
    2: (left, "LEFT"),
    3: (right, "RIGHT"),
    4: (backwards, "BACKWARDS"),
}
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

        label = "No hand detected"
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].display_name
            is_right = handedness == "Right"

            draw_landmarks(frame, landmarks, img_w, img_h)
            count = count_fingers(landmarks, is_right)

            if count in FINGER_COMMANDS:
                cmd_fn, cmd_name = FINGER_COMMANDS[count]
                cmd_fn()
                label = f"{count} finger(s): {cmd_name}"
            else:
                stop()
                label = f"{count} finger(s): STOP"

        cv2.putText(frame, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 200, 0), 4)
        cv2.imshow("Finger Counter  [q to quit]", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
