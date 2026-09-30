"""
robot_finger_control.py — Main entry point.
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

from helpers import get_direction, draw_overlay, download_model, MODEL_PATH

from ugot import ugot


# --- Robot functions ---
def robot_forward():
    print("FORWARD")
    got.mecanum_move_speed(direction=0, speed=20)


def robot_backward():
    print("BACKWARD")
    got.mecanum_move_speed(direction=1, speed=20)


def robot_left():
    print("LEFT")
    got.mecanum_turn_speed(turn=2, speed=45)


def robot_right():
    print("RIGHT")
    got.mecanum_turn_speed(turn=3, speed=45)


def robot_stop():
    print("STOP")
    got.mecanum_stop()


def dispatch(direction):
    """Call the appropriate robot function."""
    actions = {
        "forward": robot_forward,
        "backward": robot_backward,
        "left": robot_left,
        "right": robot_right,
        "stop": robot_stop,
    }
    actions[direction]()


# --- Main loop ---
INDEX_FINGER_TIP = 8

got = ugot.UGOT()
got.initialize("192.168.1.91")

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

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        direction = "stop"
        tip_px = (w // 2, h // 2)  # default dot position

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            tip = lm[INDEX_FINGER_TIP]
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
    print("Stopped!")


if __name__ == "__main__":
    main()
