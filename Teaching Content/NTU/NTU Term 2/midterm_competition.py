import signal
import time

import cv2
import numpy as np
from ugot import ugot

# --- CONSTANTS TO CHANGE ---
# Robot connection
ROBOT_IP = "192.168.1.204"

# Arm constants (used by pickup() at the start and putdown() at the end)
ARM_LOWERED_POSE = (0, -20, -40)  # Joint angles to grab / release the object
ARM_CARRY_POSE = (0, 45, 45)  # Joint angles to carry the object
ARM_MOVE_TIME_MS = 500  # How long each arm move takes
PICKUP_DELAY = 1  # Seconds to wait after each step of pickup()
PUTDOWN_DELAY = 1  # Seconds to wait after each step of putdown()

# Line following constants
LINE_SPEED = 30
LINE_TURN_GAIN = 0.3  # Turn speed per pixel of line offset
LINE_MISSED_FRAMES_TO_STOP = 3  # Frames in a row with no line before switching

# Face approach constants
TARGET_FACE = "Ryan"
FACE_FWD_SPEED = 20
FACE_STRAFE_SPEED = 20
FACE_CENTER_TOLERANCE_PX = 10  # How far the face can be from center (px each side)
FACE_STOP_HEIGHT_PX = 250  # Face box height (px) that counts as close enough
FACE_MISSED_FRAMES_TO_STOP = 3  # Frames in a row without the face before stopping

got = ugot.UGOT()
got.initialize(ROBOT_IP)

got.load_models(["line_recognition", "face_recognition"])
got.set_track_recognition_line(0)
got.open_camera()


def pickup():
    got.mechanical_joint_control(*ARM_LOWERED_POSE, ARM_MOVE_TIME_MS)
    time.sleep(PICKUP_DELAY)
    got.mechanical_clamp_close()
    time.sleep(PICKUP_DELAY)
    got.mechanical_joint_control(*ARM_CARRY_POSE, ARM_MOVE_TIME_MS)
    time.sleep(PICKUP_DELAY)


def putdown():
    got.mechanical_joint_control(*ARM_LOWERED_POSE, ARM_MOVE_TIME_MS)
    time.sleep(PUTDOWN_DELAY)
    got.mechanical_clamp_release()
    time.sleep(PUTDOWN_DELAY)
    got.mechanical_joint_control(*ARM_CARRY_POSE, ARM_MOVE_TIME_MS)
    time.sleep(PUTDOWN_DELAY)


def line_follow(speed=LINE_SPEED, turn_gain=LINE_TURN_GAIN):
    offset, line_type, _, _ = got.get_single_track_total_info()

    # No line this frame: send no command so the robot keeps its last motion
    if line_type == 0:
        return line_type

    got.mecanum_move_xyz(x_speed=0, y_speed=int(speed), z_speed=int(turn_gain * offset))

    return line_type


def face_approach(
    target_face=TARGET_FACE,
    fwd_speed=FACE_FWD_SPEED,
    strafe_speed=FACE_STRAFE_SPEED,
    center_tolerance=FACE_CENTER_TOLERANCE_PX,
    stop_height=FACE_STOP_HEIGHT_PX,
):
    """Returns "arrived", "approaching", or "missed" (target face not in view)."""
    faces = got.get_face_recognition_total_info()

    if faces:
        face_name = faces[0][0]
        if face_name == target_face:
            c_x = faces[0][1]  # Horizontal center of the face in the frame (0-640 px)
            h = faces[0][3]  # Height of the face bounding box (proxy for distance)
            if h < stop_height:
                if c_x < 320 - center_tolerance:
                    # Face is too far LEFT - strafe left while moving forward
                    got.mecanum_move_xyz(
                        x_speed=-int(strafe_speed), y_speed=int(fwd_speed), z_speed=0
                    )
                elif c_x > 320 + center_tolerance:
                    # Face is too far RIGHT - strafe right while moving forward
                    got.mecanum_move_xyz(
                        x_speed=int(strafe_speed), y_speed=int(fwd_speed), z_speed=0
                    )
                else:
                    # Face is centered but still small (far away) — move straight forward
                    got.mecanum_move_xyz(x_speed=0, y_speed=int(fwd_speed), z_speed=0)
                return "approaching"
            else:
                # Face is large enough (close) - we've arrived. Centering isn't
                # checked here, so an off-center face also counts

                got.mecanum_stop()
                print(f"Reached {target_face}!")
                return "arrived"

    # Target face not in view this frame: send no command so the robot keeps
    # its last motion; main() stops it after too many misses in a row
    return "missed"


def main():
    state = "line following"
    missed_frames = 0

    # reset arm
    got.mechanical_joint_control(*ARM_CARRY_POSE, ARM_MOVE_TIME_MS)
    got.mechanical_clamp_release()
    time.sleep(1)

    pickup()

    while True:
        frame = got.read_camera_data()
        if not frame:
            print("Failed to grab frame")
            break

        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        cv2.imshow("Robot Feed", data)

        if state == "line following":
            line_type = line_follow(speed=LINE_SPEED, turn_gain=LINE_TURN_GAIN)
            if line_type == 0:
                missed_frames += 1
            else:
                missed_frames = 0

            if missed_frames >= LINE_MISSED_FRAMES_TO_STOP:
                print("no line")
                got.mecanum_stop()
                state = "face finding"
                missed_frames = 0
                print("Looking for face...")

        elif state == "face finding":
            result = face_approach(
                target_face=TARGET_FACE,
                fwd_speed=FACE_FWD_SPEED,
                strafe_speed=FACE_STRAFE_SPEED,
                center_tolerance=FACE_CENTER_TOLERANCE_PX,
                stop_height=FACE_STOP_HEIGHT_PX,
            )
            if result == "missed":
                missed_frames += 1
                if missed_frames >= FACE_MISSED_FRAMES_TO_STOP:
                    got.mecanum_stop()
            else:
                missed_frames = 0

            if result == "arrived":
                putdown()
                got.mecanum_translate_speed_times(angle=-90, speed=60, times=75, unit=1)
                got.mecanum_move_speed_times(direction=1, speed=60, times=360, unit=1)
                break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    # The Stop button in config_gui.py sends Ctrl+Break on Windows; treat it like Ctrl+C
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    try:
        main()
    except KeyboardInterrupt:
        print("Stopped")
    finally:
        # However the run ends, stop the wheels: otherwise the robot keeps its last motion
        cv2.destroyAllWindows()
        got.mecanum_stop()
