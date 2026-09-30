import time

import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.1.209")

got.load_models(["line_recognition", "face_recognition"])
got.set_track_recognition_line(0)
got.open_camera()


def pickup():
    got.mechanical_joint_control(0, -10, -20, 500)
    time.sleep(1)
    got.mechanical_clamp_close()
    time.sleep(1)
    got.mechanical_joint_control(0, 45, 45, 500)
    time.sleep(1)


def putdown():
    got.mechanical_joint_control(0, -10, -20, 500)
    time.sleep(1)
    got.mechanical_clamp_release()
    time.sleep(1)
    got.mechanical_joint_control(0, 45, 45, 500)
    time.sleep(1)


def line_follow(mult=0.25, speed=20):
    offset, line_type, _, _ = got.get_single_track_total_info()

    got.mecanum_move_xyz(x_speed=0, y_speed=int(speed), z_speed=int(mult * offset))

    return line_type


def face_approach(
    target_face="Stranger", gap=10, strafe_spd=10, fwd_spd=10, height=150
):
    faces = got.get_face_recognition_total_info()

    if faces:
        face_name = faces[0][0]
        if face_name == target_face:
            c_x = faces[0][1]  # Horizontal center of the face in the frame (0-640 px)
            h = faces[0][3]  # Height of the face bounding box (proxy for distance)
            if h < height:
                if c_x < 320 - gap:
                    # Face is too far LEFT - strafe left while moving forward
                    got.mecanum_move_xyz(
                        x_speed=-strafe_spd, y_speed=fwd_spd, z_speed=0
                    )
                elif c_x > 320 + gap:
                    # Face is too far RIGHT - strafe right while moving forward
                    got.mecanum_move_xyz(x_speed=strafe_spd, y_speed=fwd_spd, z_speed=0)
                else:
                    # Face is centered but still small (far away) — move straight forward
                    got.mecanum_move_xyz(x_speed=0, y_speed=fwd_spd, z_speed=0)
            else:
                # Face is centered AND large enough - we've arrived
                got.mecanum_stop()
                print(f"Reached {target_face}!")
                return True

    else:
        got.mecanum_stop()
        return False


def main():
    state = "line following"

    # reset arm
    got.mechanical_joint_control(0, 45, 45, 500)
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
            line_type = line_follow(mult=0.25, speed=30)
            if line_type == 0:
                print("no line")
                got.mecanum_stop()
                state = "face finding"
                print("Looking for face...")

        elif state == "face finding":
            arrived = face_approach(
                target_face="Ryan", gap=10, strafe_spd=10, fwd_spd=10, height=200
            )
            if arrived:
                putdown()
                break

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    got.mecanum_stop()


if __name__ == "__main__":
    main()
