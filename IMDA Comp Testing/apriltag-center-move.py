from ugot import ugot
import cv2
import numpy as np
import time

got = ugot.UGOT()

# Constants
# Good values: SPEED = 15, SIDE_SPEED = 10
SPEED = 15
SIDE_SPEED = 10

def go_there():
    got.mecanum_move_speed_times(0, 30, 60, 1)
    got.mecanum_translate_speed_times(-90, 30, 90, 1)
    got.mecanum_move_speed_times(0, 30, 120, 1)
    got.mecanum_translate_speed_times(90, 30, 60, 1)
    got.mecanum_move_speed_times(0, 30, 75, 1)
    got.mecanum_turn_speed_times(2, 45, 90, 2)

def go_back():
    got.mecanum_move_speed_times(1, 30, 90, 1)
    got.mecanum_turn_speed_times(2, 45, 90, 2)
    got.mecanum_move_speed_times(0, 30, 75, 1)
    got.mecanum_translate_speed_times(90, 30, 60, 1)
    got.mecanum_move_speed_times(0, 30, 120, 1)
    got.mecanum_translate_speed_times(-90, 30, 90, 1)
    got.mecanum_move_speed_times(0, 30, 60, 1)


def pick_up():
    got.mecanum_stop()
    got.mechanical_joint_control(0, -20, -40, 500)
    time.sleep(1)
    got.mechanical_clamp_close()
    time.sleep(1)
    got.mechanical_joint_control(0, 30, 30, 500)
    time.sleep(1)
    if got.get_apriltag_total_info():
        # Red background
        got.screen_display_background(3)
        got.mechanical_clamp_release()
        return False
    else:
        # Green background
        got.screen_display_background(6)
        return True 


def main():
    # --------------------
    # 1. Robot initialization
    # --------------------
    got.initialize('192.168.1.189')            # connect to robot
    got.open_camera()                          # start camera
    got.load_models(['apriltag_qrcode'])       # enable tag/QR detection
    got.mechanical_joint_control(0, 0, -20, 500)  # position arm
    got.mechanical_clamp_release()             # open gripper
    got.screen_clear()

    go_there()
    # --------------------
    # 2. Seek-and-align loop
    # --------------------
    while True:
        frame = got.read_camera_data()
        if not frame:
            break
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        tags = got.get_apriltag_total_info()
        if tags:
            # use first detection
            # cx: center x position
            # distance: distance in meters to target
            _, cx, cy, h, w, _, distance, *_ = tags[0]
            if distance < 0.19:
                if pick_up():
                    go_back()
                    break
                else:
                    got.mecanum_move_speed_times(1, 30, 20, 1)
            # steer toward center
            if cx > 340:
                got.mecanum_move_xyz(SIDE_SPEED, SPEED, 0)
            elif cx < 300:
                got.mecanum_move_xyz(-SIDE_SPEED, SPEED, 0)
            else:
                got.mecanum_move_xyz(0, SPEED, 0)

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2  = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            distance = "NaN"
            cx = "NaN"
            got.mecanum_stop()

        # Display stats
        cv2.putText(frame, f'Distance: {distance} | Center X: {cx} / 320', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # --------------------
    # 3. Grasp & finish
    # --------------------
    cv2.destroyAllWindows()
    got.mechanical_joint_control(0, -20, -35, 500)
    got.mechanical_clamp_release()


if __name__ == "__main__":
    main()
