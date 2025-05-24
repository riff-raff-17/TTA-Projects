"""
Robot control script for locating, picking up, and returning AprilTag-marked objects using a mecanum-wheeled robot. 
Integrates the UGOT API for movement, manipulation, and onboard display feedback.
Using UGOT 0.1.1 for compatibility with Python 3.6.
This version does not include camera image processing or display.
"""

from ugot import ugot
# import cv2
# import numpy as np
import time

# Instantiate the UGOT robot controller
got = ugot.UGOT()

# Movement speed constants
SPEED = 15       # Forward/backward speed
SIDE_SPEED = 10  # Lateral speed

def go_there():
    """Navigate from home base into the search area and set initial heading."""
    got.mecanum_move_speed_times(0, 30, 50, 1)
    got.mecanum_turn_speed_times(2, 45, 90, 2)

def go_back():
    """Return to home base and release any held object."""
    got.mecanum_move_speed_times(1, 30, 50, 1)
    got.mecanum_turn_speed_times(2, 45, 90, 2)
    got.mecanum_move_speed_times(0, 30, 60, 1)
    # Always open the gripper to drop the object
    got.mechanical_clamp_release()

def pick_up():
    """
    Attempt to pick up an object directly beneath the gripper.
    Returns True on successful grasp (no AprilTag detected after closing),
    False otherwise.
    """
    # Halt all wheel motion
    got.mecanum_stop() 
    # Lower the arm to approach object
    got.mechanical_joint_control(0, -20, -40, 500)
    time.sleep(1)
    # Close gripper on object
    got.mechanical_clamp_close()
    time.sleep(1)
    # Lift the arm back up
    got.mechanical_joint_control(0, 30, 30, 500)
    time.sleep(1)

    # Verify pick by checking for tag visibility
    if got.get_qrcode_apriltag_total_info()[1] != -1:
        # Tag still visible -> failed grasp
        got.screen_display_background(3)  # red background
        got.mechanical_clamp_release()    # drop object
        return False
    else:
        # Tag no longer visible -> successful grasp
        got.screen_display_background(6)  # green background
        return True

def seek_qrcode():
    """
    Continuously capture frames, detect AprilTags, align and approach them,
    and attempt pickup when within threshold distance.
    """
    attempts = 0
    while True:
        # # Read raw JPEG from camera and decode to OpenCV image
        # frame = got.read_camera_data()
        # if not frame:
        #     break
        # frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        # Retrieve all detected AprilTags
        tags = got.get_qrcode_apriltag_total_info()
        # NB: UGOT 0.1.1 assumes only one tag is present at a time
        if tags[1] != -1: # UGOT sets all entries except first to -1 if no tags are detected
            # Unpack properties of the first detection
            # _, cx, cy, h, w, _, distance, *_ = tags[0]
            _, _, cx, _, _, _, _, distance, *_ = tags

            # If within pickup range, try grasp
            if distance < 0.19:
                if not pick_up():
                    attempts += 1
                else:
                    break  # success, exit loop

                if attempts > 3:
                    break  # too many failures, give up

                # Back up slightly before retrying
                got.mecanum_move_speed_times(1, 30, 20, 1)

            # Strafe to center the tag in camera FOV
            if cx > 340:
                got.mecanum_move_xyz(SIDE_SPEED, SPEED, 0)
            elif cx < 300:
                got.mecanum_move_xyz(-SIDE_SPEED, SPEED, 0)
            else:
                got.mecanum_move_xyz(0, SPEED, 0)

            # # Draw detection bounding box
            # x1, y1 = int(cx - w/2), int(cy - h/2)
            # x2, y2 = int(cx + w/2), int(cy + h/2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            # No tag detected: stop movement and mark values as NaN
            distance = cx = "NaN"
            got.mecanum_stop()

        # # Overlay distance and center-x information on frame
        # cv2.putText(
        #     frame,
        #     f'Distance: {distance} | Center X: {cx} / 320',
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (0, 255, 0),
        #     2
        # )

        # # Display processed image; exit on 'q' keypress
        # cv2.imshow('Image', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Clean up on exit
    got.mecanum_stop()
    # cv2.destroyAllWindows()

def main():
    """Initialize systems, perform search-and-fetch routine, then return home."""
    got.initialize('192.168.1.189')           # Connect to robot over network
    # got.open_camera()                         # Start video stream
    got.load_models(['apriltag_qrcode'])      # Enable AprilTag/QR detection
    # Pre-position arm and open gripper
    got.mechanical_joint_control(0, 0, -20, 500)
    got.mechanical_clamp_release()
    got.screen_clear()                        # Clear any previous display

    go_there()    # Navigate to search area
    seek_qrcode()   # Locate tag, approach, and pick up
    go_back()     # Return and drop object

if __name__ == "__main__":
    main()
