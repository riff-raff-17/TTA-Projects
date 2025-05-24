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
    got.screen_print_text_newline(f"NAVIGATING TO CODE", 1)
    time.sleep(3)
    got.screen_clear()

def go_back():
    got.screen_print_text_newline(f"NAVIGATING TO BASE", 1)
    time.sleep(3)
    got.screen_clear()


def pick_up():
    got.screen_display_background(8)
    got.screen_print_text_newline("Picking up...", 1)

    # Simulate pickup
    time.sleep(1)
    got.screen_print_text_newline("3..", 1)
    time.sleep(1)
    got.screen_print_text_newline("2..", 1)
    time.sleep(1)
    got.screen_print_text_newline("1..", 1)
    time.sleep(1)

    # Verify pick by checking for tag visibility
    if got.get_qrcode_apriltag_total_info()[1] != -1:
        # Tag still visible -> failed grasp
        got.screen_display_background(3)  # red background
        got.screen_print_text_newline("FAILED", 1)
        time.sleep(1)
        return False
    else:
        # Tag no longer visible -> successful grasp
        got.screen_display_background(6)  # green background
        got.screen_print_text_newline("SUCCESS", 1)
        time.sleep(1)
        return True

def seek_qrcode():
    """
    Continuously capture frames, detect AprilTags, align and approach them,
    and attempt pickup when within threshold distance.
    """
    attempts = 3
    got.screen_print_text_newline(f"BEGIN CENTERING", 1)
    time.sleep(2)
    while True:
        # Retrieve all detected AprilTags
        tags = got.get_qrcode_apriltag_total_info()
        # NB: UGOT 0.1.1 assumes only one tag is present at a time
        if tags[1] != -1:
            _, _, cx, _, _, _, _, distance, *_ = tags

            # If within pickup range, try grasp
            if distance < 0.15:
                got.screen_print_text_newline(f"ATTEMPTS LEFT: {3 - attempts}", 1)
                if not pick_up():
                    attempts += 1
                else:
                    break  # success, exit loop

                if attempts > 3:
                    got.screen_print_text_newline(f"TOO MANY ATTEMPTS. STOPPING...", 1)
                    time.sleep(1)
                    break  # too many failures, give up
            else:
                # Strafe to center the tag in camera FOV
                if cx > 340:
                    got.screen_display_background(3)
                    got.screen_print_text_newline("GO RIGHT", 1)
                elif cx < 300:
                    got.screen_display_background(3)
                    got.screen_print_text_newline("GO LEFT", 1)
                else:
                    got.screen_display_background(6)
                    got.screen_print_text_newline("GO STRAIGHT", 1)

        else:
            # No tag detected: stop movement and mark values as NaN
            distance = cx = "NaN"
            got.screen_display_background(0)
            got.screen_print_text_newline("NO CODE DETECTED", 1)
        
        time.sleep(0.75)
        got.screen_clear()


def main():
    """Initialize systems, perform search-and-fetch routine, then return home."""
    got.initialize('192.168.88.1')           # Connect to robot over network
    # got.open_camera()                         # Start video stream
    got.load_models(['apriltag_qrcode'])      # Enable AprilTag/QR detection
    # Pre-position arm and open gripper
    got.screen_clear()                        # Clear any previous display

    go_there()    # Navigate to search area
    seek_qrcode()   # Locate tag, approach, and pick up
    go_back()     # Return and drop object

if __name__ == "__main__":
    main()
