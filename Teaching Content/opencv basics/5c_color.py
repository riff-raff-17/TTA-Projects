import cv2
import numpy as np
from ugot import ugot

# --- Connect to the robot and open the camera ---
got = ugot.UGOT()
got.initialize("192.168.1.193")
got.open_camera()

print("Camera opened. Press 't' to toggle tracking, 'q' to quit.")

# --- HSV color range for a RED object ---
# Hue for red wraps around 0/180 in HSV, so we need two ranges to capture it fully
RED_LOW1 = np.array([0, 120, 70])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])

MIN_AREA = 2000  # ignore tiny blobs (noise); increase if getting false detections

# --- NEW: Steering settings ---
TURN_SPEED = 40  # how fast the robot turns to chase the object
DEADZONE = 100   # how many pixels off-center before we bother turning
TURN_LEFT = 2    # direction constant for the mecanum drive
TURN_RIGHT = 3   # direction constant for the mecanum drive


def find_object(frame):
    """
    Convert frame to HSV, build a red mask, find the biggest blob.
    Returns (cx, cy, area, mask).
    cx/cy/area are None if no object is found.
    """
    # Convert from BGR (what OpenCV uses) to HSV (easier to filter by color)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Build a mask for each red range, then combine them
    mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find the outlines of all blobs in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, mask

    # Pick the largest blob
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)

    if area < MIN_AREA:
        return None, None, None, mask

    # Calculate the center of the blob from its bounding box
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy, area, mask


# --- NEW: tracking starts off; press 't' to enable ---
tracking = False

while True:
    # Grab a raw frame from the robot's camera
    frame = got.read_camera_data()

    # Check 1: did we actually receive any data?
    if frame is None or len(frame) == 0:
        print("Failed to grab frame")
        break

    # Decode the raw bytes into an image we can work with
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check 2: did the decode succeed?
    if data is None:
        print("Failed to decode frame")
        break

    # --- NEW: Draw a center line so you can see where "straight ahead" is ---
    frame_cx = data.shape[1] // 2
    cv2.line(data, (frame_cx, 0), (frame_cx, data.shape[0]), (255, 255, 0), 1)

    # Draw the deadzone boundaries (object must cross these before the robot turns)
    cv2.line(data, (frame_cx - DEADZONE, 0), (frame_cx - DEADZONE, data.shape[0]), (0, 165, 255), 1)
    cv2.line(data, (frame_cx + DEADZONE, 0), (frame_cx + DEADZONE, data.shape[0]), (0, 165, 255), 1)

    # Run color detection on the frame
    cx, cy, area, mask = find_object(data)

    # Show the mask in a second window (white = detected color, black = everything else)
    cv2.imshow("Mask", mask)

    if cx is not None:
        # Draw a dot at the detected object's center
        cv2.circle(data, (cx, cy), 12, (0, 255, 0), -1)
        cv2.putText(data, f"Object at x={cx}, y={cy}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- NEW: Steering logic ---
        if tracking:
            error = cx - frame_cx  # negative = object is left, positive = right

            if error < -DEADZONE:       # object is to the left → turn left
                got.mecanum_turn_speed(TURN_LEFT, TURN_SPEED)
            elif error > DEADZONE:      # object is to the right → turn right
                got.mecanum_turn_speed(TURN_RIGHT, TURN_SPEED)
            else:                       # object is roughly centered → stop turning
                got.mecanum_stop()
    else:
        if tracking:
            got.mecanum_stop()          # nothing found → stop
        cv2.putText(data, "No object found", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- NEW: HUD showing whether tracking is on or off ---
    mode = "TRACKING ON  (t=off)" if tracking else "TRACKING OFF (t=on)"
    cv2.putText(data, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the live camera feed in a window
    cv2.imshow("Camera Feed", data)

    # Wait 1ms for a keypress; quit if the user presses 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("t"):   # --- NEW: toggle tracking on/off ---
        tracking = not tracking
        if not tracking:
            got.mecanum_stop()

# Clean up
cv2.destroyAllWindows()
print("Camera closed. Goodbye!")