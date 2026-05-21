import cv2
import numpy as np
from ugot import ugot

# --- Connect to the robot and open the camera ---
got = ugot.UGOT()
got.initialize("192.168.1.46")
got.open_camera()

print("Camera opened. Press 't' to toggle tracking, 'q' to quit.")

# --- HSV color range for a RED object ---
# Hue for red wraps around 0/180 in HSV, so we need two ranges to capture it fully
RED_LOW1 = np.array([0, 120, 70])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])

MIN_AREA = 2000  # ignore tiny blobs (noise); increase if getting false detections

# --- Steering settings ---
TURN_SPEED = 40  # how fast the robot turns to chase the object
DEADZONE = 100  # how many pixels off-center before we bother turning
TURN_LEFT = 2  # direction constant for the mecanum drive
TURN_RIGHT = 3  # direction constant for the mecanum drive

# --- NEW: Forward/backward distance control settings ---
DRIVE_SPEED = 20  # how fast the robot drives toward/away from the object
FORWARD = 0  # direction constant for forward drive
BACKWARD = 1  # direction constant for backward drive
TARGET_AREA = 18000  # blob area (px²) that means "just right" distance
AREA_DEADZONE = 4000  # area must differ from TARGET by this much before we move
#   Tune TARGET_AREA by watching the "Area" readout in the HUD at your desired distance.

# --- NEW: Confidence HUD settings ---
# MAX_AREA is used only to scale the confidence bar; set it to roughly the largest
# blob area you'd ever expect (e.g. object fills most of the frame).
MAX_AREA = 80000


def find_object(frame):
    """
    Convert frame to HSV, build a red mask, find the biggest blob.
    Returns (cx, cy, area, bbox, mask).
      cx, cy  - center of the blob (None if not found)
      area    - contour area in pixels² (None if not found)
      bbox    - (x, y, w, h) bounding rect (None if not found)
      mask    - the binary mask image (always returned for display)
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
        return None, None, None, None, mask

    # Pick the largest blob
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)

    if area < MIN_AREA:
        return None, None, None, None, mask

    # Calculate the center of the blob from its bounding box
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    # --- NEW: return the full bbox so the caller can draw it ---
    return cx, cy, area, (x, y, w, h), mask


# --- tracking starts off; press 't' to enable ---
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

    frame_h, frame_w = data.shape[:2]
    frame_cx = frame_w // 2

    # --- Draw a center line so you can see where "straight ahead" is ---
    cv2.line(data, (frame_cx, 0), (frame_cx, frame_h), (255, 255, 0), 1)

    # Draw the deadzone boundaries (object must cross these before the robot turns)
    cv2.line(
        data, (frame_cx - DEADZONE, 0), (frame_cx - DEADZONE, frame_h), (0, 165, 255), 1
    )
    cv2.line(
        data, (frame_cx + DEADZONE, 0), (frame_cx + DEADZONE, frame_h), (0, 165, 255), 1
    )

    # Run color detection on the frame
    cx, cy, area, bbox, mask = find_object(data)

    # Show the mask in a second window (white = detected color, black = everything else)
    cv2.imshow("Mask", mask)

    if cx is not None:
        x, y, w, h = bbox

        # --- NEW: Draw bounding box around the detected object ---
        cv2.rectangle(data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # --- NEW: Draw a small crosshair at the object's center ---
        CROSS = 10
        cv2.line(data, (cx - CROSS, cy), (cx + CROSS, cy), (0, 255, 0), 2)
        cv2.line(data, (cx, cy - CROSS), (cx, cy + CROSS), (0, 255, 0), 2)

        # --- NEW: Label the bounding box with its pixel dimensions ---
        cv2.putText(
            data,
            f"{w}x{h}px",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Object position readout (existing, unchanged position)
        cv2.putText(
            data,
            f"Object at x={cx}, y={cy}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # --- NEW: Area / distance readout ---
        area_diff = area - TARGET_AREA
        dist_hint = (
            "TOO CLOSE"
            if area_diff > AREA_DEADZONE
            else "TOO FAR" if area_diff < -AREA_DEADZONE else "GOOD DIST"
        )
        cv2.putText(
            data,
            f"Area={int(area)}  [{dist_hint}]",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # --- NEW: Confidence bar (scaled to MAX_AREA) ---
        conf = min(area / MAX_AREA, 1.0)  # 0.0 – 1.0
        bar_x, bar_y, bar_w, bar_h = 10, 110, 200, 16
        cv2.rectangle(
            data, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1
        )
        cv2.rectangle(
            data,
            (bar_x, bar_y),
            (bar_x + int(bar_w * conf), bar_y + bar_h),
            (0, 200, 0),
            -1,
        )
        cv2.rectangle(
            data, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1
        )
        cv2.putText(
            data,
            f"Conf {int(conf * 100)}%",
            (bar_x + bar_w + 6, bar_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        # --- Steering + distance logic (tracking gate unchanged) ---
        if tracking:
            error = cx - frame_cx
            area_diff = area - TARGET_AREA  # move this calc outside the bbox block too

            turning = abs(error) > DEADZONE
            too_close = area_diff > AREA_DEADZONE
            too_far = area_diff < -AREA_DEADZONE

            if turning and too_far:
                # Turn AND drive forward simultaneously - use mecanum strafe/move combo
                # or just prioritize turning until roughly centered
                got.mecanum_turn_speed(
                    TURN_LEFT if error < 0 else TURN_RIGHT, TURN_SPEED
                )
            elif turning:
                got.mecanum_turn_speed(
                    TURN_LEFT if error < 0 else TURN_RIGHT, TURN_SPEED
                )
            elif too_close:
                got.mecanum_move_speed(BACKWARD, DRIVE_SPEED)
            elif too_far:
                got.mecanum_move_speed(FORWARD, DRIVE_SPEED)
            else:
                got.mecanum_stop()  # only ONE stop, when everything is satisfied
    else:
        if tracking:
            got.mecanum_stop()  # nothing found → stop
        cv2.putText(
            data,
            "No object found",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # --- HUD showing whether tracking is on or off ---
    mode = "TRACKING ON  (t=off)" if tracking else "TRACKING OFF (t=on)"
    cv2.putText(data, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the live camera feed in a window
    cv2.imshow("Camera Feed", data)

    # Wait 1ms for a keypress; quit if the user presses 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("t"):  # toggle tracking on/off
        tracking = not tracking
        if not tracking:
            got.mecanum_stop()

# Clean up
cv2.destroyAllWindows()
print("Camera closed. Goodbye!")
