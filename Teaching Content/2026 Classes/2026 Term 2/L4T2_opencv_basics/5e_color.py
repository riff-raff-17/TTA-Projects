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

# --- Steering settings ---
TURN_SPEED = 40  # how fast the robot turns to chase the object
TURN_LEFT = 2  # direction constant for the mecanum drive
TURN_RIGHT = 3  # direction constant for the mecanum drive

# Hysteresis thresholds for turning.
# The robot starts turning once error exceeds TURN_ENTER,
# and doesn't stop until error drops below TURN_EXIT.
TURN_ENTER = 100  # px — start turning when error is bigger than this
TURN_EXIT = 60  # px — stop turning only once error is smaller than this

# --- Forward/backward distance control settings ---
DRIVE_SPEED = 40  # how fast the robot drives toward/away from the object
FORWARD = 1  # direction constant for forward drive
BACKWARD = 0  # direction constant for backward drive
TARGET_AREA = 18000  # blob area (px²) that means "just right" distance
#   Tune TARGET_AREA by watching the "Area" readout in the HUD at your desired distance.

# Hysteresis thresholds for distance.
# The robot starts driving once area error exceeds AREA_ENTER,
# and doesn't stop until area error drops below AREA_EXIT.
AREA_ENTER = 4000  # px² — start driving when area diff is bigger than this
AREA_EXIT = 2000  # px² — stop driving only once area diff is smaller than this

# --- Confidence HUD settings ---
# MAX_AREA is used only to scale the confidence bar; set it to roughly the largest
# blob area you'd ever expect (e.g. object fills most of the frame).
MAX_AREA = 80000


# ---------------------------------------------------------------------------
# Hysteresis state
# Each flag remembers whether the robot is *currently* in that behaviour.
# It flips ON when the error exceeds ENTER, and flips OFF only when it falls
# below EXIT — preventing rapid toggling right at the threshold.
# ---------------------------------------------------------------------------
is_turning = False  # True while we are actively correcting left/right
is_driving = False  # True while we are actively correcting distance


# ---------------------------------------------------------------------------
# Command deduplication
# We track the last string we sent to the motors.  send_command() compares
# before calling any ugot API, so we never spam the same command every frame.
# ---------------------------------------------------------------------------
last_command = None  # e.g. "turn_left", "forward", "stop", "none"


def send_command(cmd):
    """Issue a motor command only when it differs from the previous one."""
    global last_command
    if cmd == last_command:
        return  # nothing changed — skip the API call
    last_command = cmd

    if cmd == "turn_left":
        got.mecanum_turn_speed(TURN_LEFT, TURN_SPEED)
    elif cmd == "turn_right":
        got.mecanum_turn_speed(TURN_RIGHT, TURN_SPEED)
    elif cmd == "forward":
        got.mecanum_move_speed(FORWARD, DRIVE_SPEED)
    elif cmd == "backward":
        got.mecanum_move_speed(BACKWARD, DRIVE_SPEED)
    elif cmd == "stop":
        got.mecanum_stop()
    # "none" means tracking is off — don't touch the motors at all


def find_object(frame):
    """
    Convert frame to HSV, build a red mask, find the biggest blob.
    Returns (cx, cy, area, bbox, mask).
      cx, cy  - center of the blob (None if not found)
      area    - contour area in pixels² (None if not found)
      bbox    - (x, y, w, h) bounding rect (None if not found)
      mask    - the binary mask image (always returned for display)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, None, mask

    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)

    if area < MIN_AREA:
        return None, None, None, None, mask

    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy, area, (x, y, w, h), mask


# --- tracking starts off; press 't' to enable ---
tracking = False

while True:
    frame = got.read_camera_data()

    if frame is None or len(frame) == 0:
        print("Failed to grab frame")
        break

    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if data is None:
        print("Failed to decode frame")
        break

    frame_h, frame_w = data.shape[:2]
    frame_cx = frame_w // 2

    # Draw center line and deadzone boundaries on the camera feed
    cv2.line(data, (frame_cx, 0), (frame_cx, frame_h), (255, 255, 0), 1)
    cv2.line(
        data,
        (frame_cx - TURN_ENTER, 0),
        (frame_cx - TURN_ENTER, frame_h),
        (0, 165, 255),
        1,
    )
    cv2.line(
        data,
        (frame_cx + TURN_ENTER, 0),
        (frame_cx + TURN_ENTER, frame_h),
        (0, 165, 255),
        1,
    )

    cx, cy, area, bbox, mask = find_object(data)

    # --- Mask window: colorize so it's easier to read at a glance ---
    # Convert the binary mask to a 3-channel image, then tint it red.
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 0] = 0  # zero out the blue channel  → leaves only red
    mask_color[:, :, 1] = 0  # zero out the green channel → pure red tint
    cv2.imshow("Mask (red = detected)", mask_color)

    if cx is not None:
        x, y, w, h = bbox

        # Bounding box
        cv2.rectangle(data, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crosshair at object center
        CROSS = 10
        cv2.line(data, (cx - CROSS, cy), (cx + CROSS, cy), (0, 255, 0), 2)
        cv2.line(data, (cx, cy - CROSS), (cx, cy + CROSS), (0, 255, 0), 2)

        # Bounding box size label
        cv2.putText(
            data,
            f"{w}x{h}px",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Object position readout
        cv2.putText(
            data,
            f"Object at x={cx}, y={cy}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # Area / distance readout
        area_diff = area - TARGET_AREA
        dist_hint = (
            "TOO CLOSE"
            if area_diff > AREA_ENTER
            else "TOO FAR" if area_diff < -AREA_ENTER else "GOOD DIST"
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

        # Confidence bar
        conf = min(area / MAX_AREA, 1.0)
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

        # --- Motor logic (unified single-command-per-frame) ---
        if tracking:
            error = cx - frame_cx  # negative = left, positive = right

            # --- Hysteresis: update is_turning state ---
            if not is_turning and abs(error) > TURN_ENTER:
                is_turning = True  # crossed the outer threshold → start turning
            elif is_turning and abs(error) < TURN_EXIT:
                is_turning = False  # crossed the inner threshold → stop turning

            # --- Hysteresis: update is_driving state ---
            if not is_driving and abs(area_diff) > AREA_ENTER:
                is_driving = True  # crossed the outer threshold → start driving
            elif is_driving and abs(area_diff) < AREA_EXIT:
                is_driving = False  # crossed the inner threshold → stop driving

            # --- Decide on a single command for this frame ---
            # Turning takes priority over driving so the robot faces the object
            # before it tries to close or open the distance.
            if is_turning:
                cmd = "turn_left" if error < 0 else "turn_right"
            elif is_driving:
                cmd = "backward" if area_diff > 0 else "forward"
            else:
                cmd = "stop"

            send_command(cmd)

            # Show the active command on the HUD
            cv2.putText(
                data,
                f"CMD: {cmd}",
                (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

    else:
        # Reset hysteresis state when the object is lost so we don't carry
        # stale is_turning / is_driving flags into the next detection.
        is_turning = False
        is_driving = False

        if tracking:
            send_command("stop")
        cv2.putText(
            data,
            "No object found",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # HUD: tracking mode
    mode = "TRACKING ON  (t=off)" if tracking else "TRACKING OFF (t=on)"
    cv2.putText(data, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Camera Feed", data)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("t"):
        tracking = not tracking
        if not tracking:
            # Reset state and stop motors when tracking is toggled off
            is_turning = False
            is_driving = False
            send_command("stop")

# Clean up
cv2.destroyAllWindows()
print("Camera closed. Goodbye!")
