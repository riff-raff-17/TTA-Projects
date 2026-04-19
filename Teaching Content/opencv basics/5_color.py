import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.1.163") 
got.open_camera()

# --- Tunable settings ---
TURN_SPEED = 40  # how fast the robot turns to chase the object
MIN_AREA = 2000  # ignore tiny blobs (noise); increase if getting false detections

# --- HSV color range for a RED object ---
# HSV hue for red wraps around 0/180, so we need two ranges
RED_LOW1 = np.array([0, 120, 70])
RED_HIGH1 = np.array([10, 255, 255])
RED_LOW2 = np.array([170, 120, 70])
RED_HIGH2 = np.array([180, 255, 255])


def find_object(frame):
    """
    Convert frame to HSV, build a red mask, find the biggest blob.
    Returns (cx, cy, area, mask) — mask is always returned so we can display it.
    Result fields cx/cy/area are None if nothing found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Combine both red ranges into one mask
    mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
    mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find outlines of blobs in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, mask

    # Pick the biggest blob
    biggest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(biggest)

    if area < MIN_AREA:
        return None, None, None, mask

    # Get center of the blob using its bounding box
    x, y, w, h = cv2.boundingRect(biggest)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy, area, mask


def main():
    tracking = False  # press 't' to toggle tracking on/off

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

        frame_w = data.shape[1]
        frame_cx = frame_w // 2  # horizontal center of the frame

        # Draw a center line so you can see where "straight ahead" is
        cv2.line(data, (frame_cx, 0), (frame_cx, data.shape[0]), (255, 255, 0), 1)

        cx, cy, area, mask = find_object(data)

        # Show the mask in a second window (white = detected color, black = everything else)
        cv2.imshow("Mask", mask)

        if cx is not None:

            # Draw a circle at the object center
            cv2.circle(data, (cx, cy), 12, (0, 255, 0), -1)

            # Draw bounding rectangle (just re-derive it from the center for display)
            cv2.putText(
                data,
                f"Object at x={cx}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # --- Steering logic ---
            if tracking:
                error = cx - frame_cx  # negative = object is left, positive = right

                if error < -30:  # object is to the left → turn left
                    got.mecanum_turn_speed(2, TURN_SPEED)
                elif error > 30:  # object is to the right → turn right
                    got.mecanum_turn_speed(3, TURN_SPEED)
                else:  # object is roughly centered → stop turning
                    got.mecanum_stop()
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

        # HUD
        mode = "TRACKING ON  (t=off)" if tracking else "TRACKING OFF (t=on)"
        cv2.putText(
            data, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.imshow("Color Tracker", data)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            got.mecanum_stop()
            break
        elif key == ord("t"):  # toggle tracking on/off
            tracking = not tracking
            if not tracking:
                got.mecanum_stop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
