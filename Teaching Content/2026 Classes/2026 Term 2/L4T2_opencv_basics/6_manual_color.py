import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize("192.168.1.197")
got.open_camera()

# Speed ranges
MOVE_MIN, MOVE_MAX = 5, 80
TURN_MIN, TURN_MAX = 5, 280


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


# ── Color tracking state ──────────────────────────────────────────────────────

tracking = False  # whether tracking mode is active
hsv_lower = None  # np.array([H-tol, S-tol, V-tol])
hsv_upper = None  # np.array([H+tol, S+tol, V+tol])

# Mouse drag state
drag_start = None  # (x, y) where mouse button went down
drag_end = None  # (x, y) current drag position (while held)
drag_done = False  # True when mouse button was just released


def mouse_callback(event, x, y, flags, param):
    global drag_start, drag_end, drag_done
    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        drag_end = (x, y)
        drag_done = False
    elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
        drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
        drag_end = (x, y)
        drag_done = True


def sample_color_from_rect(frame_bgr, pt1, pt2):
    """Return (hsv_lower, hsv_upper) sampled from the dragged rectangle."""
    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
    # Guard against a zero-size box
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None, None

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = hsv.mean(axis=(0, 1))  # average H, S, V across the region

    h_tol, s_tol, v_tol = 15, 60, 60
    lower = np.array(
        [max(0, mean[0] - h_tol), max(0, mean[1] - s_tol), max(0, mean[2] - v_tol)],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            min(179, mean[0] + h_tol),
            min(255, mean[1] + s_tol),
            min(255, mean[2] + v_tol),
        ],
        dtype=np.uint8,
    )
    return lower, upper


def find_largest_blob(frame_bgr, lower, upper):
    """Return (cx, cy, radius) of the largest matching blob, or None."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:  # ignore tiny blobs
        return None

    (cx, cy), radius = cv2.minEnclosingCircle(c)
    return int(cx), int(cy), int(radius)


# ── Tracking control ──────────────────────────────────────────────────────────

# Dead-zone: fraction of frame width where we don't turn
DEAD_ZONE = 0.15
# Minimum blob radius before we try to move forward
MIN_RADIUS = 30
# Desired blob radius (how close we want to get)
TARGET_RADIUS = 80


def drive_toward(cx, cy, radius, frame_w, frame_h, move_speed, turn_speed):
    """Issue one movement command based on blob position."""
    center_x = frame_w / 2
    offset = (cx - center_x) / center_x  # -1 … +1

    if abs(offset) > DEAD_ZONE:
        # Turn toward the target
        if offset > 0:
            got.mecanum_turn_speed(3, turn_speed)  # right
        else:
            got.mecanum_turn_speed(2, turn_speed)  # left
    elif radius < MIN_RADIUS:
        got.mecanum_move_speed(0, move_speed)  # forward
    elif radius > TARGET_RADIUS:
        got.mecanum_move_speed(1, move_speed)  # backward
    else:
        got.mecanum_stop()  # we're there


# ── Main loop ─────────────────────────────────────────────────────────────────


def main():
    global tracking, hsv_lower, hsv_upper
    global drag_start, drag_end, drag_done

    move_speed = 30
    turn_speed = 45

    cv2.namedWindow("Webcam Feed")
    cv2.setMouseCallback("Webcam Feed", mouse_callback)

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

        h, w = data.shape[:2]

        # ── Handle completed drag: sample color ───────────────────────────
        if drag_done and drag_start and drag_end:
            hsv_lower, hsv_upper = sample_color_from_rect(data, drag_start, drag_end)
            if hsv_lower is not None:
                tracking = True
                print(f"Tracking color  lower={hsv_lower}  upper={hsv_upper}")
            drag_start = drag_end = None
            drag_done = False

        # ── Draw live drag rectangle ──────────────────────────────────────
        if drag_start and drag_end and not drag_done:
            cv2.rectangle(data, drag_start, drag_end, (0, 255, 255), 2)

        # ── Tracking mode ─────────────────────────────────────────────────
        if tracking and hsv_lower is not None:
            result = find_largest_blob(data, hsv_lower, hsv_upper)
            if result:
                cx, cy, radius = result
                # Draw blob indicator
                cv2.circle(data, (cx, cy), radius, (0, 255, 0), 2)
                cv2.circle(data, (cx, cy), 4, (0, 255, 0), -1)
                drive_toward(cx, cy, radius, w, h, move_speed, turn_speed)
            else:
                got.mecanum_stop()  # lost target

        # ── HUD ───────────────────────────────────────────────────────────
        mode = "TRACKING" if tracking else "MANUAL"
        cv2.putText(
            data,
            f"Move: {move_speed}  Turn: {turn_speed}  [{mode}]",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if not tracking:
            cv2.putText(
                data,
                "Drag to select color",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                data,
                "c = clear  |  drag = reselect",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Webcam Feed", data)

        key = cv2.waitKey(1) & 0xFF

        # --- WASD manual movement (always available) ---
        if key == ord("w"):
            tracking = False
            got.mecanum_move_speed(0, move_speed)
        elif key == ord("s"):
            tracking = False
            got.mecanum_move_speed(1, move_speed)
        elif key == ord("a"):
            tracking = False
            got.mecanum_turn_speed(2, turn_speed)
        elif key == ord("d"):
            tracking = False
            got.mecanum_turn_speed(3, turn_speed)
        elif key == ord(" "):
            got.mecanum_stop()
        elif key == ord("q"):
            break

        # --- c to clear tracking ---
        elif key == ord("c"):
            tracking = False
            hsv_lower = None
            hsv_upper = None
            drag_start = drag_end = None
            got.mecanum_stop()
            print("Tracking cleared — manual mode")

        # --- Arrow keys adjust speeds ---
        if key == 0:
            move_speed = clamp(move_speed + 5, MOVE_MIN, MOVE_MAX)
            print(f"Move speed -> {move_speed}")
        elif key == 1:
            move_speed = clamp(move_speed - 5, MOVE_MIN, MOVE_MAX)
            print(f"Move speed -> {move_speed}")
        elif key == 2:
            turn_speed = clamp(turn_speed - 5, TURN_MIN, TURN_MAX)
            print(f"Turn speed -> {turn_speed}")
        elif key == 3:
            turn_speed = clamp(turn_speed + 5, TURN_MIN, TURN_MAX)
            print(f"Turn speed -> {turn_speed}")

    got.mecanum_stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
