"""
final_helpers.py — All-in-one module for robot control.

Contains:
  - Direction logic (get_direction)
  - Overlay drawing (draw_overlay)
  - Model download (download_model)
  - Robot action functions (forward / backward / left / right / stop / dispatch)
  - Main webcam loop (run_control_loop) — callable from a script
  - Notebook entry point (run_in_notebook) — inline display, no ipywidgets needed
  - AprilTag approach (AP_centralization_approaching)
  - Object pick-up (pick_up)
  - Face search and approach (face_find_and_approach)

Usage from a script:
    from final_helpers import run_control_loop
    run_control_loop(robot_ip="192.168.1.91")

Usage from a Jupyter notebook:
    from final_helpers import run_in_notebook
    try:
        run_in_notebook(robot_ip="192.168.1.91")
    except KeyboardInterrupt:
        pass
"""

import os
import time
import threading
import cv2
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode

from ugot import ugot

# ── Constants ──────────────────────────────────────────────────────────────────

DEADZONE = 0.15  # 15 % on each side → 30 % total dead-band
INDEX_FINGER_TIP = 8

DIRECTION_COLORS = {
    "forward":  (0, 200, 0),
    "backward": (0, 0, 200),
    "left":     (200, 150, 0),
    "right":    (0, 150, 200),
    "stop":     (160, 160, 160),
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = "hand_landmarker.task"

# ── Model download ─────────────────────────────────────────────────────────────

def download_model():
    """Download the MediaPipe hand-landmark model if it isn't present yet."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.\n")

# ── Direction logic ────────────────────────────────────────────────────────────

def get_direction(x_norm: float, y_norm: float) -> str:
    """
    Map a normalised (0–1) finger position to a robot command.

    x_norm : 0 = left edge,  1 = right edge
    y_norm : 0 = top edge,   1 = bottom edge  (MediaPipe convention)

    Returns one of: 'forward', 'backward', 'left', 'right', 'stop'
    """
    dx = x_norm - 0.5   # signed distance from centre (−0.5 … +0.5)
    dy = y_norm - 0.5

    if abs(dx) < DEADZONE and abs(dy) < DEADZONE:
        return "stop"

    # Whichever axis has the larger deviation wins
    if abs(dy) >= abs(dx):
        return "backward" if dy > 0 else "forward"
    else:
        return "right" if dx > 0 else "left"

# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_overlay(frame, direction: str, tip_px):
    """Draw deadzone box, fingertip dot, direction label, and crosshair."""
    h, w = frame.shape[:2]

    # Deadzone rectangle
    dz_x1 = int((0.5 - DEADZONE) * w)
    dz_x2 = int((0.5 + DEADZONE) * w)
    dz_y1 = int((0.5 - DEADZONE) * h)
    dz_y2 = int((0.5 + DEADZONE) * h)
    cv2.rectangle(frame, (dz_x1, dz_y1), (dz_x2, dz_y2), (220, 220, 220), 2)
    mid_y = (dz_y1 + dz_y2) // 2
    cv2.putText(frame, "DEAD", (dz_x1 + 4, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, "ZONE", (dz_x1 + 4, mid_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Fingertip dot
    color = DIRECTION_COLORS[direction]
    cv2.circle(frame, tip_px, 14, color, -1)
    cv2.circle(frame, tip_px, 14, (255, 255, 255), 2)

    # Direction label
    cv2.putText(frame, direction.upper(), (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3, cv2.LINE_AA)

    # Crosshair at centre
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (200, 200, 200), 1)

    # Quit hint
    cv2.putText(frame, "q = quit", (w - 100, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

# ── Robot action functions ─────────────────────────────────────────────────────

def robot_forward(got):
    print("FORWARD")
    got.mecanum_move_speed(direction=0, speed=20)

def robot_backward(got):
    print("BACKWARD")
    got.mecanum_move_speed(direction=1, speed=20)

def robot_left(got):
    print("LEFT")
    got.mecanum_turn_speed(turn=2, speed=45)

def robot_right(got):
    print("RIGHT")
    got.mecanum_turn_speed(turn=3, speed=45)

def robot_stop(got):
    print("STOP")
    got.mecanum_stop()

def dispatch(direction: str, got):
    """Call the appropriate robot function based on the direction string."""
    actions = {
        "forward":  robot_forward,
        "backward": robot_backward,
        "left":     robot_left,
        "right":    robot_right,
        "stop":     robot_stop,
    }
    actions[direction](got)

# ── Main control loop ──────────────────────────────────────────────────────────

def run_control_loop(
    robot_ip: str,
    frame_callback=None,
    stop_flag=None,
):
    """
    Start the hand-tracking + robot control loop.

    Parameters
    ----------
    robot_ip : str
        IP address of the UGOT robot (e.g. "192.168.1.91").
    frame_callback : callable | None
        Optional function called with each annotated BGR frame so a Jupyter
        notebook (or any other consumer) can display it inline.
        Signature: frame_callback(bgr_frame: np.ndarray) -> None
    stop_flag : threading.Event | None
        If provided, the loop exits when stop_flag.is_set() is True.
        If None, the loop exits only when the user presses 'q'.
    """
    download_model()

    # Connect to robot
    got = ugot.UGOT()
    got.initialize(robot_ip)

    # Build MediaPipe landmarker
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_direction = None
    print("Hand tracking started. Show your index finger to the camera.")
    print("Press 'q' to quit.\n")

    while True:
        # Honour external stop signal (used by notebook widgets)
        if stop_flag is not None and stop_flag.is_set():
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)          # mirror so left/right feel natural
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect(mp_image)

        direction = "stop"
        tip_px = (w // 2, h // 2)          # default dot position

        if results.hand_landmarks:
            lm = results.hand_landmarks[0]
            tip = lm[INDEX_FINGER_TIP]
            tip_px = (int(tip.x * w), int(tip.y * h))
            direction = get_direction(tip.x, tip.y)

        draw_overlay(frame, direction, tip_px)

        # Hand frame off to caller (e.g. notebook display)
        if frame_callback is not None:
            frame_callback(frame)
        else:
            cv2.imshow("Robot Finger Control", frame)

        # Only send a new command when the direction changes
        if direction != last_direction:
            dispatch(direction, got)
            last_direction = direction

        # 'q' to quit (only meaningful when an OpenCV window exists)
        if frame_callback is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    landmarker.close()
    cap.release()
    if frame_callback is None:
        cv2.destroyAllWindows()
    print("Stopped!")

# ── Notebook entry point ───────────────────────────────────────────────────────

def run_in_notebook(robot_ip: str, display_every_n: int = 3):
    """
    Start hand-tracking inline inside a Jupyter notebook cell.

    Frames are JPEG-encoded and pushed into a live IPython display handle so
    the image updates in place — no ipywidgets required.

    Blocks until the loop finishes, so KeyboardInterrupt (■ or Ctrl-C) stops
    it cleanly from a try/except in the calling cell:

        try:
            run_in_notebook("192.168.1.91")
        except KeyboardInterrupt:
            pass

    Parameters
    ----------
    robot_ip : str
        IP address of the UGOT robot (e.g. "192.168.1.91").
    display_every_n : int
        Render every Nth frame. Lower = smoother but heavier. Default is 3.
    """
    from IPython.display import display as ipy_display, Image as ipy_Image

    stop_flag = threading.Event()
    display_handle = ipy_display(ipy_Image(data=b""), display_id=True)
    frame_counter = [0]  # list so the closure can mutate it

    def _frame_callback(bgr_frame):
        frame_counter[0] += 1
        if frame_counter[0] % display_every_n != 0:
            return
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            display_handle.update(ipy_Image(data=buf.tobytes()))

    def _run():
        try:
            run_control_loop(
                robot_ip=robot_ip,
                frame_callback=_frame_callback,
                stop_flag=stop_flag,
            )
        except Exception as exc:
            print(f"Loop ended with error: {exc}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    print("Hand tracking running. Interrupt the kernel (■ or Ctrl-C) to stop.\n")

    try:
        thread.join()  # block here so KeyboardInterrupt reaches this cell
    except KeyboardInterrupt:
        stop_flag.set()
        thread.join(timeout=5)
        print("Stopped!")

# ── AprilTag functions ─────────────────────────────────────────────────────────

def AP_centralization_approaching(got, distance=0.15, gap=20, fwd_spd=10, strafe_spd=10):
    """
    Drive toward a detected AprilTag, keeping it centered in the camera frame.

    Parameters
    ----------
    got        : ugot.UGOT
        Connected robot instance.
    distance   : float
        Stop when the tag is within this many meters (default 0.15 m).
    gap        : int
        Pixel tolerance around center (320 px) before strafing (default 20 px).
    fwd_spd    : int
        Forward drive speed in cm/s (default 10).
    strafe_spd : int
        Left/right correction speed in cm/s (default 10).
    """
    try:
        # Get an initial reading to confirm a tag is visible before entering the loop.
        AP_info = got.get_apriltag_total_info()
        AP_x = AP_info[0][1]        # Horizontal pixel position (0=left, 640=right)
        AP_distance = AP_info[0][6] # Estimated distance to the tag in meters

        while True:
            # Refresh tag data every iteration for responsive corrections.
            AP_info = got.get_apriltag_total_info()
            AP_x = AP_info[0][1]
            AP_distance = AP_info[0][6]

            if AP_x < 320 - gap:
                # Tag is to the LEFT of center — strafe left to re-align.
                # mecanum_move_xyz(x, y, z): x=strafe, y=forward, z=rotation
                got.mecanum_move_xyz(-strafe_spd, strafe_spd, 0)
            elif AP_x > 320 + gap:
                # Tag is to the RIGHT of center — strafe right to re-align.
                got.mecanum_move_xyz(strafe_spd, strafe_spd, 0)
            elif AP_distance > distance:
                # Tag is centered but still too far — drive straight forward.
                got.mecanum_move_xyz(0, fwd_spd, 0)
            else:
                # Tag is centered AND within target distance — stop and exit.
                got.mecanum_stop()
                print("Close enough — stopping.")
                break

    except IndexError:
        print("ERROR: AprilTag cannot be seen.")
        got.mecanum_stop()


def pick_up(got):
    """
    Pick up the object identified by the closest visible AprilTag.

    Assumes the robot is already aligned and close to the target
    (i.e., AP_centralization_approaching() has just completed).

    Parameters
    ----------
    got : ugot.UGOT
        Connected robot instance.
    """
    AP_info = got.get_apriltag_total_info()
    try:
        AP_x = AP_info[0][1]
        AP_distance = AP_info[0][6]

        # Move arm to a neutral ready position and open the gripper.
        # joint_control(j1, j2, j3, duration_ms): j2=30, j3=30 tilts arm slightly forward.
        got.mechanical_joint_control(0, 30, 30, 1000)
        got.mechanical_clamp_release()  # Open gripper before extending arm
        time.sleep(2)                   # Wait for gripper to fully open

        # Calculate arm joint angles based on the tag's camera position.
        # joint1 (base): convert pixel offset from center to degrees.
        #   Negative factor corrects for the camera being mirrored horizontally.
        joint1 = int((AP_x - 320) * -1 / 10)

        # joint3 (furthest): convert distance (m) to an extension angle.
        # The -80 offset accounts for the arm's resting angle calibration.
        joint3 = int(AP_distance * 100 - 80)

        # Move arm to the computed pick-up pose.
        got.mechanical_joint_control(joint1, 0, joint3, 500)
        print(f"Joint1 value is: {joint1}, Joint3 value is: {joint3}.")
        time.sleep(1)  # Wait for arm to reach the target pose

        # Grasp the object and lift the arm back to the carry position.
        got.mechanical_clamp_close()
        time.sleep(2)                           # Wait for gripper to fully close
        got.mechanical_joint_control(0, 30, 30, 1000)  # Return arm to neutral carry pose

    except IndexError:
        print("ERROR: AprilTag cannot be seen.")

# ── Face recognition ───────────────────────────────────────────────────────────

def face_find_and_approach(
    got,
    gap=10,
    target_name="Stranger",
    turn_spd=15,
    strafe_spd=10,
    fwd_spd=10,
    height=80,
    adjust_turn=10,
):
    """
    Spin until the target person is found, then approach them.

    Phase 1 — Search:
        The robot turns continuously, checking each frame for the target's
        face (via face_recognition) or name tag (via OCR). When found, it
        stops and does a small corrective turn to face them, then moves to
        Phase 2.

    Phase 2 — Approach:
        The robot drives toward the face, strafing left/right to keep it
        centered in frame, until the face appears large enough (close enough).

    Parameters
    ----------
    got          : ugot.UGOT
        Connected robot instance.
    gap          : int
        Pixel tolerance around center (320 px) before strafing (default 10 px).
    target_name  : str
        Name to search for in face recognition or OCR results (default "Stranger").
    turn_spd     : int
        Spin speed while searching in Phase 1 (default 15).
    strafe_spd   : int
        Left/right correction speed during approach in Phase 2 (default 10).
    fwd_spd      : int
        Forward drive speed during approach in Phase 2 (default 10).
    height       : int
        Face bounding-box height (px) at which the robot considers itself
        close enough and stops (default 80).
    adjust_turn  : int
        Fine-correction turn amount after the target is first spotted,
        in unit=2 degree-units (default 10).
    """
    face_name = None  # Will hold the name from face recognition once a face is found

    try:
        # ── Phase 1: Spin and search ──────────────────────────────────────────
        while True:
            # Turn slowly to scan the environment
            got.mecanum_turn_speed(turn=3, speed=turn_spd)

            # Read text visible in the frame (e.g. a name tag)
            name = got.get_words_result()

            # Check for any recognized faces in the frame
            faces = got.get_face_recognition_total_info()
            if faces:
                face_name = faces[0][0]  # faces[0] = first face; [0] = its name

            # If either the OCR text or the face name matches our target, found!
            if name == target_name or face_name == target_name:
                got.mecanum_stop()
                print(f"Saw {target_name}!")

                # Small corrective turn to center the robot on the target.
                # turn=3 is clockwise; times=10, unit=2 means ~10 degree-units.
                got.mecanum_turn_speed_times(turn=3, speed=20, times=adjust_turn, unit=2)
                break  # Exit Phase 1, move on to Phase 2

        # ── Phase 2: Approach the target ──────────────────────────────────────
        while True:
            name = got.get_words_result()
            faces = got.get_face_recognition_total_info()

            if not faces:
                # Lost the face; inch forward slowly to try to find it again
                got.mecanum_translate_speed(angle=0, speed=fwd_spd)
            else:
                c_x = faces[0][1]  # Horizontal center of the face (0–640 px)
                h   = faces[0][3]  # Height of face bounding box (proxy for distance)

                if h < height:
                    if c_x < 320 - gap:
                        # Face is too far LEFT — strafe left while moving forward
                        got.mecanum_move_xyz(x_speed=-strafe_spd, y_speed=fwd_spd, z_speed=0)
                    elif c_x > 320 + gap:
                        # Face is too far RIGHT — strafe right while moving forward
                        got.mecanum_move_xyz(x_speed=strafe_spd, y_speed=fwd_spd, z_speed=0)
                    else:
                        # Face is centered but still small (far) — drive straight forward
                        got.mecanum_move_xyz(x_speed=0, y_speed=fwd_spd, z_speed=0)
                else:
                    # Face is centered AND large enough — arrived!
                    got.mecanum_stop()
                    print(f"Reached {target_name}!")
                    break

        got.mecanum_stop()

    except KeyboardInterrupt:
        print("Interrupted.")
        got.mecanum_stop()
