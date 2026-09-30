import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# ---------------------------------------------------------------------------
# Model downloads
# ---------------------------------------------------------------------------
POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

for _path, _url, _label in [
    (POSE_MODEL_PATH, POSE_MODEL_URL, "pose"),
    (HAND_MODEL_PATH, HAND_MODEL_URL, "hand"),
]:
    if not os.path.exists(_path):
        print(f"Downloading {_label} landmark model...")
        urllib.request.urlretrieve(_url, _path)
        print("Done.")

# ---------------------------------------------------------------------------
# Landmark indices (right arm — mirrored frame means this is user's right)
# ---------------------------------------------------------------------------
RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16
RIGHT_HIP = 24

# Hand landmark indices (HandLandmarker)
HAND_THUMB_TIP = 4
HAND_INDEX_TIP = 8


# ---------------------------------------------------------------------------
# Robot command placeholders
# ---------------------------------------------------------------------------
def set_joint2(angle: float):
    """Joint 2: up/down contribution. Range -45–+45°."""
    print(f"Robot: JOINT2 -> {angle:.1f}°")


def set_joint3(angle: float):
    """Joint 3: up/down contribution. Range ~0–90°."""
    print(f"Robot: JOINT3 -> {angle:.1f}°")


def set_gripper(open_pct: float):
    """Gripper: 0.0 = fully closed, 1.0 = fully open."""
    print(f"Robot: GRIPPER -> {open_pct * 100:.0f}% open")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """
    Angle at point b, formed by vectors b->a and b->c.
    Points are (x, y) or (x, y, z) tuples/arrays.
    Returns angle in degrees [0, 180].
    """
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def lm_to_xy(lm):
    """Return (x, y) from a NormalizedLandmark."""
    return (lm.x, lm.y)


def lm_dist(a, b):
    """Euclidean distance between two landmarks in normalised coords."""
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ---------------------------------------------------------------------------
# Pose → servo mapping
# ---------------------------------------------------------------------------
def pose_to_joints(pose_landmarks):
    """
    Derive joint angles from pose landmarks.

    Joint 2 + Joint 3 (up/down, summed, range -90 to +90 total):
        Arm straight/down (elbow ~180°) → total -90° (J2=-45, J3=-45).
        Arm horizontal    (elbow ~90°)  → total  0°  (J2=0,   J3=0).
        Arm bent fully up (elbow ~0°)   → total +90° (J2=+45, J3=+45).
    """
    shoulder = pose_landmarks[RIGHT_SHOULDER]
    elbow = pose_landmarks[RIGHT_ELBOW]
    wrist = pose_landmarks[RIGHT_WRIST]

    elbow_angle = calculate_angle(lm_to_xy(shoulder), lm_to_xy(elbow), lm_to_xy(wrist))
    total_vertical = float(np.interp(elbow_angle, [0, 180], [90, -90]))
    total_vertical = float(np.clip(total_vertical, -90, 90))
    return total_vertical / 2.0, total_vertical / 2.0


def hand_to_gripper(hand_landmarks):
    """
    Binary gripper state from HandLandmarker landmarks.
    Uses true thumb tip (4) and index tip (8).
    Normalised by the hand's own wrist-to-middle-mcp span so it's
    distance-invariant. Pinching = CLOSED, apart = OPEN.
    """
    thumb_tip = hand_landmarks[HAND_THUMB_TIP]
    index_tip = hand_landmarks[HAND_INDEX_TIP]
    wrist = hand_landmarks[0]
    mid_mcp = hand_landmarks[9]  # middle finger MCP — stable reference

    hand_ref = lm_dist(wrist, mid_mcp) + 1e-6
    pinch_dist = lm_dist(thumb_tip, index_tip)
    pinch_ratio = pinch_dist / hand_ref

    # Ratio < ~0.4 → pinching (closed); > ~0.4 → open
    GRIPPER_THRESHOLD = 0.4
    return 1.0 if pinch_ratio > GRIPPER_THRESHOLD else 0.0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),  # left arm
    (12, 14),
    (14, 16),  # right arm
    (11, 23),
    (12, 24),
    (23, 24),  # torso
]


def draw_pose(frame, landmarks, img_w, img_h):
    pts = {i: (int(lm.x * img_w), int(lm.y * img_h)) for i, lm in enumerate(landmarks)}
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for idx in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, 11, 23, 24]:
        if idx in pts:
            cv2.circle(frame, pts[idx], 6, (0, 220, 0), -1)


def draw_hand(frame, hand_landmarks, img_w, img_h, gripper_open):
    """Highlight just the thumb and index tips used for pinch detection."""
    color = (0, 200, 255) if gripper_open > 0.5 else (0, 80, 255)
    for idx in [HAND_THUMB_TIP, HAND_INDEX_TIP]:
        lm = hand_landmarks[idx]
        cx, cy = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(frame, (cx, cy), 8, color, -1)
    # Line between tips to visualise pinch distance
    t = hand_landmarks[HAND_THUMB_TIP]
    i = hand_landmarks[HAND_INDEX_TIP]
    cv2.line(
        frame,
        (int(t.x * img_w), int(t.y * img_h)),
        (int(i.x * img_w), int(i.y * img_h)),
        color,
        2,
    )


def draw_hud(frame, j2, j3, gripper):
    lines = [
        f"Joint 2 (lift):  {j2:5.1f} deg",
        f"Joint 3 (lift):  {j3:5.1f} deg",
        f"Total vertical:  {j2+j3:5.1f} deg",
        f"Gripper:         {'OPEN' if gripper > 0.5 else 'CLOSED'}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (20, 40 + i * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0),
            2,
        )


# ---------------------------------------------------------------------------
# Smoothing — simple exponential moving average
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    # --- Pose landmarker ---
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # --- Hand landmarker ---
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # One EMA smoother per joint (gripper is binary — no smoothing needed)
    smoothers = [EMA(alpha=0.25) for _ in range(2)]  # j2, j3

    prev_commands = (None, None, None)
    gripper = 0.0  # default closed if hand not detected

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )

        pose_result = pose_landmarker.detect(mp_image)
        hand_result = hand_landmarker.detect(mp_image)

        pose_label = "No pose detected"

        # --- Hand: update gripper if hand visible ---
        if hand_result.hand_landmarks:
            hand_lms = hand_result.hand_landmarks[0]
            gripper = hand_to_gripper(hand_lms)
            draw_hand(frame, hand_lms, img_w, img_h, gripper)

        # --- Pose: update joints if pose visible ---
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            draw_pose(frame, pose_lms, img_w, img_h)

            j2_raw, j3_raw = pose_to_joints(pose_lms)
            j2 = smoothers[0].update(j2_raw)
            j3 = smoothers[1].update(j3_raw)

            commands = (j2, j3, gripper)

            thresholds = (1.0, 1.0, 0.5)
            if prev_commands[0] is None or any(
                abs(commands[i] - prev_commands[i]) > thresholds[i] for i in range(3)
            ):
                set_joint2(j2)
                set_joint3(j3)
                set_gripper(gripper)
                prev_commands = commands

            draw_hud(frame, j2, j3, gripper)
            pose_label = "Pose detected — mirroring arm"

        cv2.putText(
            frame,
            pose_label,
            (20, img_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )
        cv2.imshow("Pose Robot Control — press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pose_landmarker.close()
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
