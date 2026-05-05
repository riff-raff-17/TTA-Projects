"""
PART 4 — Pose detection + elbow angle → joint angles
======================================================
Goal: load PoseLandmarker, detect your pose, compute the elbow bend angle,
      and map it to J2/J3 values. Prints joint angles to terminal.
      No hand detection yet — just the arm.

New concepts vs Part 3:
  - PoseLandmarker (different model to HandLandmarker)
  - calculate_angle() using dot products
  - np.interp() to remap a raw angle into a servo range

What you'll see: webcam with your arm skeleton drawn and a HUD showing
                 live J2/J3 values. Terminal prints them when they change.
Press Q to quit.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading pose landmark model...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    print("Done.")

# Pose landmark indices (right arm)
RIGHT_SHOULDER = 12
RIGHT_ELBOW    = 14
RIGHT_WRIST    = 16

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),            # right arm
    (11, 23), (12, 24), (23, 24),  # torso
]

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def lm_to_xy(lm):
    return (lm.x, lm.y)

def calculate_angle(a, b, c):
    """Angle at b between vectors b→a and b→c. Returns degrees [0, 180]."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def pose_to_joints(pose_landmarks):
    """
    Elbow bend angle → J2 and J3.
    Arm down  (elbow ~180°) → total -90°  (J2 = -45, J3 = -45)
    Arm level (elbow ~90°)  → total   0°  (J2 =   0, J3 =   0)
    Arm up    (elbow ~0°)   → total +90°  (J2 = +45, J3 = +45)
    """
    shoulder = pose_landmarks[RIGHT_SHOULDER]
    elbow    = pose_landmarks[RIGHT_ELBOW]
    wrist    = pose_landmarks[RIGHT_WRIST]

    elbow_angle    = calculate_angle(lm_to_xy(shoulder), lm_to_xy(elbow), lm_to_xy(wrist))
    total_vertical = float(np.interp(elbow_angle, [0, 180], [90, -90]))
    total_vertical = float(np.clip(total_vertical, -90, 90))
    return total_vertical / 2.0, total_vertical / 2.0

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_pose(frame, landmarks, img_w, img_h):
    pts = {i: (int(lm.x * img_w), int(lm.y * img_h)) for i, lm in enumerate(landmarks)}
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for idx in [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, 11, 23, 24]:
        if idx in pts:
            cv2.circle(frame, pts[idx], 6, (0, 220, 0), -1)

def draw_hud(frame, j2, j3):
    lines = [
        f"Joint 2: {j2:+6.1f} deg",
        f"Joint 3: {j3:+6.1f} deg",
        f"Total:   {j2+j3:+6.1f} deg",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 50 + i * 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)

# ---------------------------------------------------------------------------
# Smoothing
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
pose_options = vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

smoothers   = [EMA(0.25), EMA(0.25)]
prev_j2, prev_j3 = None, None

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
    result = pose_landmarker.detect(mp_image)

    if result.pose_landmarks:
        lms = result.pose_landmarks[0]
        draw_pose(frame, lms, img_w, img_h)

        j2_raw, j3_raw = pose_to_joints(lms)
        j2 = smoothers[0].update(j2_raw)
        j3 = smoothers[1].update(j3_raw)

        draw_hud(frame, j2, j3)

        # Print to terminal when joint angles change by more than 1°
        if prev_j2 is None or abs(j2 - prev_j2) > 1.0 or abs(j3 - prev_j3) > 1.0:
            print(f"J2: {j2:+6.1f}°  J3: {j3:+6.1f}°  Total: {j2+j3:+6.1f}°")
            prev_j2, prev_j3 = j2, j3
    else:
        cv2.putText(frame, "No pose detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

    cv2.imshow("Part 4 — Pose + Joint Angles", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pose_landmarker.close()
cap.release()
cv2.destroyAllWindows()