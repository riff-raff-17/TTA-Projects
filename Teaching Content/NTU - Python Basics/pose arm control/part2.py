"""
PART 2 — Hand detection + skeleton
====================================
Goal: load MediaPipe HandLandmarker, detect your hand, and draw the skeleton.

New concepts vs Part 1:
  - Downloading a model file
  - Creating a MediaPipe landmarker
  - Converting a cv2 frame → mp.Image
  - Drawing landmarks and connections onto the frame

What you'll see: your webcam feed with a green hand skeleton overlaid.
Press Q to quit.
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request
import os

# ---------------------------------------------------------------------------
# Download model
# ---------------------------------------------------------------------------
HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    print("Done.")

# ---------------------------------------------------------------------------
# Hand skeleton connections (pairs of landmark indices)
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (5,9),(9,10),(10,11),(11,12),      # middle
    (9,13),(13,14),(14,15),(15,16),    # ring
    (13,17),(17,18),(18,19),(19,20),   # pinky
    (0,17),                            # palm base
]

def draw_hand_skeleton(frame, landmarks, img_w, img_h):
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 5, (0, 220, 0), -1)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
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
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=__import__("cv2").cvtColor(frame, __import__("cv2").COLOR_BGR2RGB),
    )

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        draw_hand_skeleton(frame, result.hand_landmarks[0], img_w, img_h)
        cv2.putText(frame, "Hand detected!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    cv2.imshow("Part 2 — Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hand_landmarker.close()
cap.release()
cv2.destroyAllWindows()