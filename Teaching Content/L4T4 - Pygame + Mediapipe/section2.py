"""
=============================================================
PART 2 — The HandController: Turning Landmarks into Actions
=============================================================
Goal: Build a reusable HandController class that converts raw
MediaPipe landmarks into clean game-ready outputs:

    controller.gesture   ->  "IDLE" | "THRUST" | "SHOOT" | "BRAKE"
    controller.steering  ->  float from -1.0 (left) to +1.0 (right)
    controller.tilt_deg  ->  raw tilt angle in degrees (useful for debug)

Nothing in this file knows about Pygame or game logic.
The HandController is a self-contained "translator" that we
will import unchanged into every future part.

New concepts vs Part 1:
  - Extracting a meaningful angle from two landmarks
  - "Finger up" logic as a building block for gestures
  - A dead zone to stop drift when the hand is roughly flat
  - A named gesture system you can extend freely

Run:
    python part2_hand_controller.py

Controls:
    Q  — quit
    D  — toggle debug overlay
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import time
import math

# ---------------------------------------------------------------------------
# Model (identical to Part 1 — safe to run again, skips if already downloaded)
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to '{MODEL_PATH}'...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

# Landmark index constants — same as Part 1, kept here so HandController
# is self-contained when we import it later.
WRIST      = 0
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_TIP  = 8
MIDDLE_TIP = 12
RING_TIP   = 16
PINKY_TIP  = 20

FINGER_TIPS  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS  = [2, INDEX_MCP, 9, 13, 17]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ---------------------------------------------------------------------------
# HandController
# ---------------------------------------------------------------------------
# This is the class you will import into every future part.
# Its job: take a list of 21 landmarks, produce game-ready values.
#
# Design principles:
#   1. It never touches the camera or the detector — it only processes
#      landmarks that are handed to it from outside.
#   2. All outputs are simple Python types (str, float, bool) so game
#      code never has to touch MediaPipe types directly.
#   3. If no hand is detected, it returns safe neutral values.
# ---------------------------------------------------------------------------

class HandController:

    # Gesture names — defined as class-level constants so game code can do:
    #   if controller.gesture == HandController.THRUST: ...
    IDLE   = "IDLE"
    THRUST = "THRUST"
    SHOOT  = "SHOOT"
    BRAKE  = "BRAKE"

    # Steering dead zone: if |tilt| is below this angle (degrees), treat
    # steering as exactly 0.  Prevents the ship drifting when hand is flat.
    DEAD_ZONE_DEG = 10.0

    # The tilt angle (degrees) that maps to full -1 or +1 steering.
    # Tilt your hand less than this to steer gently.
    MAX_TILT_DEG  = 40.0

    def __init__(self):
        # These are the outputs game code will read every frame.
        self.gesture  = HandController.IDLE  # current gesture string
        self.steering = 0.0                  # -1.0 (left) … +1.0 (right)
        self.tilt_deg = 0.0                  # raw angle, handy for debug

        # Internal: store the last landmarks so draw helpers can use them
        self._landmarks = None

    # ------------------------------------------------------------------
    # update() — call this once per frame with the new landmark list.
    # Pass None when no hand is detected.
    # ------------------------------------------------------------------
    def update(self, landmarks):
        self._landmarks = landmarks

        if landmarks is None:
            self.gesture  = HandController.IDLE
            self.steering = 0.0
            self.tilt_deg = 0.0
            return

        # 1. Compute steering from hand tilt
        self.tilt_deg = self._compute_tilt(landmarks)
        self.steering = self._tilt_to_steering(self.tilt_deg)

        # 2. Recognise gesture from finger configuration
        self.gesture  = self._recognise_gesture(landmarks)

    # ------------------------------------------------------------------
    # Steering: angle of the line from WRIST → INDEX_MCP
    # ------------------------------------------------------------------
    # Why these two landmarks?
    #   - The wrist is the most stable point on the hand.
    #   - INDEX_MCP (the index knuckle) gives a clean "top of hand" direction.
    #   - Together they define the hand's tilt axis reliably even when
    #     fingers are moving.
    #
    # The angle is measured from horizontal (0°):
    #   tilting right  → positive degrees → positive steering
    #   tilting left   → negative degrees → negative steering
    # ------------------------------------------------------------------
    def _compute_tilt(self, lms):
        wrist     = lms[WRIST]
        index_mcp = lms[INDEX_MCP]

        # dx / dy in normalised space
        # Note: we subtract wrist from index_mcp so the vector points
        # "up the hand" (from palm toward knuckles).
        dx = index_mcp.x - wrist.x
        dy = index_mcp.y - wrist.y   # remember: y increases downward

        # atan2 gives us the angle of the vector in radians.
        # We negate dy because screen y is inverted vs standard maths.
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalise to -90..+90 range centred on "hand pointing up" (90°).
        # A perfectly vertical hand (pointing straight up) = 0° tilt.
        tilt = angle_deg - 90.0

        # Clamp to a sensible range
        return max(-90.0, min(90.0, tilt))

    def _tilt_to_steering(self, tilt_deg):
        # Apply dead zone
        if abs(tilt_deg) < self.DEAD_ZONE_DEG:
            return 0.0

        # Remove the dead zone from the remaining range so the transition
        # at the dead zone boundary is smooth rather than a sudden jump.
        sign   = 1.0 if tilt_deg > 0 else -1.0
        amount = abs(tilt_deg) - self.DEAD_ZONE_DEG
        range_ = self.MAX_TILT_DEG - self.DEAD_ZONE_DEG

        # Clamp and normalise to -1..+1
        return sign * min(amount / range_, 1.0)

    # ------------------------------------------------------------------
    # Gesture recognition
    # ------------------------------------------------------------------
    # We build up from the simplest primitive: is_finger_up().
    # Each gesture is just a specific combination of fingers up/down.
    #
    # Gesture map:
    #   All fingers curled (fist)          -> THRUST
    #   Index + Middle up, rest down       -> SHOOT
    #   All fingers open                   -> BRAKE
    #   Anything else                      -> IDLE
    # ------------------------------------------------------------------
    def _is_finger_up(self, lms, tip_idx, mcp_idx):
        """True if the fingertip is above (smaller y than) its base knuckle."""
        return lms[tip_idx].y < lms[mcp_idx].y

    def _finger_states(self, lms):
        """Return a list of 5 bools: [thumb_up, index_up, middle_up, ring_up, pinky_up]."""
        return [
            self._is_finger_up(lms, FINGER_TIPS[i], FINGER_MCPS[i])
            for i in range(5)
        ]

    def _recognise_gesture(self, lms):
        thumb, index, middle, ring, pinky = self._finger_states(lms)

        # THRUST — make a fist (all fingers down)
        if not any([thumb, index, middle, ring, pinky]):
            return HandController.THRUST

        # SHOOT — index and middle up, ring and pinky down (finger guns)
        if index and middle and not ring and not pinky:
            return HandController.SHOOT

        # BRAKE — open hand (all fingers up)
        if all([index, middle, ring, pinky]):
            return HandController.BRAKE

        return HandController.IDLE


# ---------------------------------------------------------------------------
# Detector setup (same pattern as Part 1)
# ---------------------------------------------------------------------------
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# Drawing helpers (pure OpenCV, same style as Part 1)
# ---------------------------------------------------------------------------

def lm_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def draw_skeleton(frame, lms, w, h):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px(lms[a], w, h), lm_px(lms[b], w, h),
                 (0, 200, 255), 2, cv2.LINE_AA)
    for i, lm in enumerate(lms):
        r = 7 if i in FINGER_TIPS else 4
        c = (0, 255, 150) if i in FINGER_TIPS else (255, 255, 255)
        cv2.circle(frame, lm_px(lm, w, h), r, c, -1, cv2.LINE_AA)
        cv2.circle(frame, lm_px(lm, w, h), r, (0, 0, 0), 1, cv2.LINE_AA)

def draw_tilt_arrow(frame, lms, w, h, steering):
    """Draw a line showing the wrist→index_mcp tilt axis."""
    wrist_pt = lm_px(lms[WRIST], w, h)
    mcp_pt   = lm_px(lms[INDEX_MCP], w, h)

    # Colour shifts from green (neutral) toward red (full tilt)
    intensity = int(abs(steering) * 255)
    color     = (0, 255 - intensity, intensity)

    cv2.arrowedLine(frame, wrist_pt, mcp_pt, color, 3,
                    cv2.LINE_AA, tipLength=0.3)

def draw_steering_bar(frame, steering, x, y, bar_w=200, bar_h=20):
    """Horizontal bar: left half = left steer, right half = right steer."""
    mid = x + bar_w // 2

    # Background
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    # Centre line
    cv2.line(frame, (mid, y), (mid, y + bar_h), (120, 120, 120), 1)

    # Fill from centre toward the steered side
    fill_w = int(abs(steering) * (bar_w // 2))
    if steering > 0:
        cv2.rectangle(frame, (mid, y + 2), (mid + fill_w, y + bar_h - 2),
                      (0, 200, 255), -1)
    elif steering < 0:
        cv2.rectangle(frame, (mid - fill_w, y + 2), (mid, y + bar_h - 2),
                      (0, 200, 255), -1)

    cv2.putText(frame, f"Steering: {steering:+.2f}",
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

def draw_gesture_panel(frame, controller, x, y):
    """Show the current gesture with a colour-coded label."""
    GESTURE_COLORS = {
        HandController.IDLE:   (150, 150, 150),
        HandController.THRUST: (0,   200, 255),
        HandController.SHOOT:  (0,   80,  255),
        HandController.BRAKE:  (0,   255, 100),
    }
    GESTURE_HINTS = {
        HandController.IDLE:   "neutral hand position",
        HandController.THRUST: "make a fist",
        HandController.SHOOT:  "index + middle finger up",
        HandController.BRAKE:  "open hand (all fingers up)",
    }

    color = GESTURE_COLORS.get(controller.gesture, (255, 255, 255))

    # Background panel
    cv2.rectangle(frame, (x - 6, y - 20), (x + 280, y + 120), (0, 0, 0), -1)

    cv2.putText(frame, "GESTURE", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, controller.gesture, (x, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"({GESTURE_HINTS[controller.gesture]})",
                (x, y + 52), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (150, 150, 150), 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (x, y + 62), (x + 274, y + 62), (50, 50, 50), 1)

    # All four gestures listed as a reminder
    cv2.putText(frame, "Gestures:", (x, y + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1, cv2.LINE_AA)
    hints = [
        ("THRUST", "fist"),
        ("SHOOT",  "index+middle up"),
        ("BRAKE",  "open hand"),
        ("IDLE",   "anything else"),
    ]
    for i, (name, hint) in enumerate(hints):
        c = GESTURE_COLORS[name]
        cv2.putText(frame, f"  {name:<8} {hint}",
                    (x, y + 90 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1, cv2.LINE_AA)

def draw_debug_overlay(frame, controller, lms, w, h):
    """Full debug view: skeleton + tilt arrow + tilt angle readout."""
    draw_skeleton(frame, lms, w, h)
    draw_tilt_arrow(frame, lms, w, h, controller.steering)

    wrist_pt = lm_px(lms[WRIST], w, h)
    cv2.putText(frame, f"tilt: {controller.tilt_deg:+.1f} deg",
                (wrist_pt[0] + 10, wrist_pt[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

# ---------------------------------------------------------------------------
# FPS counter (same as Part 1, with the @property removed as you did)
# ---------------------------------------------------------------------------
class FPSCounter:
    def __init__(self, smoothing=20):
        self._times     = []
        self._smoothing = smoothing

    def tick(self):
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._smoothing:
            self._times.pop(0)

    def get_fps(self):
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    controller  = HandController()
    fps_counter = FPSCounter()
    show_debug  = True

    print("=== Part 2: HandController ===")
    print("  Gestures to try:")
    print("    Fist            -> THRUST")
    print("    Index+Middle up -> SHOOT")
    print("    Open hand       -> BRAKE")
    print("    Tilt left/right -> steering")
    print("  Q — quit   D — toggle debug overlay")
    print("==============================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(mp_image)

        # Feed landmarks (or None) into the controller
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            controller.update(lms)
        else:
            lms = None
            controller.update(None)

        # ---- Draw -------------------------------------------------------

        if lms is not None and show_debug:
            draw_debug_overlay(frame, controller, lms, w, h)
        elif lms is None:
            cv2.putText(frame, "Show your hand...",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (100, 100, 255), 2, cv2.LINE_AA)

        # Gesture panel (bottom-left)
        draw_gesture_panel(frame, controller, 14, h - 230)

        # Steering bar (bottom-left, above gesture panel)
        draw_steering_bar(frame, controller.steering, 14, h - 250)

        # FPS
        fps_counter.tick()
        cv2.putText(frame, f"FPS: {fps_counter.get_fps():.0f}",
                    (w - 90, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Key hints
        cv2.putText(frame, "[Q] quit   [D] debug overlay",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Part 2 - HandController", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nDone.")


if __name__ == "__main__":
    main()