"""
=============================================================
PART 0 — What Even Is MediaPipe?
=============================================================
This script is purely educational.

The goal is to answer three questions:
  1. What data does MediaPipe actually give me?
  2. What do the numbers mean?
  3. How do I pull out specific landmarks I care about?

Run:
    python part0_mediapipe_intro.py

Press 1 / 2 / 3 / 4 to switch between four "modes" that each
show you a different slice of the data.
=============================================================
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import os
import math

# ---------------------------------------------------------------------------
# Model download (same as Part 1)
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not os.path.exists(MODEL_PATH):
    print(f"Downloading model to '{MODEL_PATH}'...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,  # one hand is enough to learn with
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# The landmarks — 21 points
# ---------------------------------------------------------------------------
#
#   This is the most important thing to memorise.
#   Every finger has 4 landmarks: MCP (knuckle) -> PIP -> DIP -> TIP

LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_MCP",
    6: "INDEX_PIP",
    7: "INDEX_DIP",
    8: "INDEX_TIP",
    9: "MIDDLE_MCP",
    10: "MIDDLE_PIP",
    11: "MIDDLE_DIP",
    12: "MIDDLE_TIP",
    13: "RING_MCP",
    14: "RING_PIP",
    15: "RING_DIP",
    16: "RING_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [1, 5, 9, 13, 17]  # base knuckles (good "is finger up?" reference)
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]

# ---------------------------------------------------------------------------
# Tiny utilities
# ---------------------------------------------------------------------------


def px(lm, w, h):
    """Normalised landmark -> pixel (x, y)."""
    return (int(lm.x * w), int(lm.y * h))


def dist(lm_a, lm_b):
    """Euclidean distance between two landmarks in normalised space."""
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)


def draw_skeleton(frame, lms, w, h, color=(0, 200, 255)):
    """Draw bones + joints onto frame."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px(lms[a], w, h), px(lms[b], w, h), color, 2, cv2.LINE_AA)
    for i, lm in enumerate(lms):
        r = 7 if i in FINGER_TIPS else 4
        c = (0, 255, 120) if i in FINGER_TIPS else (255, 255, 255)
        cv2.circle(frame, px(lm, w, h), r, c, -1, cv2.LINE_AA)


def label(frame, text, pos, color=(220, 220, 220), scale=0.45, thickness=1):
    cv2.putText(
        frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def box(frame, x, y, w, h, color=(0, 0, 0), alpha=0.6):
    """Semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ---------------------------------------------------------------------------
# MODE 1 — Raw numbers
# ---------------------------------------------------------------------------
# Shows exactly what MediaPipe returns: a list of 21 (x, y, z) floats.
# x and y are 0-1 (fraction of frame width/height). z is depth vs wrist.
# ---------------------------------------------------------------------------
def mode_raw_numbers(frame, lms, w, h):
    draw_skeleton(frame, lms, w, h)

    # Panel on the right side
    panel_x, panel_y, line_h = w - 310, 50, 17
    box(frame, panel_x - 6, panel_y - 16, 300, 21 * line_h + 20)
    label(
        frame,
        "Raw landmark data (x, y, z)",
        (panel_x, panel_y - 2),
        color=(0, 255, 180),
        scale=0.4,
    )

    for i, lm in enumerate(lms):
        name = LANDMARK_NAMES[i]
        text = f"{i:>2}  {name:<12} ({lm.x:.2f}, {lm.y:.2f}, {lm.z:+.2f})"
        label(frame, text, (panel_x, panel_y + (i + 1) * line_h), scale=0.33)

    # Callout: explain the coordinate system
    box(frame, 8, h - 75, 420, 65)
    label(frame, "x=0 is LEFT edge,  x=1 is RIGHT edge", (14, h - 58), scale=0.4)
    label(frame, "y=0 is TOP edge,   y=1 is BOTTOM edge", (14, h - 40), scale=0.4)
    label(
        frame,
        "z is depth vs wrist: negative = closer to camera",
        (14, h - 22),
        scale=0.4,
    )


# ---------------------------------------------------------------------------
# MODE 2 — Spotlight a single landmark
# ---------------------------------------------------------------------------
# Pick one landmark (INDEX_TIP = 8) and watch it closely.
# This is the kind of focus you'll apply to each gesture you build.
# ---------------------------------------------------------------------------
SPOTLIGHT_IDX = 8  # change this number and re-run to explore others


def mode_spotlight(frame, lms, w, h):
    draw_skeleton(frame, lms, w, h, color=(80, 80, 80))

    lm = lms[SPOTLIGHT_IDX]
    name = LANDMARK_NAMES[SPOTLIGHT_IDX]
    pt = px(lm, w, h)

    # Crosshair on the spotlighted landmark
    cv2.circle(frame, pt, 18, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (pt[0] - 25, pt[1]), (pt[0] + 25, pt[1]), (0, 255, 255), 1)
    cv2.line(frame, (pt[0], pt[1] - 25), (pt[0], pt[1] + 25), (0, 255, 255), 1)

    # Projection lines to the edges so you can see relative position
    cv2.line(frame, (0, pt[1]), (pt[0], pt[1]), (0, 150, 150), 1)  # left
    cv2.line(frame, (pt[0], 0), (pt[0], pt[1]), (0, 150, 150), 1)  # top
    label(frame, f"x={lm.x:.3f}", (4, pt[1] - 4), color=(0, 220, 220), scale=0.4)
    label(frame, f"y={lm.y:.3f}", (pt[0] + 4, 14), color=(0, 220, 220), scale=0.4)

    # Info panel
    box(frame, 8, 38, 340, 80)
    label(
        frame,
        f"Spotlight: #{SPOTLIGHT_IDX} — {name}",
        (14, 58),
        color=(0, 255, 255),
        scale=0.55,
        thickness=1,
    )
    label(frame, f"x = {lm.x:.4f}  (0=left,  1=right)", (14, 80), scale=0.4)
    label(frame, f"y = {lm.y:.4f}  (0=top,   1=bottom)", (14, 95), scale=0.4)
    label(frame, f"z = {lm.z:+.4f}  (neg = closer to cam)", (14, 110), scale=0.4)

    box(frame, 8, h - 40, 370, 30)
    label(
        frame,
        "Edit SPOTLIGHT_IDX in the script to watch a different landmark",
        (14, h - 20),
        scale=0.38,
        color=(180, 180, 100),
    )


# ---------------------------------------------------------------------------
# MODE 3 — Distances between landmarks
# ---------------------------------------------------------------------------
# Distance is fundamental to gestures like "pinch" and "fist".
# Here we show the distance from THUMB_TIP (4) to each other fingertip,
# and draw a bar so you can see how it changes as you move your fingers.
# ---------------------------------------------------------------------------
def mode_distances(frame, lms, w, h):
    draw_skeleton(frame, lms, w, h)

    thumb_tip = lms[4]

    box(frame, 8, 38, 310, 160)
    label(
        frame,
        "Distance: THUMB_TIP -> each fingertip",
        (14, 56),
        color=(255, 200, 0),
        scale=0.42,
    )

    bar_max_w = 220
    for i, (tip_idx, fname) in enumerate(zip(FINGER_TIPS, FINGER_NAMES)):
        d = dist(thumb_tip, lms[tip_idx])
        bar_w = int(min(d / 0.5, 1.0) * bar_max_w)  # 0.5 = "fully open" reference
        y_row = 80 + i * 24

        label(frame, f"{fname:<7} {d:.3f}", (14, y_row + 10), scale=0.4)

        # Background bar
        cv2.rectangle(
            frame, (120, y_row), (120 + bar_max_w, y_row + 14), (50, 50, 50), -1
        )
        # Filled bar — green when close (pinch), red when far
        fill_color = (
            int(255 * (d / 0.5)),  # R
            int(255 * (1 - d / 0.5)),  # G
            0,
        )
        fill_color = tuple(max(0, min(255, c)) for c in fill_color)
        cv2.rectangle(frame, (120, y_row), (120 + bar_w, y_row + 14), fill_color, -1)

    # Draw lines from thumb tip to each fingertip
    for tip_idx in FINGER_TIPS[1:]:  # skip thumb->thumb
        d = dist(thumb_tip, lms[tip_idx])
        thickness = max(1, int((0.3 - min(d, 0.3)) / 0.3 * 4))
        cv2.line(
            frame,
            px(thumb_tip, w, h),
            px(lms[tip_idx], w, h),
            (255, 200, 0),
            thickness,
            cv2.LINE_AA,
        )

    box(frame, 8, h - 40, 340, 30)
    label(
        frame,
        "Try: pinch each finger to thumb — watch bars go green",
        (14, h - 20),
        scale=0.38,
        color=(180, 180, 100),
    )


# ---------------------------------------------------------------------------
# MODE 4 — "Is this finger up?" logic
# ---------------------------------------------------------------------------
# The simplest useful gesture: is a finger extended or curled?
# Rule: fingertip y < MCP knuckle y  =  finger is up
# (Remember y=0 is the TOP of the frame, so a smaller y = higher up)
# ---------------------------------------------------------------------------
def is_finger_up(lms, tip_idx, mcp_idx):
    """True if the fingertip is above its base knuckle (finger extended)."""
    return lms[tip_idx].y < lms[mcp_idx].y


def mode_finger_up(frame, lms, w, h):
    states = [is_finger_up(lms, FINGER_TIPS[i], FINGER_MCPS[i]) for i in range(5)]

    # Draw each finger bone in a highlight colour based on state
    finger_connections = [
        [(0, 1), (1, 2), (2, 3), (3, 4)],
        [(0, 5), (5, 6), (6, 7), (7, 8)],
        [(0, 9), (9, 10), (10, 11), (11, 12)],
        [(0, 13), (13, 14), (14, 15), (15, 16)],
        [(0, 17), (17, 18), (18, 19), (19, 20)],
    ]
    for i, (connections, up) in enumerate(zip(finger_connections, states)):
        color = (0, 255, 100) if up else (60, 60, 200)
        for a, b in connections:
            cv2.line(frame, px(lms[a], w, h), px(lms[b], w, h), color, 3, cv2.LINE_AA)

    # Joints
    for i, lm in enumerate(lms):
        cv2.circle(frame, px(lm, w, h), 5, (255, 255, 255), -1)

    # Info panel
    box(frame, 8, 38, 300, 200)
    label(
        frame,
        "Finger up/down detector",
        (14, 58),
        color=(0, 255, 180),
        scale=0.52,
        thickness=1,
    )
    label(
        frame,
        "Rule: tip.y < mcp.y  ->  finger is UP",
        (14, 78),
        scale=0.38,
        color=(180, 180, 180),
    )

    for i, (fname, up) in enumerate(zip(FINGER_NAMES, states)):
        state_str = "UP  " if up else "DOWN"
        color = (0, 255, 100) if up else (100, 100, 255)
        label(
            frame,
            f"{fname:<7}  {state_str}  "
            f"(tip.y={lms[FINGER_TIPS[i]].y:.2f}  mcp.y={lms[FINGER_MCPS[i]].y:.2f})",
            (14, 105 + i * 20),
            color=color,
            scale=0.38,
        )

    # Show the count
    n_up = sum(states)
    label(
        frame,
        f"{n_up} finger(s) up",
        (14, 225),
        scale=0.45,
        color=(255, 255, 0),
        thickness=1,
    )

    box(frame, 8, h - 40, 370, 30)
    label(
        frame,
        "This simple rule is the foundation of ALL gesture detection",
        (14, h - 20),
        scale=0.38,
        color=(180, 180, 100),
    )


# ---------------------------------------------------------------------------
# Mode metadata
# ---------------------------------------------------------------------------
MODES = {
    ord("1"): ("1: Raw Numbers", mode_raw_numbers),
    ord("2"): ("2: Spotlight", mode_spotlight),
    ord("3"): ("3: Distances", mode_distances),
    ord("4"): ("4: Finger Up/Down", mode_finger_up),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    current_mode_key = ord("1")
    current_mode_name, current_mode_fn = MODES[current_mode_key]

    print("=== Part 0: What Even Is MediaPipe? ===")
    print("  1 — Raw numbers (all 21 landmarks)")
    print("  2 — Spotlight a single landmark")
    print("  3 — Distances (foundation of pinch)")
    print("  4 — Finger up/down (foundation of all gestures)")
    print("  Q — quit")
    print("========================================\n")
    print("Suggested order: work through 1 -> 2 -> 3 -> 4")
    print("In mode 2, edit SPOTLIGHT_IDX at the top of the script to watch")
    print("different landmarks.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]  # just the first hand
            current_mode_fn(frame, lms, w, h)
        else:
            label(
                frame,
                "Show your hand to the camera...",
                (w // 2 - 160, h // 2),
                color=(100, 100, 255),
                scale=0.7,
                thickness=2,
            )

        # Mode switcher bar at top
        box(frame, 0, 0, w, 22, color=(20, 20, 20), alpha=0.8)
        mode_bar = "  |  ".join(f"[{chr(k)}] {name}" for k, (name, _) in MODES.items())
        label(
            frame,
            mode_bar + "  |  [Q] quit",
            (8, 15),
            color=(160, 160, 160),
            scale=0.38,
        )

        # Current mode highlight
        label(
            frame,
            f"MODE: {current_mode_name}",
            (w - 220, 15),
            color=(0, 220, 255),
            scale=0.4,
            thickness=1,
        )

        cv2.imshow("Part 0 - MediaPipe Intro", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in MODES:
            current_mode_key = key
            current_mode_name, current_mode_fn = MODES[key]
            print(f"Switched to {current_mode_name}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
