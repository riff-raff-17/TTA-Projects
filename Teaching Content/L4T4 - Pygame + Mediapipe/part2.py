"""
=============================================================
PART 2 — The HandController
=============================================================
Goal: build a reusable HandController class that sits on top
of Part 1's foundation and converts raw landmarks into clean
game-ready values:

    controller.gesture   -> "IDLE" | "THRUST" | "SHOOT" | "BRAKE"
    controller.steering  -> float  -1.0 (left) .. +1.0 (right)
    controller.tilt_deg  -> raw tilt angle in degrees (for debug)

What this file adds on top of Part 1:
  - HandController class                (new)
  - draw_tilt_arrow()                   (new drawing helper)
  - draw_steering_bar()                 (new drawing helper)
  - draw_gesture_panel()                (new drawing helper)
  - main loop that shows all the above  (new)

What this file does NOT touch:
  - hand_common.py                      (unchanged)
  - part1_mediapipe_hello.py            (unchanged)

Depends on: hand_common.py (must be in the same folder)

Run:
    python part2_hand_controller.py

Controls:
    Q  — quit
    D  — toggle debug overlay
=============================================================
"""

import cv2
import mediapipe as mp
import math
from hand_common import (
    make_detector,
    draw_hand,
    FPSCounter,
    WRIST,
    INDEX_MCP,
    FINGER_TIPS,
    FINGER_MCPS,
    HAND_CONNECTIONS,
    lm_px,
)

# ---------------------------------------------------------------------------
# HandController
# ---------------------------------------------------------------------------
# This class is the only new concept in Part 2.
# Its job: take a list of 21 landmarks, return game-ready values.
#
# Design rules:
#   - Never touches the camera or detector directly.
#   - All outputs are plain Python types (str, float).
#   - Safe to call with None when no hand is visible.
#   - Will be imported unchanged by Part 3 onwards.
# ---------------------------------------------------------------------------


class HandController:

    # Gesture name constants — use these in game code instead of raw strings:
    #   if controller.gesture == HandController.THRUST: ...
    IDLE = "IDLE"
    THRUST = "THRUST"
    SHOOT = "SHOOT"
    BRAKE = "BRAKE"

    # Steering tuning
    DEAD_ZONE_DEG = 10.0  # tilt smaller than this -> steering = 0 (no drift)
    MAX_TILT_DEG = 40.0  # tilt at this angle     -> steering = ±1.0 (full)

    def __init__(self):
        self.gesture = HandController.IDLE
        self.steering = 0.0
        self.tilt_deg = 0.0

    def update(self, landmarks):
        """Call once per frame. Pass None when no hand is detected."""
        if landmarks is None:
            self.gesture = HandController.IDLE
            self.steering = 0.0
            self.tilt_deg = 0.0
            return

        self.tilt_deg = self._compute_tilt(landmarks)
        self.steering = self._tilt_to_steering(self.tilt_deg)
        self.gesture = self._recognise_gesture(landmarks)

    # ------------------------------------------------------------------
    # Steering — angle of WRIST -> INDEX_MCP vector
    # ------------------------------------------------------------------
    # These two landmarks sit on the rigid palm so the angle stays stable
    # even while fingers are moving.
    # Tilting right -> positive steering, tilting left -> negative.
    # ------------------------------------------------------------------
    def _compute_tilt(self, lms):
        wrist = lms[WRIST]
        index_mcp = lms[INDEX_MCP]
        dx = index_mcp.x - wrist.x
        dy = index_mcp.y - wrist.y  # y increases downward in image space
        # atan2 gives the vector angle; negate dy to flip to maths convention
        angle_deg = math.degrees(math.atan2(-dy, dx))
        # Shift so a perfectly upright hand = 0 tilt
        tilt = angle_deg - 90.0
        return max(-90.0, min(90.0, tilt))

    def _tilt_to_steering(self, tilt_deg):
        if abs(tilt_deg) < self.DEAD_ZONE_DEG:
            return 0.0
        sign = 1.0 if tilt_deg > 0 else -1.0
        amount = abs(tilt_deg) - self.DEAD_ZONE_DEG
        range_ = self.MAX_TILT_DEG - self.DEAD_ZONE_DEG
        return sign * min(amount / range_, 1.0)

    # ------------------------------------------------------------------
    # Gestures — combinations of fingers up/down
    # ------------------------------------------------------------------
    # Built on the same rule from Part 0 Mode 4:
    #   tip.y < mcp.y  =>  finger is extended (up)
    #
    #   THRUST  — fist (all fingers down)
    #   SHOOT   — index + middle up, ring + pinky down (finger guns)
    #   BRAKE   — all fingers up (open hand)
    #   IDLE    — anything else
    # ------------------------------------------------------------------
    def _is_finger_up(self, lms, tip_idx, mcp_idx):
        return lms[tip_idx].y < lms[mcp_idx].y

    def _finger_states(self, lms):
        return [
            self._is_finger_up(lms, FINGER_TIPS[i], FINGER_MCPS[i]) for i in range(5)
        ]

    def _recognise_gesture(self, lms):
        thumb, index, middle, ring, pinky = self._finger_states(lms)

        if not any([thumb, index, middle, ring, pinky]):
            return HandController.THRUST

        if index and middle and not ring and not pinky:
            return HandController.SHOOT

        if all([index, middle, ring, pinky]):
            return HandController.BRAKE

        return HandController.IDLE


# ---------------------------------------------------------------------------
# Part 2 drawing helpers — new additions, do not modify hand_common.py
# ---------------------------------------------------------------------------


def draw_tilt_arrow(frame, lms, w, h, steering):
    """Arrow from wrist to index knuckle, coloured by steering intensity."""
    wrist_pt = lm_px(lms[WRIST], w, h)
    mcp_pt = lm_px(lms[INDEX_MCP], w, h)
    intensity = int(abs(steering) * 255)
    color = (0, 255 - intensity, intensity)
    cv2.arrowedLine(frame, wrist_pt, mcp_pt, color, 3, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(
        frame,
        f"tilt: {lms[WRIST].x:.0f}",  # placeholder, real tilt below
        (wrist_pt[0] + 10, wrist_pt[1] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )


def draw_steering_bar(frame, controller, x, y, bar_w=220, bar_h=22):
    """Horizontal bar: centre = neutral, fills left or right with tilt."""
    mid = x + bar_w // 2
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
    cv2.line(frame, (mid, y), (mid, y + bar_h), (120, 120, 120), 1)

    fill_w = int(abs(controller.steering) * (bar_w // 2))
    if controller.steering > 0:
        cv2.rectangle(
            frame, (mid, y + 2), (mid + fill_w, y + bar_h - 2), (0, 200, 255), -1
        )
    elif controller.steering < 0:
        cv2.rectangle(
            frame, (mid - fill_w, y + 2), (mid, y + bar_h - 2), (0, 200, 255), -1
        )

    cv2.putText(
        frame,
        f"Steering: {controller.steering:+.2f}  "
        f"(tilt {controller.tilt_deg:+.1f} deg)",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def draw_gesture_panel(frame, controller, x, y):
    """Colour-coded gesture label + reference card."""
    COLORS = {
        HandController.IDLE: (150, 150, 150),
        HandController.THRUST: (0, 200, 255),
        HandController.SHOOT: (80, 80, 255),
        HandController.BRAKE: (0, 255, 100),
    }
    HINTS = {
        HandController.IDLE: "any neutral position",
        HandController.THRUST: "make a fist",
        HandController.SHOOT: "index + middle finger up",
        HandController.BRAKE: "open hand (all fingers up)",
    }
    color = COLORS[controller.gesture]

    cv2.rectangle(frame, (x - 6, y - 20), (x + 290, y + 115), (0, 0, 0), -1)
    cv2.putText(
        frame,
        "GESTURE",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        controller.gesture,
        (x, y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"({HINTS[controller.gesture]})",
        (x, y + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    cv2.line(frame, (x, y + 62), (x + 284, y + 62), (40, 40, 40), 1)

    ref = [
        ("THRUST", "fist"),
        ("SHOOT", "index+middle up"),
        ("BRAKE", "open hand"),
        ("IDLE", "anything else"),
    ]
    for i, (name, hint) in enumerate(ref):
        cv2.putText(
            frame,
            f"  {name:<8} {hint}",
            (x, y + 76 + i * 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            COLORS[name],
            1,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    detector = make_detector(num_hands=1)
    fps_counter = FPSCounter()
    controller = HandController()
    show_debug = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=== Part 2: HandController ===")
    print("  Fist            -> THRUST")
    print("  Index+Middle up -> SHOOT")
    print("  Open hand       -> BRAKE")
    print("  Tilt left/right -> steering")
    print("  Q — quit   D — debug overlay")
    print("==============================\n")

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
            lms = result.hand_landmarks[0]
            controller.update(lms)
        else:
            lms = None
            controller.update(None)

        # Draw
        if lms is not None:
            if show_debug:
                draw_hand(frame, lms, w, h)
                draw_tilt_arrow(frame, lms, w, h, controller.steering)
        else:
            cv2.putText(
                frame,
                "Show your hand...",
                (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (100, 100, 255),
                2,
                cv2.LINE_AA,
            )

        draw_steering_bar(frame, controller, 14, h - 260)
        draw_gesture_panel(frame, controller, 14, h - 230)

        fps_counter.tick()
        cv2.putText(
            frame,
            f"FPS: {fps_counter.get_fps():.0f}",
            (w - 90, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "[Q] quit   [D] debug overlay",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

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
