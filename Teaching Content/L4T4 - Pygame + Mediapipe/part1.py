"""
=============================================================
PART 1 — MediaPipe Hello World
=============================================================
Goal: confirm MediaPipe is working and understand the
21-landmark hand model with live visualisation.

Depends on: hand_common.py (must be in the same folder)

Run:
    python part1_mediapipe_hello.py

Controls:
    Q  — quit
    L  — toggle landmark index labels
    D  — toggle coordinate data panel
=============================================================
"""

import cv2
import mediapipe as mp
from hand_common import (
    make_detector,
    draw_hand,
    draw_landmark_indices,
    draw_data_panel,
    FPSCounter,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
detector = make_detector(num_hands=2)
fps_counter = FPSCounter()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_indices = True
    show_data = True

    print("=== Part 1: MediaPipe Hello World ===")
    print("  Q — quit  |  L — landmark labels  |  D — data panel")
    print("=====================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(mp_image)

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.hand_landmarks, result.handedness
            ):
                hand_label = handedness[0].display_name

                draw_hand(frame, hand_landmarks, img_w, img_h)

                if show_indices:
                    draw_landmark_indices(frame, hand_landmarks, img_w, img_h)
                if show_data:
                    draw_data_panel(frame, hand_landmarks, hand_label, img_w, img_h)

                wrist = hand_landmarks[0]
                index_tip = hand_landmarks[8]
                print(
                    f"[{hand_label}]  "
                    f"WRIST=({wrist.x:.3f}, {wrist.y:.3f})  "
                    f"INDEX_TIP=({index_tip.x:.3f}, {index_tip.y:.3f})",
                    end="\r",
                )
        else:
            print("No hands detected...                     ", end="\r")

        fps_counter.tick()
        cv2.putText(
            frame,
            f"FPS: {fps_counter.get_fps():.0f}",
            (img_w - 90, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "[Q] quit   [L] landmarks   [D] data",
            (10, img_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("Part 1 - MediaPipe Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            show_indices = not show_indices
        elif key == ord("d"):
            show_data = not show_data

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
