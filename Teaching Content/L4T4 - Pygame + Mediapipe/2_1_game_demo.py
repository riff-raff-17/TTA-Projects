"""
=============================================================
PART 2 — Hand-Controlled Character (MediaPipe + Pygame)
=============================================================
Goal: use your hand to move a character around a pygame window.

  - Index finger tip (landmark 8) steers the character.
  - Pinch (thumb tip + index tip close together) makes the
    character "activate" (changes colour + grows briefly).

Depends on: hand_common.py

Run:
    pip install pygame opencv-python mediapipe
    python 2_hand_game.py

Controls:
    ESC / close window — quit
=============================================================
"""

import cv2
import mediapipe as mp
import pygame
import math

from hand_common import (
    make_detector,
    INDEX_TIP,
    THUMB_TIP,
    FPSCounter,
)

# --- Config ---
SCREEN_W, SCREEN_H = 960, 640
CAM_W, CAM_H = 640, 480

# How much to smooth the character's motion (0 = no smoothing, 1 = frozen)
SMOOTHING = 0.75

# Distance between thumb tip and index tip (in normalised units) below
# which we consider the hand "pinching"
PINCH_THRESHOLD = 0.05


# --- Helpers ---
def norm_distance(a, b):
    """Euclidean distance between two normalised landmarks (ignoring z)."""
    return math.hypot(a.x - b.x, a.y - b.y)


def lerp(a, b, t):
    return a + (b - a) * t


# --- Main ---
def main():
    # --- Pygame setup ---
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Part 2 - Hand-Controlled Character")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # --- Webcam + MediaPipe setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    detector = make_detector(num_hands=1)
    fps = FPSCounter()

    # --- Character state ---
    char_x, char_y = SCREEN_W / 2, SCREEN_H / 2
    target_x, target_y = char_x, char_y
    is_pinching = False

    running = True
    while running:
        # --- Input: webcam frame ---
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        img_h, img_w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(mp_image)

        hand_visible = False
        if result.hand_landmarks:
            hand_visible = True
            lms = result.hand_landmarks[0]
            index_tip = lms[INDEX_TIP]
            thumb_tip = lms[THUMB_TIP]

            # Normalised (0-1) -> pygame screen coordinates
            target_x = index_tip.x * SCREEN_W
            target_y = index_tip.y * SCREEN_H

            # Pinch detection
            is_pinching = norm_distance(index_tip, thumb_tip) < PINCH_THRESHOLD

            # (Optional) draw the fingertip on the webcam preview
            cv2.circle(
                frame,
                (int(index_tip.x * img_w), int(index_tip.y * img_h)),
                8,
                (0, 255, 0),
                -1,
            )
        else:
            is_pinching = False

        # --- Update: smooth character towards target ---
        char_x = lerp(char_x, target_x, 1 - SMOOTHING)
        char_y = lerp(char_y, target_y, 1 - SMOOTHING)

        # --- Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # --- Render ---
        screen.fill((20, 22, 30))

        # Character
        radius = 40 if is_pinching else 28
        color = (255, 180, 60) if is_pinching else (80, 200, 255)
        pygame.draw.circle(screen, color, (int(char_x), int(char_y)), radius)
        pygame.draw.circle(
            screen, (255, 255, 255), (int(char_x), int(char_y)), radius, 2
        )

        # HUD
        hud_lines = [
            f"FPS: {clock.get_fps():.0f}",
            f"Hand: {'YES' if hand_visible else 'no'}",
            f"Pinch: {'YES' if is_pinching else 'no'}",
            f"Pos: ({int(char_x)}, {int(char_y)})",
        ]
        for i, line in enumerate(hud_lines):
            surf = font.render(line, True, (220, 220, 220))
            screen.blit(surf, (10, 10 + i * 20))

        if not hand_visible:
            msg = font.render(
                "Show your hand to the camera!", True, (255, 120, 120)
            )
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, 20))

        pygame.display.flip()
        fps.tick()

        # -------- Optional webcam preview window --------
        cv2.imshow("Camera (press Q in this window to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

        clock.tick(60)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    pygame.quit()


if __name__ == "__main__":
    main()