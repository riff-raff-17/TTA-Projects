"""
Stage 1 - The window and the loop
=================================
Every Pygame program is the same shape: open a window, then loop until
the user quits. That loop runs about 60 times a second, and each time
round it does three things:

    1. look at what the user did      (events)
    2. work out what changed          (update - nothing yet!)
    3. draw the whole screen again    (draw)

Right now the window is empty. That is fine. What matters is that it
opens, stays responsive, and closes properly.

Run with:   python stage_01.py
Needs:      pip install pygame

Quit with the X button or the Esc key.
"""

import pygame

# ---------------------------------------------------------------------
# SETTINGS
# Constants go at the top in CAPITALS so they are easy to find later.
# ---------------------------------------------------------------------

WINDOW_W, WINDOW_H = 900, 520  # window size in pixels
FPS = 60  # how many times per second we redraw

BG = (24, 26, 32)  # a color is (red, green, blue), 0-255


def main():
    pygame.init()  # start Pygame
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))  # make the window
    pygame.display.set_caption("Robot Control Panel")  # title bar text
    clock = pygame.time.Clock()  # used to limit the speed

    print("Window open. Press Esc or click the X to quit.")

    running = True
    while running:
        # Wait just long enough that the loop runs FPS times a second.
        # Without this, the program would spin as fast as possible
        # and make your laptop fan very unhappy.
        clock.tick(FPS)

        # --- 1. handle events ---
        # Pygame collects everything the user did since last time into
        # a list of events. We look at each one in turn.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # the X button
                running = False
            elif event.type == pygame.KEYDOWN:  # a key was pressed
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- 2. update ----
        # Nothing to update yet. Later this is where the robot commands
        # and sensor readings will live.

        # --- 3. draw ---
        screen.fill(BG)  # paint over everything from last frame
        pygame.display.flip()  # show the result on the actual screen

    print("Closing down.")
    pygame.quit()


# This line means "only run main() if this file was started directly".
# It is a Python habit worth copying from the start.
if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Change BG to (120, 30, 60). What color is the window now?
#   2. Set FPS to 5. The window still works - but notice how slowly it
#      reacts when you press Esc. Why?
#   3. Delete the `screen.fill(BG)` line and drag another window across
#      yours. What happens, and what does that tell you about why we
#      redraw the whole screen every single frame?
# ---------------------------------------------------------------------
