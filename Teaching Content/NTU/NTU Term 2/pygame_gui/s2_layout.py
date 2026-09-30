"""
Stage 2 - The dashboard layout
==============================
Same loop as stage 1, but now we draw something. Nothing reacts yet -
this stage is only about putting shapes and words on the screen.

Two new ideas:

  * pygame.Rect - a rectangle, stored as (left, top, width, height).
    Pygame uses these everywhere, and in stage 6 we will ask a Rect
    whether the mouse is inside it.

  * Helper functions - drawing text takes three fiddly lines, so we
    write draw_text() once and call it everywhere instead.

Remember that the screen's y axis points DOWN. (0, 0) is the top left
corner, so a bigger y means further down the window.

Run with:   python stage_02.py
Needs:      pip install pygame
"""

import pygame

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

WINDOW_W, WINDOW_H = 900, 520
FPS = 60

PANEL_W = 340  # the control panel fills the left of the window

# The whole color scheme lives here. Change a value and every part of
# the program that uses it changes too.
BG = (24, 26, 32)
PANEL = (34, 37, 46)
BTN = (52, 57, 70)
BTN_EDGE = (80, 86, 104)
TEXT = (232, 234, 240)
MUTED = (138, 145, 163)
ACCENT = (80, 205, 165)


# ---------------------------------------------------------------------
# DRAWING HELPERS
# ---------------------------------------------------------------------


def draw_text(surface, text, pos, font, color=TEXT, center=False):
    """Draw some text and return nothing.

    `color` and `center` have default values, so most calls can leave
    them out. Pass center=True to treat `pos` as the middle of the text
    instead of its top left corner.
    """
    image = font.render(text, True, color)  # turn the string into a picture
    rect = image.get_rect()  # a Rect the same size as it
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)  # "blit" means paste it on


def draw_button(surface, rect, label, font):
    """Draw one rounded box with a label in the middle of it."""
    pygame.draw.rect(surface, BTN, rect, border_radius=8)
    pygame.draw.rect(surface, BTN_EDGE, rect, width=2, border_radius=8)
    draw_text(surface, label, rect.center, font, TEXT, center=True)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()

    # Fonts must be created after pygame.init(). The first name in the
    # list that exists on this computer gets used.
    font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
    font = pygame.font.SysFont("consolas,menlo,monospace", 17)
    font_small = pygame.font.SysFont("consolas,menlo,monospace", 13)

    # The five driving buttons, laid out as a cross:
    #
    #            FWD            <- 130 across, 110 down
    #     LEFT  STOP  RIGHT
    #            BACK
    #
    # Each button is 80 wide and 60 tall, with a 10 pixel gap.
    fwd_rect = pygame.Rect(130, 110, 80, 60)
    left_rect = pygame.Rect(40, 180, 80, 60)
    stop_rect = pygame.Rect(130, 180, 80, 60)
    right_rect = pygame.Rect(220, 180, 80, 60)
    back_rect = pygame.Rect(130, 250, 80, 60)

    # Where the sensor graph will go in stage 9.
    graph_rect = pygame.Rect(370, 245, 500, 225)

    running = True
    while running:
        clock.tick(FPS)

        # ---- 1. handle input ----------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # ---- 2. update ----------------------------------------------
        # Still nothing.

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)

        # The left panel, drawn first so everything else sits on top
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        # Left side: title and buttons
        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(screen, "nothing works yet!", (24, 66), font_small, MUTED)

        # Five almost identical lines. That repetition is a hint that
        # something better is coming - see stages 6 and 7.
        draw_button(screen, fwd_rect, "FWD", font)
        draw_button(screen, left_rect, "LEFT", font)
        draw_button(screen, stop_rect, "STOP", font)
        draw_button(screen, right_rect, "RIGHT", font)
        draw_button(screen, back_rect, "BACK", font)

        # Right side: the sensor readout
        draw_text(screen, "DISTANCE SENSOR", (370, 22), font_big)
        draw_text(screen, "--- . -", (370, 62), font_big, MUTED)

        pygame.draw.rect(screen, PANEL, graph_rect, border_radius=8)
        pygame.draw.rect(screen, BTN_EDGE, graph_rect, width=1, border_radius=8)
        draw_text(
            screen, "graph goes here", graph_rect.center, font_small, MUTED, center=True
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Move the FWD button 20 pixels to the left. Which number in the
#      Rect did you change, and which way does that axis run?
#   2. Add a sixth button below BACK labelled "SPIN". Copy one of the
#      existing lines - how many separate places did you have to edit?
#   3. Swap PANEL and BG in the color list at the top. Every panel in
#      the window changes at once. That is the payoff for naming colors
#      instead of typing (34, 37, 46) all over the place.
#   4. Try drawing the left panel AFTER the buttons. Where did they go?
# ---------------------------------------------------------------------
