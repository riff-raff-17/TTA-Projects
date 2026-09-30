"""
Stage 3 - Keyboard control
==========================
The layout from stage 2, but now the program remembers something.

One variable, `command`, holds what the robot has most recently been
told to do. Pressing a key changes it, and the drawing code reads it.
That is the whole idea behind an interactive program:

    input changes state  ->  drawing shows state

Nothing is ever drawn directly by the key handler. The keys only change
`command`, and the draw section rebuilds the screen from it each frame.
Keeping those two jobs apart is what stops GUI code turning into soup.

New ideas:
  * KEYDOWN events and pygame's key constants (pygame.K_UP and friends)
  * a dict used as a lookup table, instead of a long if/elif chain
  * f-strings, for putting a variable inside a piece of text

Run with:   python stage_03.py
Needs:      pip install pygame
"""

import pygame

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

WINDOW_W, WINDOW_H = 900, 520
FPS = 60

PANEL_W = 340

BG = (24, 26, 32)
PANEL = (34, 37, 46)
BTN = (52, 57, 70)
BTN_EDGE = (80, 86, 104)
TEXT = (232, 234, 240)
MUTED = (138, 145, 163)
ACCENT = (80, 205, 165)
INK = (14, 20, 26)  # dark text, for use on a bright button


# Which key means which command. A dict is a set of key: value pairs,
# and looking something up in one is much tidier than writing
# "if event.key == pygame.K_UP: ... elif event.key == pygame.K_w: ..."
# over and over. Note that two different keys can give the same command.
KEYMAP = {
    pygame.K_UP: "Forward",
    pygame.K_w: "Forward",
    pygame.K_DOWN: "Back",
    pygame.K_s: "Back",
    pygame.K_LEFT: "Left",
    pygame.K_a: "Left",
    pygame.K_RIGHT: "Right",
    pygame.K_d: "Right",
    pygame.K_SPACE: "Stop",
}


# ---------------------------------------------------------------------
# DRAWING HELPERS
# ---------------------------------------------------------------------


def draw_text(surface, text, pos, font, color=TEXT, center=False):
    """Draw some text at a position."""
    image = font.render(text, True, color)
    rect = image.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)


def draw_button(surface, rect, label, font, active=False):
    """Draw one button. An active button is filled with the accent color."""
    if active:
        fill, label_color = ACCENT, INK
    else:
        fill, label_color = BTN, TEXT

    pygame.draw.rect(surface, fill, rect, border_radius=8)
    pygame.draw.rect(surface, BTN_EDGE, rect, width=2, border_radius=8)
    draw_text(surface, label, rect.center, font, label_color, center=True)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
    font = pygame.font.SysFont("consolas,menlo,monospace", 17)
    font_small = pygame.font.SysFont("consolas,menlo,monospace", 13)

    fwd_rect = pygame.Rect(130, 110, 80, 60)
    left_rect = pygame.Rect(40, 180, 80, 60)
    stop_rect = pygame.Rect(130, 180, 80, 60)
    right_rect = pygame.Rect(220, 180, 80, 60)
    back_rect = pygame.Rect(130, 250, 80, 60)

    graph_rect = pygame.Rect(370, 245, 500, 225)

    # THE STATE. Everything the program remembers lives in variables
    # like this one, created before the loop starts.
    command = "Stop"

    running = True
    while running:
        clock.tick(FPS)

        # ---- 1. handle input ----------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in KEYMAP:
                    # "in" asks whether that key is one of the keys we
                    # care about. If it is, look up what it means.
                    command = KEYMAP[event.key]
                    print(f"command is now: {command}")

        # ---- 2. update ----------------------------------------------
        # Nothing yet. In stage 4 this is where the robot gets told.

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(
            screen, "arrows / WASD to drive, space to stop", (24, 66), font_small, MUTED
        )

        # Each button asks the same question: am I the current command?
        draw_button(screen, fwd_rect, "FWD", font, active=(command == "Forward"))
        draw_button(screen, left_rect, "LEFT", font, active=(command == "Left"))
        draw_button(screen, stop_rect, "STOP", font, active=(command == "Stop"))
        draw_button(screen, right_rect, "RIGHT", font, active=(command == "Right"))
        draw_button(screen, back_rect, "BACK", font, active=(command == "Back"))

        draw_text(screen, "DISTANCE SENSOR", (370, 22), font_big)
        draw_text(screen, "--- . -", (370, 62), font_big, MUTED)

        # An f-string: the part inside {curly braces} is replaced by the
        # value of that variable when the line runs.
        draw_text(screen, f"command     {command}", (370, 140), font, MUTED)

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
#   1. Hold the up arrow down. How many lines get printed? A KEYDOWN
#      event happens once when the key goes down, not continuously
#      while it is held.
#   2. Now add this just above the draw section:
#          keys = pygame.key.get_pressed()
#          if keys[pygame.K_UP]:
#              print("up is being held")
#      Run it and hold the key again. That is the other way to read the
#      keyboard: events tell you what just happened, get_pressed() tells
#      you what is true right now.
#   3. Add Q and E to KEYMAP as "Spin Left" and "Spin Right". How many
#      lines did that take? Compare with doing it as if/elif.
#   4. Delete the `command = "Stop"` line before the loop. Read the
#      error carefully - it is the single most common Python error and
#      worth recognising early.
# ---------------------------------------------------------------------
