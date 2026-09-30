"""
Stage 4 - Talking to a robot
============================
Stage 3 changed a variable. This stage turns that variable into
instructions.

Three new functions appear at the top, named exactly as they are in the
robot's documentation. For now they only print. In stage 5 we fill them
in with the real thing - and because the names and arguments already
match, nothing else in the file will need to change.

That is worth noticing: you can build and test almost the whole program
before you ever plug a robot in.

New ideas:
  * writing functions that take parameters
  * named constants instead of "magic numbers"
  * doing one job in one place, so there is one thing to fix later

Run with:   python stage_04.py
Watch the terminal while you press keys - that is the point of this one.
"""

import pygame

# ---------------------------------------------------------------------
# ROBOT COMMANDS
# These match the robot's documentation exactly. Right now they just
# print what WOULD be sent.
# ---------------------------------------------------------------------


def mecanum_stop():
    """Stop the mecanum wheel vehicle."""
    print("mecanum_stop()")


def mecanum_move_speed(direction, speed):
    """Move forward or backward.

    direction : 0 forward, 1 backward
    speed     : 5 to 80, in centimetres per second
    """
    print(f"mecanum_move_speed({direction}, {speed})")


def mecanum_turn_speed(turn, speed):
    """Turn on the spot.

    turn  : 2 left, 3 right
    speed : 5 to 280, in degrees per second
    """
    print(f"mecanum_turn_speed({turn}, {speed})")


# The documentation says direction 0 means forward. Writing 0 in the
# middle of the code would leave everyone guessing, so the numbers get
# names once, here, and we use the names everywhere else.
DIR_FORWARD = 0
DIR_BACKWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

DRIVE_SPEED = 30  # cm/s   (the docs allow 5 to 80)
TURN_SPEED = 90  # deg/s  (the docs allow 5 to 280)


def issue(name):
    """Turn a command name into exactly one robot library call.

    This is the ONLY function in the program that talks to the robot.
    If the robot's library ever changes, this is the only place to edit.
    """
    if name == "Forward":
        mecanum_move_speed(DIR_FORWARD, DRIVE_SPEED)
    elif name == "Back":
        mecanum_move_speed(DIR_BACKWARD, DRIVE_SPEED)
    elif name == "Left":
        mecanum_turn_speed(TURN_LEFT, TURN_SPEED)
    elif name == "Right":
        mecanum_turn_speed(TURN_RIGHT, TURN_SPEED)
    else:
        mecanum_stop()


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
INK = (14, 20, 26)

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
                    command = KEYMAP[event.key]
                    issue(command)  # <-- the robot gets told here

        # ---- 2. update ----------------------------------------------
        # Nothing yet. The sensor arrives in stage 8.

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(
            screen, "arrows / WASD to drive, space to stop", (24, 66), font_small, MUTED
        )

        draw_button(screen, fwd_rect, "FWD", font, active=(command == "Forward"))
        draw_button(screen, left_rect, "LEFT", font, active=(command == "Left"))
        draw_button(screen, stop_rect, "STOP", font, active=(command == "Stop"))
        draw_button(screen, right_rect, "RIGHT", font, active=(command == "Right"))
        draw_button(screen, back_rect, "BACK", font, active=(command == "Back"))

        draw_text(screen, f"DRIVE  {DRIVE_SPEED} cm/s", (40, 324), font, MUTED)
        draw_text(screen, f"TURN   {TURN_SPEED} deg/s", (40, 350), font, MUTED)

        draw_text(screen, "DISTANCE SENSOR", (370, 22), font_big)
        draw_text(screen, "--- . -", (370, 62), font_big, MUTED)
        draw_text(screen, f"command     {command}", (370, 140), font, MUTED)

        pygame.draw.rect(screen, PANEL, graph_rect, border_radius=8)
        pygame.draw.rect(screen, BTN_EDGE, graph_rect, width=1, border_radius=8)
        draw_text(
            screen, "graph goes here", graph_rect.center, font_small, MUTED, center=True
        )

        pygame.display.flip()

    # The window has closed, so stop the motors. A real robot will keep
    # driving until told otherwise - including into a wall, or off a
    # desk. Always leave a program with the robot stopped.
    mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Press the keys in turn and read the terminal. Every line printed
#      is exactly what the real robot will be sent in stage 5.
#   2. Change DRIVE_SPEED to 30000 and press up. Nothing complains - the
#      docs say 5 to 80, but nothing is checking. Stage 10 adds that.
#   3. Swap DIR_FORWARD and DIR_BACKWARD. The program still runs, and
#      the display still looks right, but the robot would now drive the
#      wrong way. Constants are only as good as the documentation you
#      copied them from, so put them back.
#   4. Add a "Spin" command to KEYMAP on the Q key, then teach issue()
#      what to do with it. How many places did you edit? (Two - and one
#      of them is the only place that mentions the robot at all.)
# ---------------------------------------------------------------------
