"""
Stage 6 - Clickable buttons
===========================
The keyboard already works. Now the buttons on screen do too.

Clicking is a two step idea:
    1. a MOUSEBUTTONDOWN event tells you WHERE the click happened
    2. rect.collidepoint(position) tells you whether that point is
       inside a given rectangle

So to find which button was clicked, we check every button in turn.
That means the buttons can no longer be five separate variables - the
program has to be able to loop over them.

This stage stores them in three lists that must line up with each other:

    button_rects[2] is the rectangle,
    button_labels[2] is its label,
    button_commands[2] is what it does.

It works. It is also fragile and annoying to edit, and you are meant to
notice that. Stage 7 fixes it properly.

Run with:   python stage_06.py
"""

import pygame

from ugot import ugot


# ---------------------------------------------------------------------
# CONNECT
# ---------------------------------------------------------------------

ROBOT_IP = "192.168.0.1"        # <-- your robot's address goes here

print(f"Connecting to {ROBOT_IP} ...")
robot = ugot.UGOT()
robot.initialize(ROBOT_IP)
print("Connected.")


# ---------------------------------------------------------------------
# ROBOT COMMANDS
#
# No robot today? Replace the body of each function with the print()
# line from stage 4 and everything below still works. Nothing else in
# the program knows or cares which version is in use. Stage 11 makes
# that swap properly, with a switch.
# ---------------------------------------------------------------------

def mecanum_stop():
    """Stop the mecanum wheel vehicle."""
    robot.mecanum_stop()


def mecanum_move_speed(direction, speed):
    """direction: 0 forward, 1 backward.  speed: 5-80 cm/s."""
    robot.mecanum_move_speed(direction, speed)


def mecanum_turn_speed(turn, speed):
    """turn: 2 left, 3 right.  speed: 5-280 deg/s."""
    robot.mecanum_turn_speed(turn, speed)


DIR_FORWARD  = 0
DIR_BACKWARD = 1
TURN_LEFT    = 2
TURN_RIGHT   = 3

DRIVE_SPEED = 15      # cm/s   (the docs allow 5 to 80)
TURN_SPEED  = 45      # deg/s  (the docs allow 5 to 280)


def issue(name):
    """Turn a command name into exactly one robot library call."""
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

BG       = (24, 26, 32)
PANEL    = (34, 37, 46)
BTN      = (52, 57, 70)
BTN_EDGE = (80, 86, 104)
TEXT     = (232, 234, 240)
MUTED    = (138, 145, 163)
ACCENT   = (80, 205, 165)
INK      = (14, 20, 26)

KEYMAP = {
    pygame.K_UP: "Forward",     pygame.K_w: "Forward",
    pygame.K_DOWN: "Back",      pygame.K_s: "Back",
    pygame.K_LEFT: "Left",      pygame.K_a: "Left",
    pygame.K_RIGHT: "Right",    pygame.K_d: "Right",
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

    # THREE LISTS, ONE PER PIECE OF INFORMATION.
    # The order matters! Item number 3 of each list has to describe the
    # same button, or the panel will lie to you.
    button_rects = [
        pygame.Rect(130, 110, 80, 60),
        pygame.Rect( 40, 180, 80, 60),
        pygame.Rect(130, 180, 80, 60),
        pygame.Rect(220, 180, 80, 60),
        pygame.Rect(130, 250, 80, 60),
    ]
    button_labels   = ["FWD",     "LEFT", "STOP", "RIGHT", "BACK"]
    button_commands = ["Forward", "Left", "Stop", "Right", "Back"]

    graph_rect = pygame.Rect(370, 245, 500, 225)

    command = "Stop"
    issue(command)

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
                    issue(command)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # event.pos is where the click landed, as (x, y).
                # Check every button to see if the click was inside it.
                for i in range(len(button_rects)):
                    if button_rects[i].collidepoint(event.pos):
                        command = button_commands[i]
                        issue(command)

        # ---- 2. update ----------------------------------------------
        # Nothing yet. The sensor arrives in stage 8.

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(screen, "click, or use arrows / WASD",
                  (24, 66), font_small, MUTED)

        # Drawing has to walk the same three lists, in step, by index.
        for i in range(len(button_rects)):
            draw_button(screen, button_rects[i], button_labels[i], font,
                        active=(button_commands[i] == command))

        draw_text(screen, f"DRIVE  {DRIVE_SPEED} cm/s", (40, 324), font, MUTED)
        draw_text(screen, f"TURN   {TURN_SPEED} deg/s", (40, 350), font, MUTED)

        draw_text(screen, "DISTANCE SENSOR", (370, 22), font_big)
        draw_text(screen, "--- . -", (370, 62), font_big, MUTED)
        draw_text(screen, f"command     {command}", (370, 140), font, MUTED)
        draw_text(screen, f"robot       {ROBOT_IP}", (370, 162), font, MUTED)

        pygame.draw.rect(screen, PANEL, graph_rect, border_radius=8)
        pygame.draw.rect(screen, BTN_EDGE, graph_rect, width=1, border_radius=8)
        draw_text(screen, "graph goes here", graph_rect.center,
                  font_small, MUTED, center=True)

        pygame.display.flip()

    mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Click each button and watch the terminal. Clicking FWD and
#      pressing the up arrow send exactly the same thing - two inputs,
#      one command, one place that talks to the robot.
#   2. Try right-clicking a button. Nothing happens, because of the
#      `event.button == 1` test. Remove that test and try again.
#   3. Add a sixth button, "SPIN", at (130, 320, 80, 60). You have to
#      edit three lists and keep them in the same order. Now delete the
#      label you added but leave the other two. What breaks, and does
#      Python warn you before it happens?
#   4. Swap "LEFT" and "RIGHT" in button_labels only. The panel now
#      lies: the labels and the commands disagree, and nothing in the
#      program can tell. Put them back.
#
# WHAT THIS STAGE IS REALLY SHOWING
#   Three lists that must stay in step are three chances to get it
#   wrong. The information belongs together - one button, one thing,
#   holding its own rect, label and command. That is what a class is
#   for, and that is stage 7.
# ---------------------------------------------------------------------