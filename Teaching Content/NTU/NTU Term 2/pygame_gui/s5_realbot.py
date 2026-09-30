"""
Stage 5 - The real robot
========================
Stage 4 printed what it would send. This stage sends it.

Look at how little changed. The three functions at the top kept their
names and their arguments, so all we did was replace one print with one
real call. issue(), the keymap, the drawing code and the main loop are
untouched.

    BEFORE:  print(f"mecanum_stop()")
    AFTER:   robot.mecanum_stop()

That is the reward for writing the placeholders with the right shape in
the first place.

BEFORE YOU RUN THIS
  * Put your robot's address in ROBOT_IP below.
  * Put the robot on a book or box so its wheels spin in mid air. Test
    the directions before letting it loose on a table.
  * Keep the speeds low to begin with. They are set gently below.
  * Esc or the X button stops the motors and closes down.

Run with:   python stage_05.py
Needs:      pip install pygame ugot
"""

import pygame
from ugot import ugot

# ---------------------------------------------------------------------
# CONNECT
# These are the same lines you would type at the top of any robot
# script. They run once, when the program starts.
#
# The robot's own examples often call this object `got`. The name is up
# to you - it is just the thing you send commands to.
# ---------------------------------------------------------------------

ROBOT_IP = "192.168.0.1"        # <-- your robot's address goes here

print(f"Connecting to {ROBOT_IP} ...")
robot = ugot.UGOT()
robot.initialize(ROBOT_IP)
print("Connected.")


# ---------------------------------------------------------------------
# ROBOT COMMANDS
# Same names and arguments as stage 4. Only the bodies changed.
# ---------------------------------------------------------------------

def mecanum_stop():
    """Stop the mecanum wheel vehicle."""
    robot.mecanum_stop()


def mecanum_move_speed(direction, speed):
    """Move forward or backward.

    direction : 0 forward, 1 backward
    speed     : 5 to 80, in centimetres per second
    """
    robot.mecanum_move_speed(direction, speed)


def mecanum_turn_speed(turn, speed):
    """Turn on the spot.

    turn  : 2 left, 3 right
    speed : 5 to 280, in degrees per second
    """
    robot.mecanum_turn_speed(turn, speed)


DIR_FORWARD  = 0
DIR_BACKWARD = 1
TURN_LEFT    = 2
TURN_RIGHT   = 3

# Start slow. Turn these up once you trust the directions.
DRIVE_SPEED = 15      # cm/s   (the docs allow 5 to 80)
TURN_SPEED  = 45      # deg/s  (the docs allow 5 to 280)


def issue(name):
    """Turn a command name into exactly one robot library call.

    This is the ONLY function in the program that talks to the robot.
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

    fwd_rect   = pygame.Rect(130, 110, 80, 60)
    left_rect  = pygame.Rect( 40, 180, 80, 60)
    stop_rect  = pygame.Rect(130, 180, 80, 60)
    right_rect = pygame.Rect(220, 180, 80, 60)
    back_rect  = pygame.Rect(130, 250, 80, 60)

    graph_rect = pygame.Rect(370, 245, 500, 225)

    command = "Stop"
    issue(command)          # make sure the robot is stopped at the start

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
                    issue(command)          # <-- the robot really moves now

        # ---- 2. update ----------------------------------------------
        # Nothing yet. The sensor arrives in stage 8.

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(screen, "arrows / WASD to drive, space to stop",
                  (24, 66), font_small, MUTED)

        draw_button(screen, fwd_rect,   "FWD",   font, active=(command == "Forward"))
        draw_button(screen, left_rect,  "LEFT",  font, active=(command == "Left"))
        draw_button(screen, stop_rect,  "STOP",  font, active=(command == "Stop"))
        draw_button(screen, right_rect, "RIGHT", font, active=(command == "Right"))
        draw_button(screen, back_rect,  "BACK",  font, active=(command == "Back"))

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

    # The window has closed, so stop the motors. A movement command
    # stays in force until something cancels it - closing the program
    # is not enough on its own.
    mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Wheels off the ground, press each key and check the robot moves
#      the way the label says. If forward and back are swapped, fix the
#      numbers in DIR_FORWARD / DIR_BACKWARD - not the labels.
#   2. Press up, then close the window with the X. The robot stops,
#      because of the mecanum_stop() at the end of main(). Now comment
#      that line out and try again. Be ready to catch it.
#   3. Raise DRIVE_SPEED to 40 and drive again. Does the robot travel
#      roughly twice as far in the same time?
#   4. Press up and then hold the key down. The robot does not speed up
#      or change - one KEYDOWN sent one command, and the robot is still
#      obeying it. This is why STOP matters.
#
# IF IT WILL NOT CONNECT
#   * Is the laptop on the same network as the robot?
#   * Is the address in ROBOT_IP exactly right?
#   * The program will sit on "Connecting..." for a while before giving
#     up, and then show an error ending in the ugot library. That is
#     normal - it means the address was wrong or the robot was asleep.
# ---------------------------------------------------------------------