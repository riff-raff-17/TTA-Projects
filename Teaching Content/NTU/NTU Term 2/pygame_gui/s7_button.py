"""
Stage 7 - A Button class
========================
Stage 6 worked, but a button was scattered across three lists that had
to stay in step. A button is one thing, so let it be one thing.

    BEFORE                            AFTER
    button_rects[i]                   button.rect
    button_labels[i]                  button.label
    button_commands[i]                button.command

A class is a recipe for making objects. Button says "every button has a
rect, a label and a command, and here is how to ask it questions and
how to draw it". Each button made from that recipe keeps its own copy
of those values.

`self` is how a method refers to the particular button it was called
on. When you write button.is_over(pos), Python runs is_over with self
set to that button.

Because each button can now hold its own information, adding hover
highlighting takes about three lines - it was impractical before.

Run with:   python stage_07.py
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
# No robot today? Replace each body with the print() line from stage 4.
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

BG        = (24, 26, 32)
PANEL     = (34, 37, 46)
BTN       = (52, 57, 70)
BTN_HOVER = (68, 74, 90)      # new: a slightly lighter shade
BTN_EDGE  = (80, 86, 104)
TEXT      = (232, 234, 240)
MUTED     = (138, 145, 163)
ACCENT    = (80, 205, 165)
INK       = (14, 20, 26)

KEYMAP = {
    pygame.K_UP: "Forward",     pygame.K_w: "Forward",
    pygame.K_DOWN: "Back",      pygame.K_s: "Back",
    pygame.K_LEFT: "Left",      pygame.K_a: "Left",
    pygame.K_RIGHT: "Right",    pygame.K_d: "Right",
    pygame.K_SPACE: "Stop",
}


# ---------------------------------------------------------------------
# DRAWING HELPER
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


# ---------------------------------------------------------------------
# THE BUTTON CLASS
# ---------------------------------------------------------------------

class Button:
    """One button: where it is, what it says, and what it does."""

    def __init__(self, rect, label, command):
        """Runs once for each button you make, to set it up.

        pygame.Rect() happily takes a plain (x, y, w, h) tuple, so you
        can write the numbers straight into the list below.
        """
        self.rect = pygame.Rect(rect)
        self.label = label
        self.command = command

    def is_over(self, pos):
        """Is this point - usually the mouse - inside me?"""
        return self.rect.collidepoint(pos)

    def draw(self, surface, font, active=False):
        """Draw myself. Active means I am the current command."""
        # A button can check the mouse for itself now, so hovering is
        # easy. In stage 6 there was nowhere sensible to put this.
        hovered = self.is_over(pygame.mouse.get_pos())

        if active:
            fill, label_color = ACCENT, INK
        elif hovered:
            fill, label_color = BTN_HOVER, TEXT
        else:
            fill, label_color = BTN, TEXT

        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, BTN_EDGE, self.rect, width=2, border_radius=8)
        draw_text(surface, self.label, self.rect.center, font,
                  label_color, center=True)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
    font = pygame.font.SysFont("consolas,menlo,monospace", 17)
    font_small = pygame.font.SysFont("consolas,menlo,monospace", 13)

    # ONE list now. Each line is a whole button, and nothing can fall
    # out of step with anything else.
    buttons = [
        Button((130, 110, 80, 60), "FWD",   "Forward"),
        Button(( 40, 180, 80, 60), "LEFT",  "Left"),
        Button((130, 180, 80, 60), "STOP",  "Stop"),
        Button((220, 180, 80, 60), "RIGHT", "Right"),
        Button((130, 250, 80, 60), "BACK",  "Back"),
    ]

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
                # No more index numbers - just ask each button directly.
                for button in buttons:
                    if button.is_over(event.pos):
                        command = button.command
                        issue(command)

        # ---- 2. update ----------------------------------------------
        # Nothing yet. The sensor arrives in stage 8.

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(screen, "click, or use arrows / WASD",
                  (24, 66), font_small, MUTED)

        for button in buttons:
            button.draw(screen, font, active=(button.command == command))

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
#   1. Move the mouse over the buttons without clicking. That highlight
#      is the whole benefit of this stage in one line of code.
#   2. Add a sixth button: one new line in the `buttons` list, and one
#      new branch in issue(). Compare that with stage 6, where the same
#      change meant editing three lists in the right order.
#   3. Print a button to see what it is:
#          print(buttons[0].label, buttons[0].rect)
#      Then try buttons[0].command = "Back" before the loop starts and
#      click FWD. Objects hold their own values, and you can change them.
#   4. Give Button a new method:
#          def nudge(self, dx, dy):
#              self.rect.move_ip(dx, dy)
#      then call buttons[0].nudge(0, -20) before the loop. Every button
#      gets the ability for free - that is the other half of what a
#      class buys you.
#   5. Delete `self.` from one line inside __init__ and read the error.
#      Without self, the value is forgotten the moment __init__ ends.
# ---------------------------------------------------------------------