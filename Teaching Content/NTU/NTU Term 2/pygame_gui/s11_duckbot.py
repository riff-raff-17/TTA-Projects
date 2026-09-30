"""
Stage 11 - Real robot or fake one
=================================
Every stage since 5 has needed a robot switched on, connected and not
already borrowed by someone else. That is a bad way to write software.

So we write a second class, FakeRobot, with the same method names as
the real one. Then a switch at the top decides which one to build.

    USE_REAL_ROBOT = True   ->  robot = ugot.UGOT()
    USE_REAL_ROBOT = False  ->  robot = FakeRobot()

Nothing else in the program changes, because nothing else asks what
kind of object `robot` is. It just calls robot.mecanum_stop() and
whichever object is there does its version. Python is happy with this:
if it has the method, it can be used. The usual name for that idea is
"if it walks like a duck and quacks like a duck, treat it as a duck".

Notice the wrapper functions from stage 5 are gone. They existed to
hide the fact that the robot was an object; now that both versions are
objects, the code can just call the methods directly.

Run with:   python stage_11.py
Needs:      pip install pygame        (ugot only if USE_REAL_ROBOT is True)
"""

import math
import random
from collections import deque

import pygame

# ---------------------------------------------------------------------
# WHICH ROBOT?
# ---------------------------------------------------------------------

USE_REAL_ROBOT = False  # <-- the switch
ROBOT_IP = "192.168.0.1"  # <-- your robot's address goes here


class FakeRobot:
    """Pretends to be ugot.UGOT() so the panel runs with no robot.

    Every method matches the real one's name and arguments. The bodies
    are the print statements from stage 4, plus some invented sensor
    data so the graph has something to draw.
    """

    def initialize(self, ip):
        print(f"[fake] initialize({ip!r})")

    def mecanum_stop(self):
        print("[fake] mecanum_stop()")

    def mecanum_move_speed(self, direction, speed):
        print(f"[fake] mecanum_move_speed({direction}, {speed})")

    def mecanum_turn_speed(self, turn, speed):
        print(f"[fake] mecanum_turn_speed({turn}, {speed})")

    def read_distance_data(self, sensor_id):
        """Invent a distance that drifts in and out, with a little noise."""
        t = pygame.time.get_ticks() / 1000.0
        distance = 100.0 + 70.0 * math.sin(t * 0.7) + 15.0 * math.sin(t * 3.1)
        return max(3.0, distance + random.uniform(-2.0, 2.0))


def connect_robot():
    """Build whichever robot the switch asks for, and start it up.

    The import sits inside the `if` on purpose: a laptop with no ugot
    package installed can still run the fake version.
    """
    if USE_REAL_ROBOT:
        from ugot import ugot

        robot = ugot.UGOT()
    else:
        robot = FakeRobot()

    print(f"Connecting to {ROBOT_IP} ...")
    robot.initialize(ROBOT_IP)
    print("Connected.")
    return robot


robot = connect_robot()


DIR_FORWARD = 0
DIR_BACKWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

DISTANCE_SENSOR_ID = 1  # <-- check which sensor yours is

DRIVE_MIN, DRIVE_MAX = 5, 80  # cm/s, from the docs
TURN_MIN, TURN_MAX = 5, 280  # deg/s, from the docs


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

WINDOW_W, WINDOW_H = 900, 520
FPS = 60

PANEL_W = 340

POLL_INTERVAL = 0.1
HISTORY_LEN = 120

DISTANCE_UNITS = "cm"
DISTANCE_MAX = 200.0
TOO_CLOSE = 25.0

BG = (24, 26, 32)
PANEL = (34, 37, 46)
BTN = (52, 57, 70)
BTN_HOVER = (68, 74, 90)
BTN_EDGE = (80, 86, 104)
TEXT = (232, 234, 240)
MUTED = (138, 145, 163)
ACCENT = (80, 205, 165)
DANGER = (232, 92, 92)
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
# HELPERS
# ---------------------------------------------------------------------


def clamp(value, low, high):
    """Keep a number inside a range."""
    return max(low, min(value, high))


def draw_text(surface, text, pos, font, color=TEXT, center=False):
    """Draw some text at a position."""
    image = font.render(text, True, color)
    rect = image.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)


def draw_history(surface, history, rect, font):
    """Draw the recent readings as a line graph inside `rect`."""
    pygame.draw.rect(surface, PANEL, rect, border_radius=8)
    pygame.draw.rect(surface, BTN_EDGE, rect, width=1, border_radius=8)

    def y_for(value):
        """Turn a distance into a y position inside the graph."""
        fraction = clamp(value, 0.0, DISTANCE_MAX) / DISTANCE_MAX
        return rect.bottom - fraction * rect.height

    for value in (0, DISTANCE_MAX / 2, DISTANCE_MAX):
        y = y_for(value)
        pygame.draw.line(surface, BTN, (rect.left, y), (rect.right, y), 1)
        draw_text(surface, f"{value:.0f}", (rect.left + 6, y - 15), font, MUTED)

    y = y_for(TOO_CLOSE)
    pygame.draw.line(surface, DANGER, (rect.left, y), (rect.right, y), 1)

    if len(history) < 2:
        return

    step = rect.width / (HISTORY_LEN - 1)
    points = []
    for i, value in enumerate(history):
        x = rect.right - (len(history) - 1 - i) * step
        points.append((x, y_for(value)))

    pygame.draw.lines(surface, ACCENT, False, points, 2)
    newest = points[-1]
    pygame.draw.circle(surface, ACCENT, (int(newest[0]), int(newest[1])), 4)


# ---------------------------------------------------------------------
# THE BUTTON CLASS
# ---------------------------------------------------------------------


class Button:
    """One button: where it is, what it says, and what it does."""

    def __init__(self, rect, label, command):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.command = command

    def is_over(self, pos):
        """Is this point - usually the mouse - inside me?"""
        return self.rect.collidepoint(pos)

    def draw(self, surface, font, active=False):
        """Draw myself. Active means I am the current command."""
        hovered = self.is_over(pygame.mouse.get_pos())

        if active:
            fill, label_color = ACCENT, INK
        elif hovered:
            fill, label_color = BTN_HOVER, TEXT
        else:
            fill, label_color = BTN, TEXT

        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, BTN_EDGE, self.rect, width=2, border_radius=8)
        draw_text(surface, self.label, self.rect.center, font, label_color, center=True)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()

    font_huge = pygame.font.SysFont("consolas,menlo,monospace", 46, bold=True)
    font_big = pygame.font.SysFont("consolas,menlo,monospace", 22, bold=True)
    font = pygame.font.SysFont("consolas,menlo,monospace", 17)
    font_small = pygame.font.SysFont("consolas,menlo,monospace", 13)

    buttons = [
        Button((130, 110, 80, 60), "FWD", "Forward"),
        Button((40, 180, 80, 60), "LEFT", "Left"),
        Button((130, 180, 80, 60), "STOP", "Stop"),
        Button((220, 180, 80, 60), "RIGHT", "Right"),
        Button((130, 250, 80, 60), "BACK", "Back"),
        Button((40, 350, 125, 34), "- DRIVE", "DRIVE_DOWN"),
        Button((175, 350, 125, 34), "+ DRIVE", "DRIVE_UP"),
        Button((40, 424, 125, 34), "- TURN", "TURN_DOWN"),
        Button((175, 424, 125, 34), "+ TURN", "TURN_UP"),
    ]

    graph_rect = pygame.Rect(370, 245, 500, 225)

    command = "Stop"
    drive_speed = 15
    turn_speed = 45

    distance = None
    sensor_error = False
    time_since_poll = 0.0
    history = deque(maxlen=HISTORY_LEN)

    def issue(name):
        """Turn a command name into exactly one robot library call.

        `robot` here is whichever object connect_robot() built. This
        code never has to know or care which.
        """
        if name == "Forward":
            robot.mecanum_move_speed(DIR_FORWARD, drive_speed)
        elif name == "Back":
            robot.mecanum_move_speed(DIR_BACKWARD, drive_speed)
        elif name == "Left":
            robot.mecanum_turn_speed(TURN_LEFT, turn_speed)
        elif name == "Right":
            robot.mecanum_turn_speed(TURN_RIGHT, turn_speed)
        else:
            robot.mecanum_stop()

    def set_command(name):
        """Remember the new command and send it."""
        nonlocal command
        command = name
        issue(name)

    def change_drive(amount):
        """Change the driving speed, then re-send so it takes effect."""
        nonlocal drive_speed
        drive_speed = int(clamp(drive_speed + amount, DRIVE_MIN, DRIVE_MAX))
        if command in ("Forward", "Back"):
            issue(command)

    def change_turn(amount):
        """Change the turning speed, then re-send so it takes effect."""
        nonlocal turn_speed
        turn_speed = int(clamp(turn_speed + amount, TURN_MIN, TURN_MAX))
        if command in ("Left", "Right"):
            issue(command)

    set_command("Stop")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ---- 1. handle input ----------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    change_drive(+5)
                elif event.key == pygame.K_MINUS:
                    change_drive(-5)
                elif event.key == pygame.K_RIGHTBRACKET:
                    change_turn(+10)
                elif event.key == pygame.K_LEFTBRACKET:
                    change_turn(-10)
                elif event.key in KEYMAP:
                    set_command(KEYMAP[event.key])

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if not button.is_over(event.pos):
                        continue
                    if button.command == "DRIVE_UP":
                        change_drive(+5)
                    elif button.command == "DRIVE_DOWN":
                        change_drive(-5)
                    elif button.command == "TURN_UP":
                        change_turn(+10)
                    elif button.command == "TURN_DOWN":
                        change_turn(-10)
                    else:
                        set_command(button.command)

        # ---- 2. update: read the sensor, but not every frame --------
        time_since_poll += dt
        if time_since_poll >= POLL_INTERVAL:
            time_since_poll = 0.0
            try:
                reading = float(robot.read_distance_data(DISTANCE_SENSOR_ID))
                sensor_error = False
                if reading > 0:
                    distance = reading
                    history.append(reading)
            except Exception as error:
                sensor_error = True
                print(f"sensor read failed: {error}")

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(screen, "click, or use arrows / WASD", (24, 66), font_small, MUTED)

        for button in buttons:
            button.draw(screen, font, active=(button.command == command))

        draw_text(screen, f"DRIVE  {drive_speed} cm/s", (40, 324), font, MUTED)
        draw_text(screen, f"TURN   {turn_speed} deg/s", (40, 398), font, MUTED)

        draw_text(screen, "DISTANCE SENSOR", (370, 22), font_big)

        too_close = distance is not None and distance < TOO_CLOSE
        if sensor_error:
            reading_text, reading_color = "-- error --", DANGER
        elif distance is None:
            reading_text, reading_color = "--- . -", MUTED
        else:
            reading_text = f"{distance:.1f} {DISTANCE_UNITS}"
            reading_color = DANGER if too_close else ACCENT
        draw_text(screen, reading_text, (370, 62), font_huge, reading_color)

        draw_text(screen, f"command     {command}", (370, 140), font, MUTED)
        draw_text(
            screen,
            f"poll rate   {1 / POLL_INTERVAL:.0f} Hz   (sensor {DISTANCE_SENSOR_ID})",
            (370, 162),
            font,
            MUTED,
        )
        draw_text(screen, f"samples     {len(history)}", (370, 184), font, MUTED)

        # Say plainly which robot is being driven, so nobody spends ten
        # minutes wondering why the real one is not moving.
        draw_text(
            screen,
            f"robot       {'real' if USE_REAL_ROBOT else 'simulated'}",
            (370, 206),
            font,
            MUTED,
        )

        if too_close:
            draw_text(screen, "TOO CLOSE", (700, 140), font_big, DANGER)

        draw_history(screen, history, graph_rect, font_small)
        draw_text(
            screen,
            f"last {HISTORY_LEN * POLL_INTERVAL:.0f} seconds ({DISTANCE_UNITS})",
            (370, 480),
            font_small,
            MUTED,
        )

        pygame.display.flip()

    robot.mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Run it as it is. The whole panel works - buttons, speeds, graph -
#      with no robot anywhere. Then set USE_REAL_ROBOT = True and run
#      it again. One word changed.
#   2. Search the file for "ugot". It appears twice: once in the import
#      inside connect_robot(), once in the comment above. Everything
#      else is robot-shaped, not ugot-shaped.
#   3. Make the fake sensor fail sometimes. Add this to the top of
#      FakeRobot.read_distance_data:
#          if random.random() < 0.05:
#              raise RuntimeError("no echo")
#      Now you can test the error handling from stage 8 without pulling
#      any wires out.
#   4. Drive the fake robot forward. The graph does not react, because
#      FakeRobot does not pretend to have a world around it. Could you
#      make it? What would it have to remember?
#   5. Delete read_distance_data from FakeRobot and run it. Read the
#      AttributeError. That is the deal with duck typing: nothing checks
#      the two classes match until the moment a method is called.
# ---------------------------------------------------------------------
