"""
Robot Control Panel
===================
A small Pygame GUI for driving a UGOT mecanum robot and watching its
distance sensor.

Run with:   python robot_panel.py
Needs:      pip install pygame

Controls:
    Arrow keys / WASD   drive
    Space               stop
    + / -               drive speed
    [ / ]               turn speed
    Esc                 quit

TO CONNECT THE REAL ROBOT:
    1. Put your robot's IP address in ROBOT_IP below.
    2. Set USE_REAL_ROBOT = True.
Nothing else changes. The program talks to one object called `robot`,
which is either a real ugot.UGOT() or the FakeRobot stand-in, and both
have the same method names.
"""

import math
import random
from collections import deque

import pygame

# ---------------------------------------------------------------------
# ROBOT CONNECTION
# ---------------------------------------------------------------------

USE_REAL_ROBOT = False
ROBOT_IP = "192.168.0.1"  # <-- your robot's address


class FakeRobot:
    """Pretends to be ugot.UGOT() so the GUI runs with no robot attached.

    Every method here matches the real one's name and arguments.
    """

    def initialize(self, ip):
        print(f"[fake] initialize({ip!r})")

    def mecanum_stop(self):
        print("[fake] mecanum_stop()")

    def mecanum_move_speed(self, direction, speed):
        """direction: 0 forward, 1 backward.  speed: 5-80 cm/s."""
        print(f"[fake] mecanum_move_speed({direction}, {speed})")

    def mecanum_turn_speed(self, turn, speed):
        """turn: 2 left, 3 right.  speed: 5-280 deg/s."""
        print(f"[fake] mecanum_turn_speed({turn}, {speed})")

    def read_distance_data(self):
        """Return one distance reading as a float.

        The fake value sweeps in and out so the display has something
        to show.
        """
        t = pygame.time.get_ticks() / 1000.0
        distance = 100.0 + 70.0 * math.sin(t * 0.7) + 15.0 * math.sin(t * 3.1)
        return max(3.0, distance + random.uniform(-2.0, 2.0))


def connect_robot():
    """Build the robot object the rest of the program will use.

    This is the same three lines you normally type, just tucked into a
    function so there is one obvious place to change them.
    """
    if USE_REAL_ROBOT:
        from ugot import ugot

        robot = ugot.UGOT()
    else:
        robot = FakeRobot()

    robot.initialize(ROBOT_IP)
    return robot


# Named values from the robot's documentation, so no magic numbers
# are scattered through the code.
DIR_FORWARD = 0
DIR_BACKWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3

DRIVE_MIN, DRIVE_MAX = 5, 80  # cm/s, from mecanum_move_speed docs
TURN_MIN, TURN_MAX = 5, 280  # deg/s, from mecanum_turn_speed docs


# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------

WINDOW_W, WINDOW_H = 900, 520
FPS = 60

POLL_INTERVAL = 0.1  # seconds between sensor reads (10 times/second)
HISTORY_LEN = 120  # samples kept for the graph (12 seconds at 10 Hz)

# Check these two against your own sensor.
DISTANCE_UNITS = "cm"
DISTANCE_MAX = 200.0  # top of the graph
TOO_CLOSE = 25.0  # warn below this

BG = (24, 26, 32)
PANEL = (34, 37, 46)
BTN = (52, 57, 70)
BTN_HOVER = (68, 74, 90)
BTN_EDGE = (80, 86, 104)
TEXT = (232, 234, 240)
MUTED = (138, 145, 163)
ACCENT = (80, 205, 165)
DANGER = (232, 92, 92)


# ---------------------------------------------------------------------
# A CLICKABLE BUTTON
# ---------------------------------------------------------------------


class Button:
    def __init__(self, rect, label, command):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.command = command

    def is_over(self, pos):
        """True if the point (a mouse position) is inside this button."""
        return self.rect.collidepoint(pos)

    def draw(self, surface, font, active=False):
        if active:
            fill, label_color = ACCENT, (16, 22, 28)
        elif self.is_over(pygame.mouse.get_pos()):
            fill, label_color = BTN_HOVER, TEXT
        else:
            fill, label_color = BTN, TEXT

        pygame.draw.rect(surface, fill, self.rect, border_radius=8)
        pygame.draw.rect(surface, BTN_EDGE, self.rect, width=2, border_radius=8)
        draw_text(surface, self.label, self.rect.center, font, label_color, center=True)


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def clamp(value, low, high):
    """Keep a number inside the range the robot will accept."""
    return max(low, min(value, high))


def draw_text(surface, text, pos, font, color=TEXT, center=False):
    image = font.render(text, True, color)
    rect = image.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)


def draw_history(surface, history, rect, font):
    """Draw the recent distance readings as a line graph."""
    pygame.draw.rect(surface, PANEL, rect, border_radius=8)
    pygame.draw.rect(surface, BTN_EDGE, rect, width=1, border_radius=8)

    def y_for(value):
        """Turn a distance into a y position inside the graph."""
        fraction = clamp(value, 0.0, DISTANCE_MAX) / DISTANCE_MAX
        return rect.bottom - fraction * rect.height

    # Horizontal grid lines with labels
    for value in (0, DISTANCE_MAX / 2, DISTANCE_MAX):
        y = y_for(value)
        pygame.draw.line(surface, BTN, (rect.left, y), (rect.right, y), 1)
        draw_text(surface, f"{value:.0f}", (rect.left + 6, y - 15), font, MUTED)

    # The "too close" threshold
    y = y_for(TOO_CLOSE)
    pygame.draw.line(surface, DANGER, (rect.left, y), (rect.right, y), 1)

    if len(history) < 2:
        return

    # Newest sample sits at the right-hand edge
    step = rect.width / (HISTORY_LEN - 1)
    points = []
    for i, value in enumerate(history):
        x = rect.right - (len(history) - 1 - i) * step
        points.append((x, y_for(value)))

    pygame.draw.lines(surface, ACCENT, False, points, 2)
    pygame.draw.circle(surface, ACCENT, (int(points[-1][0]), int(points[-1][1])), 4)


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()

    try:
        robot = connect_robot()
    except Exception as error:
        print(f"Could not connect to the robot at {ROBOT_IP}: {error}")
        pygame.quit()
        return

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

    keymap = {
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

    command = "Stop"  # what the robot was last told to do
    drive_speed = 30  # cm/s
    turn_speed = 90  # deg/s

    history = deque(maxlen=HISTORY_LEN)
    distance = None  # None until the first successful reading
    sensor_error = False
    time_since_poll = 0.0

    def issue(name):
        """Translate a command name into one robot library call."""
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
        nonlocal command
        command = name
        issue(name)

    def change_drive(delta):
        nonlocal drive_speed
        drive_speed = int(clamp(drive_speed + delta, DRIVE_MIN, DRIVE_MAX))
        if command in ("Forward", "Back"):
            issue(command)  # re-send so the new speed takes effect

    def change_turn(delta):
        nonlocal turn_speed
        turn_speed = int(clamp(turn_speed + delta, TURN_MIN, TURN_MAX))
        if command in ("Left", "Right"):
            issue(command)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds since the last frame

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
                elif event.key in keymap:
                    set_command(keymap[event.key])

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

        # ---- 2. read the sensor, but not every single frame ---------
        time_since_poll += dt
        if time_since_poll >= POLL_INTERVAL:
            time_since_poll = 0.0
            try:
                reading = float(robot.read_distance_data())
                sensor_error = False
                if reading > 0:  # ignore "no echo" style values
                    distance = reading
                    history.append(reading)
            except Exception as error:  # keep the GUI alive if it fails
                sensor_error = True
                print(f"[sensor] read failed: {error}")

        # ---- 3. draw ------------------------------------------------
        screen.fill(BG)
        pygame.draw.rect(screen, PANEL, (0, 0, 340, WINDOW_H))

        draw_text(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(
            screen, "arrows / WASD to drive, space to stop", (24, 52), font_small, MUTED
        )

        for button in buttons:
            button.draw(screen, font, active=(button.command == command))

        draw_text(screen, f"DRIVE  {drive_speed} cm/s", (40, 324), font, MUTED)
        draw_text(screen, f"TURN   {turn_speed} deg/s", (40, 398), font, MUTED)

        draw_text(screen, "DISTANCE SENSOR", (370, 22), font_big)

        if sensor_error:
            reading_text, reading_color = "-- error --", DANGER
        elif distance is None:
            reading_text, reading_color = "-- . -", MUTED
        else:
            reading_text = f"{distance:.1f} {DISTANCE_UNITS}"
            reading_color = DANGER if distance < TOO_CLOSE else ACCENT
        draw_text(screen, reading_text, (370, 56), font_huge, reading_color)

        status = [
            f"command     {command}",
            f"poll rate   {1 / POLL_INTERVAL:.0f} Hz",
            f"samples     {len(history)}",
            f"robot       {'real' if USE_REAL_ROBOT else 'simulated'}",
        ]
        for i, line in enumerate(status):
            draw_text(screen, line, (370, 122 + i * 22), font, MUTED)

        if distance is not None and distance < TOO_CLOSE:
            draw_text(screen, "TOO CLOSE", (700, 130), font_big, DANGER)

        graph = pygame.Rect(370, 240, 500, 230)
        draw_history(screen, history, graph, font_small)
        draw_text(
            screen,
            f"last {HISTORY_LEN * POLL_INTERVAL:.0f} seconds ({DISTANCE_UNITS})",
            (370, 478),
            font_small,
            MUTED,
        )

        pygame.display.flip()

    robot.mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()
