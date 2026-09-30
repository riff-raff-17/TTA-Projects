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

    def read_distance_data(self, sensor_id):
        """Return one distance reading from the given sensor, as a float.

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

DISTANCE_SENSOR_ID = 1  # <-- which sensor read_distance_data() reads

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

BG_TOP = (30, 33, 42)
BG_BOTTOM = (15, 16, 21)
PANEL = (34, 37, 46)
PANEL_EDGE = (58, 63, 78)
BTN = (52, 57, 70)
BTN_EDGE = (80, 86, 104)
TEXT = (232, 234, 240)
MUTED = (138, 145, 163)
ACCENT = (80, 205, 165)
ACCENT_BRIGHT = (150, 245, 212)
AMBER = (235, 185, 95)
DANGER = (232, 92, 92)
DANGER_DIM = (150, 60, 60)
INK = (14, 20, 26)  # dark text for use on bright buttons


# ---------------------------------------------------------------------
# SMALL MATH HELPERS
# ---------------------------------------------------------------------


def clamp(value, low, high):
    """Keep a number inside a range."""
    return max(low, min(value, high))


def mix(color_a, color_b, amount):
    """Blend two colors. amount=0 gives color_a, amount=1 gives color_b."""
    amount = clamp(amount, 0.0, 1.0)
    return tuple(int(a + (b - a) * amount) for a, b in zip(color_a, color_b))


def approach(current, target, speed, dt):
    """Move a number part of the way toward a target. Used for animation."""
    return current + (target - current) * clamp(speed * dt, 0.0, 1.0)


# ---------------------------------------------------------------------
# A CLICKABLE BUTTON
# ---------------------------------------------------------------------


class Button:
    def __init__(self, rect, label, command):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.command = command
        self.glow = 0.0  # 0 = idle, 1 = this is the active command
        self.press = 0.0  # briefly 1 right after a click

    def is_over(self, pos):
        """True if the point (a mouse position) is inside this button."""
        return self.rect.collidepoint(pos)

    def bump(self):
        """Called on click, so the button can flash."""
        self.press = 1.0

    def update(self, dt, active):
        """Ease the animation values toward where they should be."""
        hovered = self.is_over(pygame.mouse.get_pos())
        target = 1.0 if active else (0.3 if hovered else 0.0)
        self.glow = approach(self.glow, target, 14, dt)
        self.press = approach(self.press, 0.0, 9, dt)

    def draw(self, surface, font):
        # Shrinking slightly on click makes it feel physical
        squeeze = int(4 * self.press)
        rect = self.rect.inflate(-squeeze, -squeeze)

        fill = mix(BTN, ACCENT, self.glow)
        edge = mix(BTN_EDGE, ACCENT_BRIGHT, self.glow)
        label_color = mix(TEXT, INK, self.glow)

        pygame.draw.rect(surface, fill, rect, border_radius=8)
        pygame.draw.rect(surface, edge, rect, width=2, border_radius=8)
        draw_text(surface, self.label, rect.center, font, label_color, center=True)


# ---------------------------------------------------------------------
# DRAWING HELPERS
# ---------------------------------------------------------------------


def draw_text(surface, text, pos, font, color=TEXT, center=False):
    image = font.render(text, True, color)
    rect = image.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)


def make_background(size):
    """Paint the gradient once at startup instead of every frame."""
    surface = pygame.Surface(size)
    width, height = size
    for y in range(height):
        surface.fill(mix(BG_TOP, BG_BOTTOM, y / height), (0, y, width, 1))
    return surface


def draw_heading(surface, text, pos, font):
    """A title with a short accent rule underneath it."""
    draw_text(surface, text, pos, font)
    x, y = pos
    pygame.draw.line(surface, ACCENT, (x, y + 30), (x + 46, y + 30), 3)


def draw_gauge(surface, rect, fraction, color):
    """A thin bar showing how full something is."""
    pygame.draw.rect(surface, BTN, rect, border_radius=3)
    filled = pygame.Rect(
        rect.left, rect.top, max(3, int(rect.width * fraction)), rect.height
    )
    pygame.draw.rect(surface, color, filled, border_radius=3)


def draw_history(surface, history, rect, font, flash):
    """Draw the recent distance readings as a filled line graph."""
    pygame.draw.rect(surface, PANEL, rect, border_radius=8)
    pygame.draw.rect(surface, PANEL_EDGE, rect, width=1, border_radius=8)

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
    pygame.draw.line(surface, DANGER_DIM, (rect.left, y), (rect.right, y), 1)

    if len(history) < 2:
        return

    # Newest sample sits at the right-hand edge
    step = rect.width / (HISTORY_LEN - 1)
    points = []
    for i, value in enumerate(history):
        x = rect.right - (len(history) - 1 - i) * step
        points.append((x, y_for(value)))

    # Soft fill under the trace, then the trace itself
    skirt = [(points[0][0], rect.bottom)] + points + [(points[-1][0], rect.bottom)]
    pygame.draw.polygon(surface, mix(PANEL, ACCENT, 0.14), skirt)
    pygame.draw.lines(surface, ACCENT, False, points, 2)

    # The newest point pings outward each time a sample arrives
    head = (int(points[-1][0]), int(points[-1][1]))
    if flash > 0.02:
        pygame.draw.circle(
            surface, mix(PANEL, ACCENT, flash * 0.7), head, int(5 + 9 * flash), 1
        )
    pygame.draw.circle(surface, ACCENT_BRIGHT, head, 4)


# ---------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()
    background = make_background((WINDOW_W, WINDOW_H))

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

    gauge = 0.0  # smoothed bar position
    sample_flash = 0.0  # ping when a new reading lands

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
        pulse = 0.5 + 0.5 * math.sin(pygame.time.get_ticks() / 1000.0 * 4)

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
                    for button in buttons:
                        if button.command == command:
                            button.bump()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if not button.is_over(event.pos):
                        continue
                    button.bump()
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
                reading = float(robot.read_distance_data(DISTANCE_SENSOR_ID))
                sensor_error = False
                if reading > 0:  # ignore "no echo" style values
                    distance = reading
                    history.append(reading)
                    sample_flash = 1.0
            except Exception as error:  # keep the GUI alive if it fails
                sensor_error = True
                print(f"[sensor] read failed: {error}")

        # ---- 3. update the animated values --------------------------
        for button in buttons:
            button.update(dt, active=(button.command == command))

        sample_flash = approach(sample_flash, 0.0, 5, dt)
        gauge_target = 0.0 if distance is None else clamp(distance / DISTANCE_MAX, 0, 1)
        gauge = approach(gauge, gauge_target, 8, dt)

        too_close = distance is not None and distance < TOO_CLOSE
        alert = mix(DANGER_DIM, DANGER, pulse)

        # ---- 4. draw ------------------------------------------------
        screen.blit(background, (0, 0))
        pygame.draw.rect(screen, PANEL, (0, 0, 340, WINDOW_H))
        pygame.draw.line(screen, PANEL_EDGE, (340, 0), (340, WINDOW_H), 1)

        draw_heading(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(
            screen, "arrows / WASD to drive, space to stop", (24, 66), font_small, MUTED
        )

        # Connection light: steady green for real, slow amber for simulated
        light = ACCENT if USE_REAL_ROBOT else mix(AMBER, PANEL, pulse * 0.5)
        pygame.draw.circle(screen, light, (300, 32), 6)

        for button in buttons:
            button.draw(screen, font)

        draw_text(screen, f"DRIVE  {drive_speed} cm/s", (40, 324), font, MUTED)
        draw_gauge(
            screen,
            pygame.Rect(40, 392, 260, 4),
            (drive_speed - DRIVE_MIN) / (DRIVE_MAX - DRIVE_MIN),
            BTN_EDGE,
        )
        draw_text(screen, f"TURN   {turn_speed} deg/s", (40, 398), font, MUTED)
        draw_gauge(
            screen,
            pygame.Rect(40, 466, 260, 4),
            (turn_speed - TURN_MIN) / (TURN_MAX - TURN_MIN),
            BTN_EDGE,
        )

        draw_heading(screen, "DISTANCE SENSOR", (370, 22), font_big)

        if sensor_error:
            reading_text, reading_color = "-- error --", alert
        elif distance is None:
            reading_text, reading_color = "-- . -", MUTED
        else:
            reading_text = f"{distance:.1f} {DISTANCE_UNITS}"
            reading_color = alert if too_close else ACCENT
        draw_text(screen, reading_text, (370, 62), font_huge, reading_color)

        draw_gauge(
            screen, pygame.Rect(370, 120, 500, 5), gauge, alert if too_close else ACCENT
        )

        status = [
            f"command     {command}",
            f"poll rate   {1 / POLL_INTERVAL:.0f} Hz   (sensor {DISTANCE_SENSOR_ID})",
            f"samples     {len(history)}",
            f"robot       {'real' if USE_REAL_ROBOT else 'simulated'}",
        ]
        for i, line in enumerate(status):
            draw_text(screen, line, (370, 140 + i * 22), font, MUTED)

        if too_close:
            draw_text(screen, "TOO CLOSE", (700, 150), font_big, alert)

        graph = pygame.Rect(370, 245, 500, 225)
        draw_history(screen, history, graph, font_small, sample_flash)

        caption = f"last {HISTORY_LEN * POLL_INTERVAL:.0f} s ({DISTANCE_UNITS})"
        if history:
            caption += (
                f"    min {min(history):5.1f}"
                f"    avg {sum(history) / len(history):5.1f}"
                f"    max {max(history):5.1f}"
            )
        draw_text(screen, caption, (370, 480), font_small, MUTED)

        pygame.display.flip()

    robot.mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()
