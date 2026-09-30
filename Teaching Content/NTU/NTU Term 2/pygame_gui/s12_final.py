"""
Stage 12 - Making it feel good
==============================
The panel already does everything. This stage is about how it feels,
and almost all of it comes from two small functions:

    mix(a, b, amount)              blend between two colors
    approach(now, target, speed, dt)   move part of the way there

That second one is the whole trick behind interfaces that feel alive.
Instead of snapping a value straight to where it should be, move it
maybe 15% of the remaining distance each frame. It arrives quickly,
slows as it gets close, and never jumps. Three lines, applied to a
button's brightness, a bar's length, a fading ring.

Two rules kept throughout:

  * Animation decorates, it never lies. The big number is always the
    exact reading. The smoothed bar underneath is the decoration.
  * Nothing expensive per frame. The gradient is painted once at
    startup. No new images, no transparency layers - just numbers
    easing toward other numbers.

Run with:   python stage_12.py
"""

import math
import random
from collections import deque

import pygame

# ---------------------------------------------------------------------
# WHICH ROBOT?
# ---------------------------------------------------------------------

USE_REAL_ROBOT = True  # <-- the switch
ROBOT_IP = "192.168.88.1"  # <-- your robot's address goes here


class FakeRobot:
    """Pretends to be ugot.UGOT() so the panel runs with no robot."""

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
    """Build whichever robot the switch asks for, and start it up."""
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

DISTANCE_SENSOR_ID = 21  # <-- check which sensor yours is

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

# A few more colors than before: two for the gradient, a brighter
# accent for highlights, a dim red so the warning has something to
# pulse between.
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
# THE TWO NEW HELPERS
# ---------------------------------------------------------------------


def clamp(value, low, high):
    """Keep a number inside a range."""
    return max(low, min(value, high))


def mix(color_a, color_b, amount):
    """Blend two colors.

    amount=0 gives color_a, amount=1 gives color_b, 0.5 gives halfway.
    zip() pairs up the reds, the greens and the blues, and each pair is
    blended the same way.
    """
    amount = clamp(amount, 0.0, 1.0)
    return tuple(int(a + (b - a) * amount) for a, b in zip(color_a, color_b))


def approach(current, target, speed, dt):
    """Move a number part of the way toward a target.

    Called every frame, this produces smooth movement that starts fast
    and eases in. speed is roughly "how many times per second it closes
    the gap" - higher is snappier.

    Multiplying by dt is what keeps it the same on a fast and a slow
    computer: a longer frame moves further.
    """
    return current + (target - current) * clamp(speed * dt, 0.0, 1.0)


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


def make_background(size):
    """Paint the gradient ONCE, at startup.

    520 thin rectangles is slow if you do it 60 times a second, and
    free if you do it once and keep the picture.
    """
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
    """A thin bar showing how full something is.

    This is the same value-to-pixels idea as the graph, in one line.
    """
    pygame.draw.rect(surface, BTN, rect, border_radius=3)
    filled = pygame.Rect(
        rect.left, rect.top, max(3, int(rect.width * fraction)), rect.height
    )
    pygame.draw.rect(surface, color, filled, border_radius=3)


def draw_history(surface, history, rect, font, flash):
    """Draw the recent readings as a filled line graph inside `rect`."""
    pygame.draw.rect(surface, PANEL, rect, border_radius=8)
    pygame.draw.rect(surface, PANEL_EDGE, rect, width=1, border_radius=8)

    def y_for(value):
        """Turn a distance into a y position inside the graph."""
        fraction = clamp(value, 0.0, DISTANCE_MAX) / DISTANCE_MAX
        return rect.bottom - fraction * rect.height

    for value in (0, DISTANCE_MAX / 2, DISTANCE_MAX):
        y = y_for(value)
        pygame.draw.line(surface, BTN, (rect.left, y), (rect.right, y), 1)
        draw_text(surface, f"{value:.0f}", (rect.left + 6, y - 15), font, MUTED)

    y = y_for(TOO_CLOSE)
    pygame.draw.line(surface, DANGER_DIM, (rect.left, y), (rect.right, y), 1)

    if len(history) < 2:
        return

    step = rect.width / (HISTORY_LEN - 1)
    points = []
    for i, value in enumerate(history):
        x = rect.right - (len(history) - 1 - i) * step
        points.append((x, y_for(value)))

    # A filled shape under the line: the same points, plus two corners
    # dropped to the bottom edge to close it off.
    skirt = [(points[0][0], rect.bottom)] + points + [(points[-1][0], rect.bottom)]
    pygame.draw.polygon(surface, mix(PANEL, ACCENT, 0.14), skirt)
    pygame.draw.lines(surface, ACCENT, False, points, 2)

    # The newest point pings outward each time a reading arrives. If
    # the ping stops, the sensor has stopped - a heartbeat, for free.
    head = (int(points[-1][0]), int(points[-1][1]))
    if flash > 0.02:
        pygame.draw.circle(
            surface, mix(PANEL, ACCENT, flash * 0.7), head, int(5 + 9 * flash), 1
        )
    pygame.draw.circle(surface, ACCENT_BRIGHT, head, 4)


# ---------------------------------------------------------------------
# THE BUTTON CLASS
# ---------------------------------------------------------------------


class Button:
    """One button, now with a little life in it.

    Two new values. Both are just numbers between 0 and 1 that get
    nudged every frame by approach().
    """

    def __init__(self, rect, label, command):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.command = command
        self.glow = 0.0  # 0 = idle, 1 = I am the active command
        self.press = 0.0  # briefly 1 right after a click

    def is_over(self, pos):
        """Is this point - usually the mouse - inside me?"""
        return self.rect.collidepoint(pos)

    def bump(self):
        """Called on click, so the button can flash."""
        self.press = 1.0

    def update(self, dt, active):
        """Ease my animation values toward where they should be."""
        hovered = self.is_over(pygame.mouse.get_pos())
        target = 1.0 if active else (0.3 if hovered else 0.0)
        self.glow = approach(self.glow, target, 14, dt)
        self.press = approach(self.press, 0.0, 9, dt)

    def draw(self, surface, font):
        """Draw myself, using whatever my animation values are now.

        Notice there is no if/elif for idle / hover / active any more.
        One number, glow, slides between them, and mix() turns that
        number into a color.
        """
        squeeze = int(4 * self.press)  # shrink slightly when clicked
        rect = self.rect.inflate(-squeeze, -squeeze)

        fill = mix(BTN, ACCENT, self.glow)
        edge = mix(BTN_EDGE, ACCENT_BRIGHT, self.glow)
        label_color = mix(TEXT, INK, self.glow)

        pygame.draw.rect(surface, fill, rect, border_radius=8)
        pygame.draw.rect(surface, edge, rect, width=2, border_radius=8)
        draw_text(surface, self.label, rect.center, font, label_color, center=True)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Robot Control Panel")
    clock = pygame.time.Clock()
    background = make_background((WINDOW_W, WINDOW_H))

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

    # Animation values live with the rest of the state
    gauge = 0.0  # smoothed bar position
    sample_flash = 0.0  # ping when a new reading lands

    def issue(name):
        """Turn a command name into exactly one robot library call."""
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

        # One shared heartbeat, running from 0 to 1 and back, about
        # twice a second. Everything that pulses reads this.
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
                elif event.key in KEYMAP:
                    set_command(KEYMAP[event.key])
                    # Flash the matching button, so the keyboard and the
                    # mouse feel like the same panel
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

        # ---- 2. update: sensor, then animation ----------------------
        time_since_poll += dt
        if time_since_poll >= POLL_INTERVAL:
            time_since_poll = 0.0
            try:
                reading = float(robot.read_distance_data(DISTANCE_SENSOR_ID))
                sensor_error = False
                if reading > 0:
                    distance = reading
                    history.append(reading)
                    sample_flash = 1.0  # start the ping
            except Exception as error:
                sensor_error = True
                print(f"sensor read failed: {error}")

        for button in buttons:
            button.update(dt, active=(button.command == command))

        sample_flash = approach(sample_flash, 0.0, 5, dt)
        gauge_target = 0.0 if distance is None else clamp(distance / DISTANCE_MAX, 0, 1)
        gauge = approach(gauge, gauge_target, 8, dt)

        too_close = distance is not None and distance < TOO_CLOSE
        alert = mix(DANGER_DIM, DANGER, pulse)  # breathing red

        # ---- 3. draw ------------------------------------------------
        screen.blit(background, (0, 0))  # the pre-painted gradient
        pygame.draw.rect(screen, PANEL, (0, 0, PANEL_W, WINDOW_H))
        pygame.draw.line(screen, PANEL_EDGE, (PANEL_W, 0), (PANEL_W, WINDOW_H), 1)

        draw_heading(screen, "ROBOT CONTROL", (24, 22), font_big)
        draw_text(screen, "click, or use arrows / WASD", (24, 66), font_small, MUTED)

        # Connection light: steady green for real, slow amber for fake
        light = ACCENT if USE_REAL_ROBOT else mix(AMBER, PANEL, pulse * 0.5)
        pygame.draw.circle(screen, light, (300, 32), 6)

        for button in buttons:
            button.draw(screen, font)

        draw_text(screen, f"DRIVE  {drive_speed} cm/s", (40, 324), font, MUTED)
        draw_gauge(
            screen,
            pygame.Rect(40, 390, 260, 4),
            (drive_speed - DRIVE_MIN) / (DRIVE_MAX - DRIVE_MIN),
            BTN_EDGE,
        )

        draw_text(screen, f"TURN   {turn_speed} deg/s", (40, 400), font, MUTED)
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
            reading_text, reading_color = "--- . -", MUTED
        else:
            # The NUMBER is always the exact reading. Only the bar
            # underneath is smoothed.
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

        draw_history(screen, history, graph_rect, font_small, sample_flash)

        # min / avg / max are free once the readings are in a list, and
        # genuinely useful when you are calibrating a sensor.
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


# ---------------------------------------------------------------------
# TRY IT
#   1. In Button.update, change the 14 in the glow line to 2, then to
#      60. Slow feels sluggish, fast feels like no animation at all.
#      Find the number you like - that is a real design decision, and
#      there is no correct answer.
#   2. Delete `* dt` from inside approach(). It still looks fine on
#      your machine. Now set FPS to 15 and compare: without dt, the
#      animation speed depends on the frame rate.
#   3. Comment out the button.update() loop. Everything still works,
#      just frozen - proof that the animation layer is separate from
#      the program underneath it.
#   4. Make the STOP button pulse gently whenever the robot is driving,
#      so your eye is drawn to it. You have `pulse` already.
#   5. Change BG_TOP and BG_BOTTOM to two colors of your own and rerun.
#      Then try making the gradient go left to right instead of top to
#      bottom - one loop in make_background().
#
# WHERE TO GO NEXT
#   * Auto-stop: refuse to drive forward when distance < TOO_CLOSE.
#     Your first piece of autonomy, and about four lines.
#   * Log every reading to a CSV file with a timestamp, then plot it
#     in a spreadsheet.
#   * Give FakeRobot a pretend wall that gets closer as you drive, so
#     the simulator responds to your driving (stage 11, exercise 4).
#   * A second sensor: make `distance` and `history` dicts keyed by
#     sensor id, and loop over a list of ids in the poll block.
# ---------------------------------------------------------------------
