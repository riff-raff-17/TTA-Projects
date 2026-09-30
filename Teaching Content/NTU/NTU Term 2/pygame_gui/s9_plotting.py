"""
Stage 9 - Plotting the history
==============================
One number tells you the distance now. A line tells you what the robot
has been doing, which is far more useful when you are debugging.

Two parts to this:

  * STORING. A deque with maxlen is a list that forgets. Append to it
    forever and it quietly drops the oldest item once it is full, so
    the program can run all day without eating memory.

  * DRAWING. The readings are in centimetres, but the screen only
    understands pixels. Everything in this stage comes down to one
    little function, y_for(), that converts one into the other.

Converting a value into a position is the skill here. Every graph,
progress bar, thermometer and dial is the same three steps:

    value  ->  fraction of the way between min and max  ->  pixel

Run with:   python stage_09.py
"""

from collections import deque

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
# No robot today? Replace each body with the print() line from stage 4,
# and have read_distance_data return something made up, like
#     return 100 + 50 * math.sin(pygame.time.get_ticks() / 1000)
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


def read_distance_data(sensor_id):
    """Return one distance reading from that sensor, as a float."""
    return robot.read_distance_data(sensor_id)


DIR_FORWARD  = 0
DIR_BACKWARD = 1
TURN_LEFT    = 2
TURN_RIGHT   = 3

DISTANCE_SENSOR_ID = 1        # <-- check which sensor yours is

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

POLL_INTERVAL = 0.1           # seconds between sensor reads
HISTORY_LEN = 120             # readings kept: 120 at 10 a second = 12 seconds

DISTANCE_UNITS = "cm"
DISTANCE_MAX = 200.0          # the top of the graph
TOO_CLOSE = 25.0              # warn below this

BG        = (24, 26, 32)
PANEL     = (34, 37, 46)
BTN       = (52, 57, 70)
BTN_HOVER = (68, 74, 90)
BTN_EDGE  = (80, 86, 104)
TEXT      = (232, 234, 240)
MUTED     = (138, 145, 163)
ACCENT    = (80, 205, 165)
DANGER    = (232, 92, 92)
INK       = (14, 20, 26)

KEYMAP = {
    pygame.K_UP: "Forward",     pygame.K_w: "Forward",
    pygame.K_DOWN: "Back",      pygame.K_s: "Back",
    pygame.K_LEFT: "Left",      pygame.K_a: "Left",
    pygame.K_RIGHT: "Right",    pygame.K_d: "Right",
    pygame.K_SPACE: "Stop",
}


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def clamp(value, low, high):
    """Keep a number inside a range. Used to stop the line leaving the box."""
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
        """Turn a distance into a y position inside the graph.

        A reading of 0 belongs at the BOTTOM of the box and DISTANCE_MAX
        at the top, but screen y grows downwards - so we measure down
        from rect.bottom instead of up from rect.top.
        """
        fraction = clamp(value, 0.0, DISTANCE_MAX) / DISTANCE_MAX
        return rect.bottom - fraction * rect.height

    # Grid lines, labelled, so the shape means something
    for value in (0, DISTANCE_MAX / 2, DISTANCE_MAX):
        y = y_for(value)
        pygame.draw.line(surface, BTN, (rect.left, y), (rect.right, y), 1)
        draw_text(surface, f"{value:.0f}", (rect.left + 6, y - 15), font, MUTED)

    # The warning threshold, so you can see readings dip below it
    y = y_for(TOO_CLOSE)
    pygame.draw.line(surface, DANGER, (rect.left, y), (rect.right, y), 1)

    # Two points is the minimum needed to draw a line
    if len(history) < 2:
        return

    # Work out the screen position of every reading. The newest sits at
    # the right edge, so a half-full graph fills from the right and the
    # line appears to scroll.
    step = rect.width / (HISTORY_LEN - 1)
    points = []
    for i, value in enumerate(history):     # enumerate gives position AND value
        x = rect.right - (len(history) - 1 - i) * step
        points.append((x, y_for(value)))

    pygame.draw.lines(surface, ACCENT, False, points, 2)

    # A dot on the newest reading, to show which end is now
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
        draw_text(surface, self.label, self.rect.center, font,
                  label_color, center=True)


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
        Button((130, 110, 80, 60), "FWD",   "Forward"),
        Button(( 40, 180, 80, 60), "LEFT",  "Left"),
        Button((130, 180, 80, 60), "STOP",  "Stop"),
        Button((220, 180, 80, 60), "RIGHT", "Right"),
        Button((130, 250, 80, 60), "BACK",  "Back"),
    ]

    graph_rect = pygame.Rect(370, 245, 500, 225)

    command = "Stop"
    issue(command)

    distance = None
    sensor_error = False
    time_since_poll = 0.0

    # maxlen makes this forget the oldest reading once it is full.
    history = deque(maxlen=HISTORY_LEN)

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
                elif event.key in KEYMAP:
                    command = KEYMAP[event.key]
                    issue(command)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in buttons:
                    if button.is_over(event.pos):
                        command = button.command
                        issue(command)

        # ---- 2. update: read the sensor, but not every frame --------
        time_since_poll += dt
        if time_since_poll >= POLL_INTERVAL:
            time_since_poll = 0.0
            try:
                reading = float(read_distance_data(DISTANCE_SENSOR_ID))
                sensor_error = False
                if reading > 0:
                    distance = reading
                    history.append(reading)      # <-- remember it
            except Exception as error:
                sensor_error = True
                print(f"sensor read failed: {error}")

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
        draw_text(screen, f"poll rate   {1 / POLL_INTERVAL:.0f} Hz"
                          f"   (sensor {DISTANCE_SENSOR_ID})",
                  (370, 162), font, MUTED)
        draw_text(screen, f"samples     {len(history)}", (370, 184), font, MUTED)

        if too_close:
            draw_text(screen, "TOO CLOSE", (700, 140), font_big, DANGER)

        draw_history(screen, history, graph_rect, font_small)
        draw_text(screen, f"last {HISTORY_LEN * POLL_INTERVAL:.0f} seconds "
                          f"({DISTANCE_UNITS})", (370, 480), font_small, MUTED)

        pygame.display.flip()

    mecanum_stop()
    pygame.quit()


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# TRY IT
#   1. Wave your hand towards and away from the sensor and watch the
#      trace. Then drive the robot at a wall and watch the line fall.
#   2. Watch the "samples" number climb to 120 and stop. That is maxlen
#      doing its job. Now remove maxlen=HISTORY_LEN so it is a plain
#      deque() and leave it running for a few minutes. The number grows
#      forever, and the graph squashes up against the right edge.
#   3. In y_for, change the last line to
#          return rect.top + fraction * rect.height
#      The graph now runs upside down. Explain to someone else why.
#   4. Set DISTANCE_MAX to 50. Readings above 50 are flattened against
#      the top by clamp(). Why is clamping better here than letting the
#      line shoot out of the box?
#   5. Make the graph show one minute instead of twelve seconds. Which
#      constant did you change, and did anything else need touching?
# ---------------------------------------------------------------------