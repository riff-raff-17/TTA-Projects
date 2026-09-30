"""
Stage 8 - Reading the sensor
============================
Time for the other half of the panel.

The obvious thing would be to call read_distance_data() once per frame.
Don't. The screen redraws 60 times a second, and a distance sensor
takes time to send a pulse and wait for the echo. Asking it that often
gives you nonsense, slows the whole loop down, or both.

So we keep a stopwatch. Every frame we add on how long the frame took,
and only when enough time has piled up do we actually read the sensor.
Ten times a second is plenty for a human to look at.

New ideas:
  * dt - how long the last frame took, in seconds
  * using dt to make something happen on a schedule
  * try / except, so a failed reading does not kill the program
  * None as "no value yet", which is different from 0

Run with:   python stage_08.py
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

POLL_INTERVAL = 0.1           # seconds between sensor reads, so 10 a second

# Check these against your own sensor before trusting the display.
DISTANCE_UNITS = "cm"
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

    # Sensor state. None means "we have not managed to read it yet",
    # which is not the same as 0 - zero would be a real measurement.
    distance = None
    sensor_error = False
    time_since_poll = 0.0

    running = True
    while running:
        # tick() gives back how many MILLIseconds the last frame took.
        # Divide by 1000 to get seconds, which is easier to think in.
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
                if reading > 0:          # some sensors return 0 for "no echo"
                    distance = reading
            except Exception as error:
                # Something went wrong talking to the sensor. Say so on
                # screen and keep running - a GUI that dies because of
                # one bad reading is no use to anyone.
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

        # Three possible things to show, so decide the text and the
        # color first, then draw once.
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

        if too_close:
            draw_text(screen, "TOO CLOSE", (700, 140), font_big, DANGER)

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
#   1. Put your hand in front of the sensor and move it slowly. Does
#      the number match reality? If it is out by a factor of ten, your
#      sensor is probably reporting millimetres - change DISTANCE_UNITS
#      and TOO_CLOSE to match.
#   2. Set POLL_INTERVAL to 0 so it reads every frame. Watch the frame
#      rate and the steadiness of the number. Then put it back.
#   3. Set POLL_INTERVAL to 2. The number is now correct but useless.
#      Somewhere between the two is the right answer, and the right
#      answer depends on what you are using the reading for.
#   4. Change DISTANCE_SENSOR_ID to 99. You should get the error text
#      on screen and messages in the terminal, but the window keeps
#      working and the buttons still drive. That is what the try /
#      except bought you.
#   5. Why `if reading > 0` rather than trusting every value? What
#      would the display do if the sensor returned 0 whenever nothing
#      was in range?
# ---------------------------------------------------------------------