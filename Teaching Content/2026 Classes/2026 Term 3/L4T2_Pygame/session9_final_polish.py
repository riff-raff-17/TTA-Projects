"""
Session 9 -- Final Polish: Debouncing, Disabled States, Connection Status
=============================================================================

This session hardens the Session 8 robot GUI rather than adding new
widgets. Three specific problems get fixed:

    1. DEBOUNCING -- a single click shouldn't be able to fire a command
       twice in a row before the robot's had a chance to act on the
       first one. Each movement button gets a short cooldown after
       firing, during which it's visually disabled and ignores clicks.
    2. DISABLED STATES -- the cooldown above is the first real use of
       Button's `enabled` flag (added back in Session 8 but never
       actually driven by anything until now).
    3. CONNECTION STATUS -- a small indicator showing whether the last
       command to the robot succeeded or failed, so a dropped
       connection is visible at a glance instead of only showing up as
       a one-off error message in the status text.
"""

import pygame
from ugot import ugot
got = ugot.UGOT()
got.initialize("192.168.1.160")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
pygame.init()

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 380

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Session 9 -- Final Polish")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (25, 25, 35)
BUTTON_IDLE_COLOR = (60, 60, 140)
BUTTON_HOVER_COLOR = (90, 90, 200)
BUTTON_PRESSED_COLOR = (130, 130, 230)
BUTTON_DISABLED_COLOR = (40, 40, 45)
STOP_IDLE_COLOR = (140, 50, 50)
STOP_HOVER_COLOR = (190, 70, 70)
STOP_PRESSED_COLOR = (220, 100, 100)
TRACK_COLOR = (80, 80, 80)
HANDLE_COLOR = (200, 200, 60)
HANDLE_HOVER_COLOR = (230, 230, 100)
TEXT_COLOR = (255, 255, 255)
DISABLED_TEXT_COLOR = (120, 120, 120)
CONNECTED_COLOR = (60, 200, 90)
DISCONNECTED_COLOR = (200, 60, 60)

font = pygame.font.SysFont(None, 24)
status_font = pygame.font.SysFont(None, 26)


# ---------------------------------------------------------------------------
# Button and Slider -- unchanged from Sessions 6 and 7.
# ---------------------------------------------------------------------------
class Button:
    """A clickable rectangle with a label and an on_click callback.

    New this session: `disable_for(ms)` puts the button into a timed
    cooldown. While disabled, handle_event() ignores clicks entirely and
    draw() shows the disabled color -- the same `enabled` flag from
    Session 8, now actually driven by something (a timer) instead of
    sitting unused.
    """

    def __init__(self, x, y, width, height, label, on_click,
                 idle_color=BUTTON_IDLE_COLOR, hover_color=BUTTON_HOVER_COLOR,
                 pressed_color=BUTTON_PRESSED_COLOR):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.on_click = on_click
        self.is_pressed = False
        self.enabled = True
        self.idle_color = idle_color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.disabled_until = 0  # a tick timestamp; 0 means "not cooling down"

    def disable_for(self, milliseconds):
        """Start (or extend) a cooldown, using real elapsed time rather
        than counting frames -- the same clock.get_time()-vs-frame-count
        distinction Snake's movement timer relied on back in Session 3,
        applied here to a UI cooldown instead of a grid step.
        """
        self.disabled_until = pygame.time.get_ticks() + milliseconds

    def update(self):
        """Check whether a cooldown has expired and re-enable if so.
        Called once per frame, before drawing or handling events --
        debouncing needs a per-frame check, not just an event-driven one,
        since "time has passed" isn't an event pygame delivers on its own.
        """
        if self.disabled_until and pygame.time.get_ticks() >= self.disabled_until:
            self.disabled_until = 0
        self.enabled = self.disabled_until == 0

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.rect.collidepoint(mouse_pos)

        if not self.enabled:
            color = BUTTON_DISABLED_COLOR
        elif self.is_pressed and is_hovering:
            color = self.pressed_color
        elif is_hovering:
            color = self.hover_color
        else:
            color = self.idle_color

        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_color = TEXT_COLOR if self.enabled else DISABLED_TEXT_COLOR
        text_surface = font.render(self.label, True, text_color)
        surface.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if not self.enabled:
            # A disabled button doesn't track is_pressed either -- if it
            # gets disabled mid-press, it shouldn't visually "stick" in a
            # pressed state once re-enabled.
            self.is_pressed = False
            return
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                self.on_click()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False


class Slider:
    """A draggable value picker: a thin track plus a circular handle."""

    def __init__(self, x, y, width, min_value, max_value, start_value):
        self.track = pygame.Rect(x, y, width, 6)
        self.min_value = min_value
        self.max_value = max_value
        self.value = start_value
        self.dragging = False
        self.handle_radius = 10

    def _value_to_x(self):
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        return self.track.x + ratio * self.track.width

    def _x_to_value(self, x):
        ratio = (x - self.track.x) / self.track.width
        ratio = max(0, min(1, ratio))
        return self.min_value + ratio * (self.max_value - self.min_value)

    def draw(self, surface):
        pygame.draw.rect(surface, TRACK_COLOR, self.track, border_radius=3)
        handle_x = int(self._value_to_x())
        handle_pos = (handle_x, self.track.centery)

        mouse_pos = pygame.mouse.get_pos()
        distance = ((mouse_pos[0] - handle_x) ** 2 + (mouse_pos[1] - self.track.centery) ** 2) ** 0.5
        is_hovering = distance <= self.handle_radius

        color = HANDLE_HOVER_COLOR if (is_hovering or self.dragging) else HANDLE_COLOR
        pygame.draw.circle(surface, color, handle_pos, self.handle_radius)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self._value_to_x()
            distance = abs(event.pos[0] - handle_x) + abs(event.pos[1] - self.track.centery)
            if distance <= self.handle_radius + 4:
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._x_to_value(event.pos[0])


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
status = "Ready"
connected = True  # optimistic until a real call tells us otherwise
COOLDOWN_MS = 400  # how long a button stays disabled after firing


def record_call_outcome(success):
    """Update the connection indicator based on whether the last real
    `got.` call succeeded. This is deliberately driven by outcomes of
    calls you're already making, not a separate guessed health-check
    method -- see the module docstring for why.
    """
    global connected
    connected = success


def make_mover(direction):
    """Builds a callback that drives the robot in `direction`, using the
    speed slider's CURRENT value at the moment of the click. The Button
    class needed no changes at all to support this -- only a new KIND of
    function is passed in as on_click, confirming the payoff of keeping
    Button independent of what its callback actually does.
    """
    def move():
        global status

        # Debouncing: figure out which button this callback belongs to
        # and start its cooldown immediately, BEFORE the robot call --
        # so even if the call is slow, the button can't be clicked again
        # until the cooldown clears. direction_to_button is defined after
        # all the buttons exist, just below.
        direction_to_button[direction].disable_for(COOLDOWN_MS)

        try:
            speed = int(speed_slider.value)
            if direction == "forward":
                got.mecanum_move_speed(0, speed)
            elif direction == "backward":
                got.mecanum_move_speed(1, speed)
            elif direction == "left":
                got.mecanum_turn_speed(2, int(speed * 1.5))
            elif direction == "right":
                got.mecanum_turn_speed(3, int(speed * 1.5))
            status = f"Moving {direction} at {speed}"
            record_call_outcome(True)
        except Exception as e:
            # Every branch -- including the failure case -- needs a
            # deliberate, designed outcome. A dropped connection should
            # show up in the status label, not crash the whole GUI.
            status = f"Connection error: {e}"
            record_call_outcome(False)

    return move


def stop():
    global status
    stop_button.disable_for(COOLDOWN_MS)
    try:
        got.mecanum_stop()
        status = "Stopped"
        record_call_outcome(True)
    except Exception as e:
        status = f"Connection error: {e}"
        record_call_outcome(False)


# ---------------------------------------------------------------------------
# Layout: directional pad in a cross, speed slider below, status at bottom.
# Designed for an operator, not a programmer -- forward at the top, stop
# clearly separated (and recolored) from the directional buttons.
# ---------------------------------------------------------------------------
PAD_CENTER_X = WINDOW_WIDTH // 2
BUTTON_W, BUTTON_H = 110, 44

forward_button = Button(PAD_CENTER_X - BUTTON_W // 2, 30, BUTTON_W, BUTTON_H,
                         "Forward", make_mover("forward"))
left_button = Button(PAD_CENTER_X - BUTTON_W - 70, 90, BUTTON_W, BUTTON_H,
                      "Left", make_mover("left"))
right_button = Button(PAD_CENTER_X + 70, 90, BUTTON_W, BUTTON_H,
                       "Right", make_mover("right"))
backward_button = Button(PAD_CENTER_X - BUTTON_W // 2, 150, BUTTON_W, BUTTON_H,
                          "Backward", make_mover("backward"))
stop_button = Button(PAD_CENTER_X - BUTTON_W // 2, 210, BUTTON_W, BUTTON_H,
                      "STOP", stop,
                      idle_color=STOP_IDLE_COLOR, hover_color=STOP_HOVER_COLOR,
                      pressed_color=STOP_PRESSED_COLOR)

movement_buttons = [forward_button, left_button, right_button, backward_button]
all_buttons = movement_buttons + [stop_button]

# Maps each direction string to its Button instance, so make_mover's
# closures (defined above, before these buttons existed) can look up
# "which button called me" by name at click time rather than needing a
# direct reference passed in -- the dict is built once buttons actually
# exist, and Python only looks it up when move() actually runs.
direction_to_button = {
    "forward": forward_button,
    "backward": backward_button,
    "left": left_button,
    "right": right_button,
}

speed_slider = Slider(40, 290, WINDOW_WIDTH - 80, 0, 100, 50)


running = True

while running:

    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        for button in all_buttons:
            button.handle_event(event)
        speed_slider.handle_event(event)

    # --- 2. Update state ------------------------------------------------------
    # Each button's cooldown is checked once per frame here, independent
    # of whatever events did or didn't arrive this frame -- a cooldown
    # expiring isn't an event pygame delivers, so it has to be polled.
    for button in all_buttons:
        button.update()

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    title_surface = font.render("UGOT Control Panel", True, TEXT_COLOR)
    screen.blit(title_surface, (PAD_CENTER_X - 75, 5))

    # Connection indicator: a small dot plus a one-word label, placed in
    # the corner -- a glance tells you the robot's reachability without
    # reading the full status sentence below.
    indicator_color = CONNECTED_COLOR if connected else DISCONNECTED_COLOR
    pygame.draw.circle(screen, indicator_color, (WINDOW_WIDTH - 20, 18), 7)
    indicator_label = font.render("Online" if connected else "Offline", True, indicator_color)
    screen.blit(indicator_label, (WINDOW_WIDTH - 90, 8))

    for button in all_buttons:
        button.draw(screen)

    speed_label = font.render("Speed", True, TEXT_COLOR)
    screen.blit(speed_label, (40, 265))
    speed_slider.draw(screen)

    speed_value_surface = font.render(f"{speed_slider.value:.0f}", True, TEXT_COLOR)
    screen.blit(speed_value_surface, (WINDOW_WIDTH - 70, 265))

    status_surface = status_font.render(status, True, TEXT_COLOR)
    screen.blit(status_surface, (20, WINDOW_HEIGHT - 35))

    pygame.display.flip()
    clock.tick(FPS)


pygame.quit()
