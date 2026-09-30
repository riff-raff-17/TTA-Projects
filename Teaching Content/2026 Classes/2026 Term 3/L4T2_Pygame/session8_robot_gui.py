"""
Session 8 -- Building the Robot Control GUI (Capstone)
==========================================================

This is the capstone build: Button and Slider from Sessions 6-7, plus the
layout habits from Session 7, wired up into a single working application
that drives a robot. Every widget on screen maps to a real robot action --
clicking or dragging it sends a real command.
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
pygame.display.set_caption("Session 8 -- Robot Control GUI")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (25, 25, 35)
BUTTON_IDLE_COLOR = (60, 60, 140)
BUTTON_HOVER_COLOR = (90, 90, 200)
BUTTON_PRESSED_COLOR = (130, 130, 230)
STOP_IDLE_COLOR = (140, 50, 50)
STOP_HOVER_COLOR = (190, 70, 70)
STOP_PRESSED_COLOR = (220, 100, 100)
TRACK_COLOR = (80, 80, 80)
HANDLE_COLOR = (200, 200, 60)
HANDLE_HOVER_COLOR = (230, 230, 100)
TEXT_COLOR = (255, 255, 255)

font = pygame.font.SysFont(None, 24)
status_font = pygame.font.SysFont(None, 26)


# ---------------------------------------------------------------------------
# Button and Slider -- unchanged from Sessions 6 and 7.
# ---------------------------------------------------------------------------
class Button:
    """A clickable rectangle with a label and an on_click callback."""

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

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.rect.collidepoint(mouse_pos)

        if not self.enabled:
            color = (40, 40, 45)
        elif self.is_pressed and is_hovering:
            color = self.pressed_color
        elif is_hovering:
            color = self.hover_color
        else:
            color = self.idle_color

        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_color = TEXT_COLOR if self.enabled else (120, 120, 120)
        text_surface = font.render(self.label, True, text_color)
        surface.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if not self.enabled:
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


def make_mover(direction):
    """Builds a callback that drives the robot in `direction`, using the
    speed slider's CURRENT value at the moment of the click. The Button
    class needed no changes at all to support this -- only a new KIND of
    function is passed in as on_click, confirming the payoff of keeping
    Button independent of what its callback actually does.
    """
    def move():
        global status
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
        except Exception as e:
            # Every branch -- including the failure case -- needs a
            # deliberate, designed outcome. A dropped connection should
            # show up in the status label, not crash the whole GUI.
            status = f"Connection error: {e}"

    return move


def stop():
    global status
    try:
        got.mecanum_stop()
        status = "Stopped"
    except Exception as e:
        status = f"Connection error: {e}"


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
    # Nothing time-based here -- the GUI layer only translates events into
    # robot commands; it contains no robot-control logic of its own (the
    # thin-dispatcher discipline from earlier sessions, now applied to
    # hardware integration).

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    title_surface = font.render("UGOT Control Panel", True, TEXT_COLOR)
    screen.blit(title_surface, (PAD_CENTER_X - 75, 5))

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
