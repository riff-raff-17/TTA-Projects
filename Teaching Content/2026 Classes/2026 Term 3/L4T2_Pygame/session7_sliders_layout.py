"""
Session 7 -- Sliders, Status Text, and Layout
=================================================

A robot control panel needs more than buttons: it needs a way to set a
CONTINUOUS value (like speed), and a clear way to display the robot's
current status. This session adds a Slider widget alongside Session 6's
Button, and shifts attention to LAYOUT -- arranging several widgets on
screen so it reads clearly rather than looking like a pile of rectangles.
"""

import pygame


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
pygame.init()

WINDOW_WIDTH = 560
WINDOW_HEIGHT = 360

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Session 7 -- Sliders, Status Text, and Layout")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (20, 20, 30)
BUTTON_IDLE_COLOR = (60, 60, 140)
BUTTON_HOVER_COLOR = (90, 90, 200)
BUTTON_PRESSED_COLOR = (130, 130, 230)
TRACK_COLOR = (80, 80, 80)
HANDLE_COLOR = (200, 200, 60)
HANDLE_HOVER_COLOR = (230, 230, 100)
TEXT_COLOR = (255, 255, 255)

label_font = pygame.font.SysFont(None, 24)
status_font = pygame.font.SysFont(None, 26)


# ---------------------------------------------------------------------------
# Arranging widgets with simple layout variables
# ---------------------------------------------------------------------------
# Rather than hard-coding every pixel value, each widget's position is an
# offset from a shared margin and row height -- a lightweight, early
# version of the layout systems real GUI frameworks provide.
MARGIN = 30
ROW_HEIGHT = 70


# ---------------------------------------------------------------------------
# The Button class -- unchanged from Session 6.
# ---------------------------------------------------------------------------
class Button:
    """A clickable rectangle with a label and an on_click callback."""

    def __init__(self, x, y, width, height, label, on_click):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.on_click = on_click
        self.is_pressed = False

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.rect.collidepoint(mouse_pos)

        if self.is_pressed and is_hovering:
            color = BUTTON_PRESSED_COLOR
        elif is_hovering:
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_IDLE_COLOR

        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        text_surface = label_font.render(self.label, True, TEXT_COLOR)
        surface.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True
                self.on_click()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False


# ---------------------------------------------------------------------------
# The Slider class -- new this session.
# ---------------------------------------------------------------------------
class Slider:
    """A draggable value picker: a thin track plus a circular handle.

    Like Button, a Slider knows nothing about what its value will be
    used for -- it just tracks a number in [min_value, max_value] and
    lets the user drag it. That independence is exactly what let the
    Button class plug into a totally different app (the cookie clicker)
    without changes, and it's what will let this Slider plug into the
    robot's speed control later without changes either.
    """

    def __init__(self, x, y, width, min_value, max_value, start_value):
        self.track = pygame.Rect(x, y, width, 6)
        self.min_value = min_value
        self.max_value = max_value
        self.value = start_value
        self.dragging = False
        self.handle_radius = 10

    def _value_to_x(self):
        """Map the current value to a pixel x-position on the track."""
        ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        return self.track.x + ratio * self.track.width

    def _x_to_value(self, x):
        """Map a pixel x-position back to a value in [min_value, max_value].

        Same idea as Session 1's clamp(): a pixel position is just a
        number whose meaning comes from a deliberate mapping. Here the
        ratio is clamped to [0, 1] BEFORE converting to a value, so the
        handle's value can never leave its valid range even if the mouse
        is dragged past the track's edges.
        """
        ratio = (x - self.track.x) / self.track.width
        ratio = max(0, min(1, ratio))  # clamp, as in Session 1
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
        # The three-event drag sequence: press down on the handle (start
        # dragging), move while the dragging flag is set (update the
        # value), release (stop dragging). This is a new combination of
        # Session 6's click-detection with a SUSTAINED, frame-by-frame
        # update step -- a click is instantaneous, a drag isn't.
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x = self._value_to_x()
            distance = abs(event.pos[0] - handle_x) + abs(event.pos[1] - self.track.centery)
            if distance <= self.handle_radius + 4:  # a little forgiving for an easy grab
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self._x_to_value(event.pos[0])


# ---------------------------------------------------------------------------
# Application state and layout
# ---------------------------------------------------------------------------
status = "Idle"


def say_hello():
    global status
    status = "Hello!"


def reset_speed():
    global status
    speed_slider.value = 50
    status = "Speed reset to 50"


# Row 1: a button on the left, a status label conceptually to its right
# (drawn separately below). Row 2: the slider, full width-ish, with its
# own label above it. Consistent left margin keeps everything aligned.
hello_button = Button(MARGIN, MARGIN, 140, 50, "Say Hi", say_hello)
reset_button = Button(MARGIN + 160, MARGIN, 140, 50, "Reset Speed", reset_speed)

slider_y = MARGIN + ROW_HEIGHT + 30
speed_slider = Slider(MARGIN, slider_y, WINDOW_WIDTH - MARGIN * 2, 0, 100, 50)


running = True

while running:

    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        hello_button.handle_event(event)
        reset_button.handle_event(event)
        speed_slider.handle_event(event)

    # --- 2. Update state ------------------------------------------------------
    # Nothing time-based here -- widgets update themselves directly inside
    # handle_event() as drag events arrive.

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    hello_button.draw(screen)
    reset_button.draw(screen)

    slider_label = label_font.render("Speed", True, TEXT_COLOR)
    screen.blit(slider_label, (MARGIN, slider_y - 28))

    speed_slider.draw(screen)

    # Rendering live status text -- extends Snake's score-display
    # technique (Session 4) to show arbitrary live state, not just a
    # score: here, the slider's current value AND a one-off status
    # message both get shown the same way.
    value_text = f"Value: {speed_slider.value:.0f}"
    value_surface = label_font.render(value_text, True, TEXT_COLOR)
    screen.blit(value_surface, (MARGIN, slider_y + 25))

    status_surface = status_font.render(f"Status: {status}", True, TEXT_COLOR)
    screen.blit(status_surface, (MARGIN, slider_y + 70))

    pygame.display.flip()
    clock.tick(FPS)


pygame.quit()
