"""
Session 6 -- From Games to GUIs: Buttons and Clickable Areas
================================================================

With Snake complete, this session pivots toward the final goal: a
graphical control panel for the UGOT robot. The fundamental unit of any
GUI is the BUTTON -- and structurally, a button is something we've
already built: a rectangle with a position, a way to detect whether the
mouse is interacting with it, and a response when it's clicked.

The pivot from game to GUI is a change in what the rectangles REPRESENT,
not a new skill. The same Rect + collidepoint() mechanics that checked
Snake's walls and food now check "is the user trying to click this".
"""

import pygame


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
pygame.init()

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 300

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Session 6 -- Buttons")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (20, 20, 30)
BUTTON_IDLE_COLOR = (60, 60, 140)
BUTTON_HOVER_COLOR = (90, 90, 200)
BUTTON_PRESSED_COLOR = (130, 130, 230)
TEXT_COLOR = (255, 255, 255)

font = pygame.font.SysFont(None, 24)
status_font = pygame.font.SysFont(None, 28)


# ---------------------------------------------------------------------------
# A reusable Button class
# ---------------------------------------------------------------------------
class Button:
    """A clickable rectangle with a label and an on_click callback.

    This class knows nothing about WHAT clicking it should do -- it just
    draws itself, reports hover/press visually, and calls whatever
    function was handed to it at creation time. That separation of
    appearance from behaviour is what lets the exact same class get
    reused, unmodified, for every button the final robot GUI will need.
    """

    def __init__(self, x, y, width, height, label, on_click):
        # Bundling position and size together, the same "group related
        # values into one object" idea from Session 1's Ball class --
        # here pygame.Rect does the bundling for us and throws in some
        # free helper methods (like collidepoint) besides.
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

        text_surface = font.render(self.label, True, TEXT_COLOR)
        surface.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
                self.is_pressed = True
                self.on_click()

        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False


# ---------------------------------------------------------------------------
# Application state and button callbacks
# ---------------------------------------------------------------------------
status = "Idle"
click_count = 0


def say_hello():
    global status
    status = "Hello!"


def say_goodbye():
    global status
    status = "Goodbye!"


def count_click():
    global status, click_count
    click_count += 1
    status = f"Clicked {click_count} time(s)"


# Three independent Button instances, each with its own callback -- the
# same class, reused, with different arguments. Creating a third button
# costs one new line, not a copy-pasted block.
buttons = [
    Button(40, 40, 130, 50, "Say Hi", say_hello),
    Button(190, 40, 130, 50, "Say Bye", say_goodbye),
    Button(340, 40, 120, 50, "Count", count_click),
]


running = True

while running:
    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        for button in buttons:
            button.handle_event(event)

    # --- 2. Update state ------------------------------------------------------
    # Nothing to update here besides what the button callbacks already
    # changed directly (status, click_count) -- there's no game clock or
    # movement timer in a GUI like this one.

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    for button in buttons:
        button.draw(screen)

    status_surface = status_font.render(f"Status: {status}", True, TEXT_COLOR)
    screen.blit(status_surface, (40, 140))

    pygame.display.flip()
    clock.tick(FPS)


pygame.quit()
