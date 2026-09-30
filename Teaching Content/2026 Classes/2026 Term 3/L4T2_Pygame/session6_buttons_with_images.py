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

UPDATE: Button now optionally accepts an image_path. Pass one in and the
button draws that image instead of a flat-color rectangle -- useful once
the robot GUI wants icon buttons (arrows, a stop sign, etc.) instead of
plain colored boxes. Buttons created without image_path behave exactly
as before, so nothing about the original three buttons had to change.
"""

import os

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

    image_path is optional. Leave it out and you get the original flat
    color rectangle. Pass a path to a .png (or other pygame-supported
    image) and the button draws that image, scaled to fill the button's
    rect, instead.
    """

    def __init__(self, x, y, width, height, label, on_click, image_path=None):
        # Bundling position and size together, the same "group related
        # values into one object" idea from Session 1's Ball class --
        # here pygame.Rect does the bundling for us and throws in some
        # free helper methods (like collidepoint) besides.
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.on_click = on_click
        self.is_pressed = False

        self.image = None
        if image_path is not None:
            # convert_alpha() keeps any transparency in the source image
            # intact -- a plain convert() would flatten transparent
            # pixels to solid black, which looks wrong for icon PNGs.
            loaded_image = pygame.image.load(image_path).convert_alpha()
            # smoothscale rather than scale: buttons are usually small,
            # and a smoothed resize avoids blocky/aliased icons.
            self.image = pygame.transform.smoothscale(loaded_image, (width, height))

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.rect.collidepoint(mouse_pos)

        if self.image is not None:
            surface.blit(self.image, self.rect)

            # There's no base color to swap for hover/press feedback
            # anymore, so instead we blit a translucent white rectangle
            # on top of the image -- same idea as before (lighter when
            # hovered, lighter still when pressed), just layered rather
            # than substituted.
            if self.is_pressed and is_hovering:
                overlay_alpha = 90
            elif is_hovering:
                overlay_alpha = 50
            else:
                overlay_alpha = 0

            if overlay_alpha:
                # pygame.SRCALPHA gives this surface a per-pixel alpha
                # channel, which is what lets fill() make it translucent
                # instead of opaque white.
                overlay = pygame.Surface(self.rect.size, pygame.SRCALPHA)
                overlay.fill((255, 255, 255, overlay_alpha))
                surface.blit(overlay, self.rect)
        else:
            # Unchanged path for buttons with no image -- the original
            # flat-color rectangle from Session 6.
            if self.is_pressed and is_hovering:
                color = BUTTON_PRESSED_COLOR
            elif is_hovering:
                color = BUTTON_HOVER_COLOR
            else:
                color = BUTTON_IDLE_COLOR
            pygame.draw.rect(surface, color, self.rect, border_radius=6)

        # Drawn on top either way, so an image button can still carry a
        # text label (e.g. an icon with a caption underneath the glyph).
        # Pass label="" for an icon-only button with no text at all.
        if self.label:
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


# The first three buttons are untouched -- no image_path argument, so
# they fall back to the original flat-color look.
buttons = [
    Button(40, 40, 130, 50, "Say Hi", say_hello),
    Button(190, 40, 130, 50, "Say Bye", say_goodbye),
    Button(340, 40, 120, 50, "Count", count_click),
]

# A fourth button demonstrates the new image_path argument. Point this
# at your own PNG -- something like an icon in the same folder as this
# script. The os.path.exists check just keeps this demo runnable even
# before you've supplied a real icon file; you don't need it once you
# have a real path.
icon_path = "stop_icon.png"
if os.path.exists(icon_path):
    buttons.append(Button(40, 110, 60, 60, "", lambda: None, image_path=icon_path))


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
