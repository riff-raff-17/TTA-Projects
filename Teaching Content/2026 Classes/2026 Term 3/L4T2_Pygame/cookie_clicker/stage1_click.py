"""
Cookie Clicker -- Stage 1: Click the Cookie
=============================================

The whole game grows from one idea already familiar from Session 6: a
button is a shape you can hover over, press, and get a callback from.
Session 6's Button was a rectangle. Here the "button" is the cookie
itself, and it's a circle -- so instead of Rect.collidepoint(), which
checks "is this point inside this rectangle", CookieButton uses the
distance formula: "is this point within `radius` pixels of the
center?" (dx^2 + dy^2 <= radius^2). Everything else about it (hover
color, pressed color, calling a function on click) works the same way
a rectangular button does.

This stage does ONE thing: click the cookie, watch the count go up.
Later stages add an economy (buying auto-clickers), then a way to
actually SEE those auto-clickers on screen.
"""

import pygame

pygame.init()

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 480

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Cookie Clicker -- Stage 1: Click the Cookie")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (255, 244, 214)
TEXT_COLOR = (70, 45, 20)

COOKIE_COLOR = (216, 148, 60)
COOKIE_HOVER_COLOR = (232, 168, 82)
COOKIE_PRESSED_COLOR = (245, 188, 104)
CHIP_COLOR = (110, 64, 24)

big_font = pygame.font.SysFont(None, 48)


# ---------------------------------------------------------------------------
# CookieButton -- the Session 6 button pattern, adapted for a circular
# hit area instead of a rectangular one.
# ---------------------------------------------------------------------------
class CookieButton:
    def __init__(self, center_x, center_y, radius, on_click):
        self.center = (center_x, center_y)
        self.radius = radius
        self.on_click = on_click
        self.is_pressed = False

    def collidepoint(self, pos):
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.collidepoint(mouse_pos)

        if self.is_pressed and is_hovering:
            color = COOKIE_PRESSED_COLOR
        elif is_hovering:
            color = COOKIE_HOVER_COLOR
        else:
            color = COOKIE_COLOR

        # Shrink slightly while pressed so the click has some visuals
        radius = self.radius - 4 if (self.is_pressed and is_hovering) else self.radius
        pygame.draw.circle(surface, color, self.center, radius)

        # A handful of fixed "chip" dots so it looks like a cookie
        chip_offsets = [(-25, -15), (10, -25), (25, 10), (-15, 20), (0, 0), (-30, 15)]
        for ox, oy in chip_offsets:
            pygame.draw.circle(
                surface, CHIP_COLOR, (self.center[0] + ox, self.center[1] + oy), 5
            )

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.collidepoint(event.pos):
            self.is_pressed = True
            self.on_click()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
cookies = 0


def click_cookie():
    global cookies
    cookies += 1


cookie = CookieButton(260, 220, 70, click_cookie)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
running = True

while running:
    clock.tick(FPS)

    # --- 1. Handle events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        cookie.handle_event(event)

    # --- 2. Update state ------------------------------------------------------
    # Nothing to update yet -- clicking is the only thing that changes
    # `cookies`, and that already happened inside click_cookie().

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    cookie.draw(screen)

    count_surface = big_font.render(f"{cookies} cookies", True, TEXT_COLOR)
    screen.blit(count_surface, (20, 20))

    pygame.display.flip()

pygame.quit()
