"""
Cookie Clicker -- Stage 2: Buy Cursors (the economy)
=======================================================

Everything from Stage 1 is still here, unchanged. This stage adds two
new ideas on top of it:

  1. A rectangular Button -- literally the Button class from Session
     6, copy-pasted in with no changes. It sits next to CookieButton
     to prove the point from that session: the SAME button class
     works for a totally different purpose (buying an upgrade instead
     of, say, moving a snake) because it never knew what clicking it
     should do in the first place -- that's the `on_click` callback's
     job.

  2. Passive income over time. Buying a cursor makes cookies appear
     on their own, once per second, without any click. That's handled
     with `dt` -- "delta time", the milliseconds since the last frame
     -- so the earn rate stays the same regardless of how fast the
     computer runs the loop. `cookies` becomes a float because it now
     grows in small fractional steps every frame instead of jumping by
     a whole cookie per click.

One extra detail: the upgrade button's label changes every frame to
show the current price, and that price grows long ("Buy Cursor +1/s
(Cost: 1000)"). Sizing the font once and hoping the text always fits
is fragile, so `fit_text()` shrinks the label to whatever width is
actually available.
"""

import pygame

pygame.init()

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 480

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Cookie Clicker -- Stage 2: Buy Cursors")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (255, 244, 214)
TEXT_COLOR = (70, 45, 20)
SUBTEXT_COLOR = (140, 100, 60)
BUTTON_TEXT_COLOR = (255, 255, 255)

COOKIE_COLOR = (216, 148, 60)
COOKIE_HOVER_COLOR = (232, 168, 82)
COOKIE_PRESSED_COLOR = (245, 188, 104)
CHIP_COLOR = (110, 64, 24)

BUTTON_IDLE_COLOR = (120, 100, 235)
BUTTON_HOVER_COLOR = (145, 128, 250)
BUTTON_PRESSED_COLOR = (175, 160, 255)
BUTTON_DISABLED_COLOR = (205, 200, 215)

big_font = pygame.font.SysFont(None, 48)
status_font = pygame.font.SysFont(None, 30)

# Cache of button fonts by size, so fit_text() isn't creating a new Font
# object every single frame.
_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont(None, size)
    return _font_cache[size]


def fit_text(text, max_width, start_size=22, min_size=12):
    """Render text at the largest size (down to min_size) that still fits
    within max_width pixels, so a long label never spills outside its
    button.
    """
    size = start_size
    while size > min_size:
        font = get_font(size)
        surface = font.render(text, True, BUTTON_TEXT_COLOR)
        if surface.get_width() <= max_width:
            return surface
        size -= 2
    return get_font(min_size).render(text, True, BUTTON_TEXT_COLOR)


# ---------------------------------------------------------------------------
# Button -- straight from Session 6, plus fit_text() for the label.
# ---------------------------------------------------------------------------
class Button:
    def __init__(self, x, y, width, height, label, on_click):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.on_click = on_click
        self.is_pressed = False
        self.enabled = True

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.rect.collidepoint(mouse_pos)

        if not self.enabled:
            color = BUTTON_DISABLED_COLOR
        elif self.is_pressed and is_hovering:
            color = BUTTON_PRESSED_COLOR
        elif is_hovering:
            color = BUTTON_HOVER_COLOR
        else:
            color = BUTTON_IDLE_COLOR

        pygame.draw.rect(surface, color, self.rect, border_radius=6)

        text_surface = fit_text(self.label, self.rect.width - 20)
        surface.blit(text_surface, text_surface.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if not self.enabled:
            return
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.is_pressed = True
            self.on_click()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False


# ---------------------------------------------------------------------------
# CookieButton -- unchanged from Stage 1.
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

        radius = self.radius - 4 if (self.is_pressed and is_hovering) else self.radius
        pygame.draw.circle(surface, color, self.center, radius)

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
cookies = 0.0  # float now: passive income adds tiny fractional amounts each frame
cookies_per_second = 0
cursor_cost = 10


def click_cookie():
    global cookies
    cookies += 1


def buy_cursor():
    global cookies, cookies_per_second, cursor_cost
    if cookies >= cursor_cost:
        cookies -= cursor_cost
        cookies_per_second += 1
        cursor_cost = round(cursor_cost * 1.15)


cookie = CookieButton(260, 220, 70, click_cookie)
cursor_button = Button(170, 400, 180, 55, "", buy_cursor)


# Main loop
running = True

while running:
    dt = clock.tick(FPS)  # milliseconds since the last frame

    # --- 1. Handle events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        cookie.handle_event(event)
        cursor_button.handle_event(event)

    # --- 2. Update state ---
    cookies += cookies_per_second * (dt / 1000)

    cursor_button.label = f"Buy Cursor +1/s (Cost: {cursor_cost})"
    cursor_button.enabled = cookies >= cursor_cost

    # --- 3. Draw the frame ---
    screen.fill(BACKGROUND_COLOR)

    cookie.draw(screen)
    cursor_button.draw(screen)

    count_surface = big_font.render(f"{int(cookies)} cookies", True, TEXT_COLOR)
    screen.blit(count_surface, (20, 20))

    rate_surface = status_font.render(
        f"{cookies_per_second} per second", True, SUBTEXT_COLOR
    )
    screen.blit(rate_surface, (20, 70))

    pygame.display.flip()

pygame.quit()
