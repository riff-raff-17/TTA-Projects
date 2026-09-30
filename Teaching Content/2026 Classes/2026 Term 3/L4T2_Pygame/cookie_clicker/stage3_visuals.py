"""
Cookie Clicker -- Stage 3: See Your Cursors
==============================================

Everything from Stage 2 is still here, unchanged -- clicking, buying,
passive income, the whole economy. This stage adds a way to actually
SEE the cursors you've bought instead of just reading a number.

The new idea is a growing LIST of small independent objects, the same
way `buttons = [Button(...), Button(...), ...]` was a list of
independent objects back in Session 6 -- except this list grows at
runtime (one new Cursor appended every successful purchase), and each
object animates itself frame by frame instead of just sitting still
waiting for a click.

Placing many small icons around a circle without them clumping
together takes a bit of care, so Cursor placement uses two ideas:

  - Cursors fill CONCENTRIC RINGS around the cookie. `ring_capacity()`
    figures out how many cursors comfortably fit on a ring based on
    its circumference -- a wider ring holds more before the next
    cursor spills into a new, wider ring further out.

  - WITHIN a ring, slots fill in golden-ratio order (0, ~62%, ~24%,
    ...) instead of straight 0, 1, 2, ... so a half-full ring still
    looks spread around the whole circle instead of bunching up on
    one side while it fills.

Each cursor also slowly orbits (a gentle constant rotation), which is
just its stored angle increasing a little every frame -- the same
`update(dt)` pattern used for the passive income in Stage 2, applied
to a visual instead of a number.
"""

import math
import random

import pygame

pygame.init()

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 480

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Cookie Clicker -- Stage 3: See Your Cursors")

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

CURSOR_COLOR = (100, 80, 210)

big_font = pygame.font.SysFont(None, 48)
status_font = pygame.font.SysFont(None, 30)

_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = pygame.font.SysFont(None, size)
    return _font_cache[size]


def fit_text(text, max_width, start_size=22, min_size=12):
    size = start_size
    while size > min_size:
        font = get_font(size)
        surface = font.render(text, True, BUTTON_TEXT_COLOR)
        if surface.get_width() <= max_width:
            return surface
        size -= 2
    return get_font(min_size).render(text, True, BUTTON_TEXT_COLOR)


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
# Cursor -- one small arrow icon per +1/sec upgrade purchased. Purely
# decorative: cookies_per_second (from Stage 2) already does the real
# math, this just gives each upgrade something visible to point to.
# ---------------------------------------------------------------------------
ORBIT_SPEED = 0.02  # degrees per millisecond -> a slow, gentle drift
CURSOR_ARROW_SHAPE = [(14, 0), (-6, -7), (-2, 0), (-6, 7)]  # local arrowhead, tip at +x

RING_BASE_RADIUS = 95  # distance of the first ring from the cookie's center
RING_STEP = 34  # how much farther out each new ring sits
CURSOR_ARC_SPACING = 55  # target pixel gap between cursors along a ring


def ring_radius(ring):
    return RING_BASE_RADIUS + ring * RING_STEP


def ring_capacity(ring):
    circumference = 2 * math.pi * ring_radius(ring)
    return max(6, int(circumference / CURSOR_ARC_SPACING))


class Cursor:
    def __init__(self, index):
        # Walk outward ring by ring, filling each ring to capacity before
        # spilling into the next, wider one.
        ring = 0
        slot_in_ring = index
        while slot_in_ring >= ring_capacity(ring):
            slot_in_ring -= ring_capacity(ring)
            ring += 1

        capacity = ring_capacity(ring)
        self.orbit_radius = ring_radius(ring)

        # Fill slots in golden-ratio order so a partially-filled ring
        # still looks spread around the whole circle.
        step = max(1, round(capacity * 0.618))
        fill_order = (slot_in_ring * step) % capacity
        ring_offset = ring * 20  # keeps rings from lining up into spokes
        self.angle = (fill_order * (360 / capacity) + ring_offset) % 360

    def update(self, dt):
        self.angle = (self.angle + ORBIT_SPEED * dt) % 360

    def position(self, center):
        rad = math.radians(self.angle)
        return (
            center[0] + math.cos(rad) * self.orbit_radius,
            center[1] + math.sin(rad) * self.orbit_radius,
        )

    def draw(self, surface, center):
        x, y = self.position(center)

        # Point the arrow inward, toward the cookie. A point sitting at
        # orbit angle theta is offset from the center by (cos theta, sin
        # theta) * radius, so the direction BACK toward the center is
        # just that angle rotated 180 degrees.
        facing = math.radians(self.angle + 180)
        cos_a, sin_a = math.cos(facing), math.sin(facing)

        points = []
        for lx, ly in CURSOR_ARROW_SHAPE:
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            points.append((x + rx, y + ry))

        pygame.draw.polygon(surface, CURSOR_COLOR, points)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
cookies = 0.0
cookies_per_second = 0
cursor_cost = 10
cursors = []  # visual list -- one Cursor object per upgrade purchased


def click_cookie():
    global cookies
    cookies += 1


def buy_cursor():
    global cookies, cookies_per_second, cursor_cost
    if cookies >= cursor_cost:
        cookies -= cursor_cost
        cookies_per_second += 1
        cursor_cost = round(cursor_cost * 1.15)
        cursors.append(Cursor(len(cursors)))


cookie = CookieButton(260, 220, 70, click_cookie)
cursor_button = Button(170, 400, 180, 55, "", buy_cursor)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
running = True

while running:
    dt = clock.tick(FPS)

    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        cookie.handle_event(event)
        cursor_button.handle_event(event)

    # --- 2. Update state ------------------------------------------------------
    cookies += cookies_per_second * (dt / 1000)

    cursor_button.label = f"Buy Cursor +1/s (Cost: {cursor_cost})"
    cursor_button.enabled = cookies >= cursor_cost

    for c in cursors:
        c.update(dt)

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    cookie.draw(screen)
    for c in cursors:
        c.draw(screen, cookie.center)
    cursor_button.draw(screen)

    count_surface = big_font.render(f"{int(cookies)} cookies", True, TEXT_COLOR)
    screen.blit(count_surface, (20, 20))

    rate_surface = status_font.render(
        f"{cookies_per_second} per second", True, SUBTEXT_COLOR
    )
    screen.blit(rate_surface, (20, 70))

    owned_surface = status_font.render(
        f"Cursors owned: {len(cursors)}", True, SUBTEXT_COLOR
    )
    screen.blit(owned_surface, (20, 100))

    pygame.display.flip()

pygame.quit()
