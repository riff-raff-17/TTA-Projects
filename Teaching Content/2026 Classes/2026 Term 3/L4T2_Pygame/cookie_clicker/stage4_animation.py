"""
Cookie Clicker -- Stage 4: Animation & Feedback (final)
==========================================================

Everything from Stage 3 is still here -- clicking, the economy, and
cursors placed evenly in rings around the cookie. This stage adds the
last bit of "juice": each cursor now visibly DOES something once a
second instead of just sitting there rotating.

Two new pieces, both reusing patterns from earlier stages:

  1. A per-cursor timer, same shape as the dt-based passive income
     from Stage 2, that flips a `fired` flag once every 1000ms (since
     each cursor contributes exactly +1/sec). When it fires, the
     cursor flashes gold and pops slightly larger for a moment -- the
     `flash` value decays a little every frame, the same fade-over-time
     idea used for the orbit rotation, just counting down instead of
     wrapping around.

  2. Floating "+1" particles, spawned wherever a cursor just fired.
     `floating_texts` is a plain list of dicts -- no new class needed,
     since a particle doesn't do anything but exist, drift upward, and
     fade out over 700ms using `surface.set_alpha()`.
"""

import math
import random

import pygame

pygame.init()

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 480

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Cookie Clicker")

clock = pygame.time.Clock()
FPS = 60

# ---------------------------------------------------------------------------
# Colors -- bright, cheerful palette. Two text colors: TEXT_COLOR for dark
# text on the light background, BUTTON_TEXT_COLOR for white text on the
# colored buttons.
# ---------------------------------------------------------------------------
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
CURSOR_FLASH_COLOR = (255, 196, 60)

big_font = pygame.font.SysFont(None, 48)
status_font = pygame.font.SysFont(None, 30)
particle_font = pygame.font.SysFont(None, 22)

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
# Button class -- same as Session 6, plus fit_text() for labels.
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
# CookieButton class -- same pattern as Button, but circular.
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
            pygame.draw.circle(surface, CHIP_COLOR, (self.center[0] + ox, self.center[1] + oy), 5)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.collidepoint(event.pos):
            self.is_pressed = True
            self.on_click()
        elif event.type == pygame.MOUSEBUTTONUP:
            self.is_pressed = False


# ---------------------------------------------------------------------------
# Cursor class -- one small arrow icon per +1/sec upgrade purchased.
# Purely decorative: cookies_per_second already does the real math, this
# just gives each upgrade something visible to point to on screen.
# ---------------------------------------------------------------------------
ORBIT_SPEED = 0.02          # degrees per millisecond -> a slow, gentle drift
CURSOR_ARROW_SHAPE = [(14, 0), (-6, -7), (-2, 0), (-6, 7)]  # local arrowhead, tip at +x

RING_BASE_RADIUS = 95   # distance of the first ring from the cookie's center
RING_STEP = 34          # how much farther out each new ring sits
CURSOR_ARC_SPACING = 55  # target pixel gap between cursors along a ring


def ring_radius(ring):
    return RING_BASE_RADIUS + ring * RING_STEP


def ring_capacity(ring):
    # A wider ring has more circumference, so it can comfortably fit
    # more cursors before they start crowding each other -- capacity
    # scales with the ring's circumference divided by the spacing we
    # want between icons.
    circumference = 2 * math.pi * ring_radius(ring)
    return max(6, int(circumference / CURSOR_ARC_SPACING))


class Cursor:
    def __init__(self, index):
        # Walk outward ring by ring, filling each ring to capacity before
        # spilling into the next, wider one -- so cursors form clean
        # concentric circles instead of one continuous spiral.
        ring = 0
        slot_in_ring = index
        while slot_in_ring >= ring_capacity(ring):
            slot_in_ring -= ring_capacity(ring)
            ring += 1

        capacity = ring_capacity(ring)
        self.orbit_radius = ring_radius(ring)

        # Within a ring, fill slots in golden-ratio order (0, ~0.618,
        # ~0.236, ...) rather than straight 0, 1, 2, ... so a
        # half-filled ring still looks spread around the whole circle
        # instead of bunching up on one side while it fills.
        step = max(1, round(capacity * 0.618))
        fill_order = (slot_in_ring * step) % capacity
        ring_offset = ring * 20  # keeps rings from lining up into spokes
        self.angle = (fill_order * (360 / capacity) + ring_offset) % 360
        # Stagger each cursor's 1-second "click" cycle so they don't all
        # flash in unison.
        self.cycle_ms = random.uniform(0, 1000)
        self.flash = 0.0  # 0..1 -- spikes to 1 on a "click", fades after

    def update(self, dt):
        self.angle = (self.angle + ORBIT_SPEED * dt) % 360
        self.cycle_ms += dt
        fired = False
        if self.cycle_ms >= 1000:
            self.cycle_ms -= 1000
            self.flash = 1.0
            fired = True
        self.flash = max(0.0, self.flash - dt / 300)
        return fired

    def position(self, center):
        rad = math.radians(self.angle)
        return (center[0] + math.cos(rad) * self.orbit_radius,
                center[1] + math.sin(rad) * self.orbit_radius)

    def draw(self, surface, center):
        x, y = self.position(center)

        # Point the arrow inward, toward the cookie. A point sitting at
        # orbit angle theta is offset from the center by (cos theta, sin
        # theta) * radius, so the direction BACK toward the center is
        # just that angle rotated 180 degrees.
        facing = math.radians(self.angle + 180)
        cos_a, sin_a = math.cos(facing), math.sin(facing)

        scale = 1.0 + 0.5 * self.flash
        color = tuple(
            int(CURSOR_COLOR[i] + (CURSOR_FLASH_COLOR[i] - CURSOR_COLOR[i]) * self.flash)
            for i in range(3)
        )

        points = []
        for lx, ly in CURSOR_ARROW_SHAPE:
            lx, ly = lx * scale, ly * scale
            rx = lx * cos_a - ly * sin_a
            ry = lx * sin_a + ly * cos_a
            points.append((x + rx, y + ry))

        pygame.draw.polygon(surface, color, points)


# ---------------------------------------------------------------------------
# Floating "+1" particles -- spawned whenever a cursor fires, drift
# upward and fade out. Each one is just a dict in a list; nothing here
# needs its own class since there's no behaviour beyond "age, then die".
# ---------------------------------------------------------------------------
FLOATING_TEXT_LIFETIME = 700  # milliseconds
floating_texts = []


def spawn_floating_text(pos):
    floating_texts.append({"x": pos[0], "y": pos[1], "age": 0.0})


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
    dt = clock.tick(FPS)  # milliseconds since the last frame

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
        if c.update(dt):
            spawn_floating_text(c.position(cookie.center))

    # Age out floating texts, removing any that have finished their life.
    for ft in floating_texts[:]:
        ft["age"] += dt
        ft["y"] -= dt * 0.04  # drift upward
        if ft["age"] >= FLOATING_TEXT_LIFETIME:
            floating_texts.remove(ft)

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    cookie.draw(screen)
    for c in cursors:
        c.draw(screen, cookie.center)

    for ft in floating_texts:
        remaining = 1 - (ft["age"] / FLOATING_TEXT_LIFETIME)
        text_surface = particle_font.render("+1", True, CURSOR_FLASH_COLOR)
        text_surface.set_alpha(int(255 * remaining))
        screen.blit(text_surface, (ft["x"], ft["y"]))

    cursor_button.draw(screen)

    count_surface = big_font.render(f"{int(cookies)} cookies", True, TEXT_COLOR)
    screen.blit(count_surface, (20, 20))

    rate_surface = status_font.render(f"{cookies_per_second} per second", True, SUBTEXT_COLOR)
    screen.blit(rate_surface, (20, 70))

    owned_surface = status_font.render(f"Cursors owned: {len(cursors)}", True, SUBTEXT_COLOR)
    screen.blit(owned_surface, (20, 100))

    pygame.display.flip()

pygame.quit()
