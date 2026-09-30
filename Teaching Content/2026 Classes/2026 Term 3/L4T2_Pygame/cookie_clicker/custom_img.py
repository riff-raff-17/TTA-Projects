import math
import random

import pygame

pygame.init()
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"Could not initialize audio ({e}). Sound effects will be disabled.")

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 480

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Cookie Clicker")

clock = pygame.time.Clock()
FPS = 60

# ---------------------------------------------------------------------------
# Customization -- point these at your own art/audio/font files to reskin
# the game. Leave any of them as None (the default) to keep the flat-color
# cookie, background, cursor icon, system font, system mouse pointer, and
# silent (sound-free) feel shown in this file. Whatever image or font you
# provide is scaled/cropped automatically to fit its slot, so nothing
# needs to be pre-sized -- drop in any photo, logo, .ttf, or .wav/.ogg
# as-is.
# ---------------------------------------------------------------------------
BACKGROUND_IMAGE_PATH = "cookie_clicker/bg.jpg"  # e.g. "assets/background.png"
COOKIE_IMAGE_PATH = "cookie_clicker/cookie.jpg"  # e.g. "assets/cookie.png"
CURSOR_ICON_IMAGE_PATH = (
    "cookie_clicker/mouse.jpg"  # e.g. "assets/cursor_icon.png" -- the orbiting +1/s icons
)
MOUSE_CURSOR_IMAGE_PATH = (
    "cookie_clicker/mouse.jpg"  # e.g. "assets/mouse_cursor.png" -- replaces the OS pointer
)
FONT_PATH = None  # e.g. "assets/MyFont.ttf"
CLICK_SOUND_PATH = None  # e.g. "assets/click.wav" -- played when you click the cookie
PURCHASE_SOUND_PATH = None  # e.g. "assets/purchase.wav" -- played when you buy a cursor

# ---------------------------------------------------------------------------
# Game balance -- tune the economy here without hunting through the code
# below.
# ---------------------------------------------------------------------------
STARTING_COOKIES = 0.0
COOKIES_PER_CLICK = 1
STARTING_CURSOR_COST = 10
CURSOR_COST_GROWTH = 1.15  # each cursor costs 15% more than the last
COOKIES_PER_SECOND_PER_CURSOR = 1  # how much +1/s a single cursor upgrade adds

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

# Cache of fonts by size, so fit_text() (and the headline fonts below)
# aren't creating a new Font object every single frame. If FONT_PATH is
# set (and loads successfully), every size below is loaded from that file
# instead of the system font.
_font_cache = {}


def _resolve_font_path():
    if not FONT_PATH:
        return None
    try:
        pygame.font.Font(FONT_PATH, 10)  # smoke-test that the file loads
        return FONT_PATH
    except (pygame.error, FileNotFoundError) as e:
        print(
            f"Could not load font '{FONT_PATH}' ({e}). Falling back to the default font."
        )
        return None


_resolved_font_path = _resolve_font_path()


def get_font(size):
    if size not in _font_cache:
        if _resolved_font_path:
            _font_cache[size] = pygame.font.Font(_resolved_font_path, size)
        else:
            _font_cache[size] = pygame.font.SysFont(None, size)
    return _font_cache[size]


big_font = get_font(48)
status_font = get_font(30)
particle_font = get_font(22)


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


def format_number(value):
    """Format a number for on-screen display, abbreviating large values
    (1200 -> "1.2K", 3400000 -> "3.4M") the way most idle games do, so the
    cookie count doesn't outgrow its label as the game progresses.
    """
    value = int(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if value >= threshold:
            return f"{value / threshold:.1f}{suffix}"
    return str(value)


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
# CookieButton class -- same pattern as Button, but circular. Optionally
# skinned with an image (see load_circular_image() below) instead of the
# flat-color circle.
# ---------------------------------------------------------------------------
class CookieButton:
    def __init__(self, center_x, center_y, radius, on_click, image=None):
        self.center = (center_x, center_y)
        self.radius = radius
        self.on_click = on_click
        self.is_pressed = False

        # `image`, if given, is a pre-scaled circular surface from
        # load_circular_image() -- already sized to fit this exact
        # radius. A second, slightly smaller copy is kept ready for the
        # "pressed" look, matching the radius - 4 shrink the flat-color
        # cookie uses below.
        self.image = image
        self.pressed_image = None
        if self.image is not None:
            pressed_diameter = (radius - 4) * 2
            self.pressed_image = pygame.transform.smoothscale(
                self.image, (pressed_diameter, pressed_diameter)
            )

    def collidepoint(self, pos):
        dx = pos[0] - self.center[0]
        dy = pos[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.collidepoint(mouse_pos)
        pressed_now = self.is_pressed and is_hovering

        if self.image is not None:
            frame = self.pressed_image if pressed_now else self.image

            # A photo can't shift color the way the flat cookie does, so
            # hover/press are shown with a soft white overlay instead.
            overlay_alpha = 90 if pressed_now else (45 if is_hovering else 0)
            if overlay_alpha:
                diameter = frame.get_width()
                frame = frame.copy()
                overlay = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
                pygame.draw.circle(
                    overlay,
                    (255, 255, 255, overlay_alpha),
                    (diameter // 2, diameter // 2),
                    diameter // 2,
                )
                frame.blit(overlay, (0, 0))

            surface.blit(frame, frame.get_rect(center=self.center))
        else:
            if pressed_now:
                color = COOKIE_PRESSED_COLOR
            elif is_hovering:
                color = COOKIE_HOVER_COLOR
            else:
                color = COOKIE_COLOR

            radius = self.radius - 4 if pressed_now else self.radius
            pygame.draw.circle(surface, color, self.center, radius)

            chip_offsets = [
                (-25, -15),
                (10, -25),
                (25, 10),
                (-15, 20),
                (0, 0),
                (-30, 15),
            ]
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
# Image and sound loading helpers for the paths in the Customization
# section above. The image loaders scale (and, if needed, crop) whatever
# you provide so it fits its slot exactly, the same way a photo editor's
# "fill" or "fit" option would -- no manual resizing required. Every
# loader here returns None (falling back to the default look, or to
# silence for sounds) if no path was given or the file couldn't be
# loaded.
# ---------------------------------------------------------------------------
def _crop_to_square(image):
    """Return the largest centered square crop of `image` -- shared by
    the circular and square image loaders below so a non-square photo
    never gets stretched out of proportion.
    """
    width, height = image.get_size()
    side = min(width, height)
    crop_rect = pygame.Rect((width - side) // 2, (height - side) // 2, side, side)
    return image.subsurface(crop_rect).copy()


def load_cover_image(path, target_width, target_height):
    """Load an image and scale+crop it to completely fill a
    target_width x target_height box without distorting it -- like CSS's
    'background-size: cover'.
    """
    if not path:
        return None
    try:
        image = pygame.image.load(path).convert()
    except (pygame.error, FileNotFoundError) as e:
        print(f"Could not load image '{path}' ({e}). Falling back to the default look.")
        return None

    img_width, img_height = image.get_size()
    scale = max(target_width / img_width, target_height / img_height)
    # math.ceil (rather than round) guarantees the scaled image is never
    # a hair smaller than the target box, which would make the crop below
    # fail on floating-point rounding edge cases.
    scaled_size = (
        max(target_width, math.ceil(img_width * scale)),
        max(target_height, math.ceil(img_height * scale)),
    )
    scaled = pygame.transform.smoothscale(image, scaled_size)

    crop_x = (scaled_size[0] - target_width) // 2
    crop_y = (scaled_size[1] - target_height) // 2
    return scaled.subsurface(
        pygame.Rect(crop_x, crop_y, target_width, target_height)
    ).copy()


def load_circular_image(path, diameter):
    """Load an image, center-crop it to a square, scale it to
    diameter x diameter, then mask it to a circle so it drops cleanly
    into a CookieButton of that size.
    """
    if not path:
        return None
    try:
        image = pygame.image.load(path).convert_alpha()
    except (pygame.error, FileNotFoundError) as e:
        print(f"Could not load image '{path}' ({e}). Falling back to the default look.")
        return None

    square = _crop_to_square(image)
    scaled = pygame.transform.smoothscale(square, (diameter, diameter))

    # Punch out everything outside the circle by zeroing its alpha there.
    # BLEND_RGBA_MIN keeps each channel's *smaller* value, and the mask's
    # RGB channels are all 255 (a no-op against any color), so this only
    # ever affects alpha -- pixels outside the circle become transparent.
    mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(
        mask, (255, 255, 255, 255), (diameter // 2, diameter // 2), diameter // 2
    )
    circular = scaled.convert_alpha()
    circular.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return circular


def load_square_image(path, size):
    """Load an image, center-crop it to a square, and scale it to
    size x size. Unlike load_circular_image(), no circular mask is
    applied, so an icon with its own transparent background (a hand, a
    paw print, an arrow, ...) keeps its natural silhouette. Used for the
    cursor-icon and mouse-cursor images, which aren't necessarily round.
    """
    if not path:
        return None
    try:
        image = pygame.image.load(path).convert_alpha()
    except (pygame.error, FileNotFoundError) as e:
        print(f"Could not load image '{path}' ({e}). Falling back to the default look.")
        return None

    square = _crop_to_square(image)
    return pygame.transform.smoothscale(square, (size, size))


def load_sound(path):
    """Load a sound effect. Returns None (silently skipped whenever it
    would play) if no path was given, the audio device failed to
    initialize, or the file couldn't be loaded.
    """
    if not path or not pygame.mixer.get_init():
        return None
    try:
        return pygame.mixer.Sound(path)
    except (pygame.error, FileNotFoundError) as e:
        print(
            f"Could not load sound '{path}' ({e}). That sound effect will be disabled."
        )
        return None


def play_sound(sound):
    if sound is not None:
        sound.play()


# ---------------------------------------------------------------------------
# Cursor class -- one small arrow icon per +1/sec upgrade purchased.
# Purely decorative: cookies_per_second already does the real math, this
# just gives each upgrade something visible to point to on screen.
# ---------------------------------------------------------------------------
ORBIT_SPEED = 0.02  # degrees per millisecond -> a slow, gentle drift
CURSOR_ARROW_SHAPE = [(14, 0), (-6, -7), (-2, 0), (-6, 7)]  # local arrowhead, tip at +x
CURSOR_ICON_SIZE = 26  # bounding box (px) a custom CURSOR_ICON_IMAGE_PATH is scaled to

RING_BASE_RADIUS = 95  # distance of the first ring from the cookie's center
RING_STEP = 34  # how much farther out each new ring sits
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
    def __init__(self, index, icon_image=None):
        # `icon_image`, if given, is a pre-scaled square surface from
        # load_square_image() that replaces the flat arrow below.
        self.icon_image = icon_image

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
        return (
            center[0] + math.cos(rad) * self.orbit_radius,
            center[1] + math.sin(rad) * self.orbit_radius,
        )

    def draw(self, surface, center):
        x, y = self.position(center)
        # Spikes to 1.5x size on a "click", the same pop whether it's the
        # flat arrow or a custom icon.
        scale = 1.0 + 0.5 * self.flash

        if self.icon_image is not None:
            # A glow behind the icon stands in for the flat arrow's
            # purple-to-gold color shift, since a photo's own colors
            # can't be recolored the same way.
            if self.flash > 0:
                glow_radius = round(self.icon_image.get_width() * 0.7)
                glow = pygame.Surface(
                    (glow_radius * 2, glow_radius * 2), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    glow,
                    (*CURSOR_FLASH_COLOR, round(160 * self.flash)),
                    (glow_radius, glow_radius),
                    glow_radius,
                )
                surface.blit(glow, glow.get_rect(center=(x, y)))

            icon = self.icon_image
            if scale != 1.0:
                size = round(icon.get_width() * scale)
                icon = pygame.transform.smoothscale(icon, (size, size))
            # Point the icon inward, toward the cookie -- same direction
            # the flat arrow points below. pygame rotates counter-
            # clockwise for positive degrees, the opposite convention
            # from the math.radians() angles used elsewhere, hence the
            # minus sign.
            rotated = pygame.transform.rotate(icon, -(self.angle + 180))
            surface.blit(rotated, rotated.get_rect(center=(x, y)))
            return

        # Point the arrow inward, toward the cookie. A point sitting at
        # orbit angle theta is offset from the center by (cos theta, sin
        # theta) * radius, so the direction BACK toward the center is
        # just that angle rotated 180 degrees.
        facing = math.radians(self.angle + 180)
        cos_a, sin_a = math.cos(facing), math.sin(facing)

        color = tuple(
            int(
                CURSOR_COLOR[i] + (CURSOR_FLASH_COLOR[i] - CURSOR_COLOR[i]) * self.flash
            )
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
cookies = STARTING_COOKIES
cookies_per_second = 0
cursor_cost = STARTING_CURSOR_COST
cursors = []  # visual list -- one Cursor object per upgrade purchased


def click_cookie():
    global cookies
    cookies += COOKIES_PER_CLICK
    play_sound(click_sound)


def buy_cursor():
    global cookies, cookies_per_second, cursor_cost
    if cookies >= cursor_cost:
        cookies -= cursor_cost
        cookies_per_second += COOKIES_PER_SECOND_PER_CURSOR
        cursor_cost = round(cursor_cost * CURSOR_COST_GROWTH)
        cursors.append(Cursor(len(cursors), icon_image=cursor_icon_image))
        play_sound(purchase_sound)


background_image = load_cover_image(BACKGROUND_IMAGE_PATH, WINDOW_WIDTH, WINDOW_HEIGHT)
cookie_image = load_circular_image(
    COOKIE_IMAGE_PATH, 140
)  # 140 = 2 * cookie's radius (70)
cursor_icon_image = load_square_image(CURSOR_ICON_IMAGE_PATH, CURSOR_ICON_SIZE)
click_sound = load_sound(CLICK_SOUND_PATH)
purchase_sound = load_sound(PURCHASE_SOUND_PATH)

mouse_cursor_image = load_square_image(MOUSE_CURSOR_IMAGE_PATH, 32)
if mouse_cursor_image is not None:
    try:
        # Hotspot is the pixel within the image that actually "clicks" --
        # (16, 16) is the center of the 32x32 icon; move it if your art
        # needs the click point somewhere else, like the tip of an arrow.
        pygame.mouse.set_cursor(pygame.cursors.Cursor((16, 16), mouse_cursor_image))
    except pygame.error as e:
        print(
            f"Could not set custom mouse cursor ({e}). Using the default system cursor."
        )

cookie = CookieButton(260, 220, 70, click_cookie, image=cookie_image)
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

    cursor_button.label = f"Buy Cursor +1/s (Cost: {format_number(cursor_cost)})"
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
    if background_image is not None:
        screen.blit(background_image, (0, 0))
    else:
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

    count_surface = big_font.render(
        f"{format_number(cookies)} cookies", True, TEXT_COLOR
    )
    screen.blit(count_surface, (20, 20))

    rate_surface = status_font.render(
        f"{format_number(cookies_per_second)} per second", True, SUBTEXT_COLOR
    )
    screen.blit(rate_surface, (20, 70))

    owned_surface = status_font.render(
        f"Cursors owned: {len(cursors)}", True, SUBTEXT_COLOR
    )
    screen.blit(owned_surface, (20, 100))

    pygame.display.flip()

pygame.quit()
