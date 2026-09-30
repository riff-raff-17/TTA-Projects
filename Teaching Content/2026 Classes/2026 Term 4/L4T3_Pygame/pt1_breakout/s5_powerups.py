"""
Breakout -- Stage 5: Power-Ups (chaos edition)
=================================================

Everything from Stage 4 is still here. This stage is almost entirely
REUSE of tools already built, pointed at a new idea:

  - on_brick_broken (the same callback Ball calls for particles and
    score popups) now ALSO decides whether a broken brick drops a
    falling capsule. Ball still has no idea any of this exists.

  - spawn_score_popup became spawn_popup_text -- the exact same
    floating-fading-text mechanism, just no longer hard-coded to
    "+N". Catching a power-up spawns "MULTI-BALL!" through the same
    pipe that spawns "+50".

  - Paddle's flash and the optional catch sound get reused as generic
    "you caught something" feedback, whether that something helps you
    or hurts you.

  - dt-based timers (from Stage 4's flash/particle decay) come back
    for power-up DURATIONS -- a paddle resize or speed boost counts
    down in milliseconds exactly like a particle's fade.

The one genuinely new structural idea: `ball` becomes `balls`, a LIST,
the same "one object becomes a growing/shrinking list" pattern as
Cookie Clicker's cursors or Snake's body segments. That single change
is what makes Multi-Ball possible, and it changes what "losing" means:
a life is only lost once EVERY ball in the list is gone, not the first
time any one of them falls.

Power-ups are intentionally a mix of helpful and harmful -- catching
one is a small gamble, which is where the "chaos" feeling mostly comes
from. You don't know if the capsule falling at you is a gift or a
trap until it lands on your paddle.

CUSTOMIZE ME: the whole POWER-UPS config block below -- drop chance,
colors, labels, durations, magnitudes. Try adding a fifth kind as an
exercise: everything you need (spawn, fall, catch, apply, revert) is
already right here to copy.
"""

import math
import os
import random

import pygame

pygame.init()

# ---------------------------------------------------------------------------
# CONFIG -- make this yours. Nothing below this section needs to change
# to completely reskin the game.
# ---------------------------------------------------------------------------
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500

BACKGROUND_COLOR = (18, 18, 24)
PADDLE_COLOR = (80, 200, 255)
BALL_COLOR = (255, 210, 80)
TEXT_COLOR = (230, 230, 230)

PADDLE_WIDTH = 100
PADDLE_HEIGHT = 14
PADDLE_SPEED = 7          # pixels moved per frame while a key is held
PADDLE_Y_OFFSET = 40      # distance from the bottom edge
PADDLE_FLASH_COLOR = (255, 255, 255)
PADDLE_FLASH_DECAY_MS = 150   # time for the hit-flash to fully fade

BALL_RADIUS = 9
BALL_SPEED = 5             # total pixels moved per frame, before any boost

BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_WIDTH = 56
BRICK_HEIGHT = 20
BRICK_PADDING = 6
BRICK_TOP_OFFSET = 60

BRICK_ROW_COLORS = [
    (235, 90, 90),
    (240, 150, 70),
    (235, 210, 70),
    (120, 200, 100),
    (100, 170, 235),
]
BRICK_ROW_SCORES = [50, 40, 30, 20, 10]

BRICK_PARTICLE_COUNT = 8
BRICK_PARTICLE_SPEED_RANGE = (60, 160)
BRICK_PARTICLE_LIFETIME_MS = 400
BRICK_PARTICLE_SIZE = 5

POPUP_COLOR = (255, 230, 140)
POPUP_LIFETIME_MS = 700

STARTING_LIVES = 3

# --- Power-ups -------------------------------------------------------------
POWERUP_DROP_CHANCE = 0.5      # chance a broken brick drops a capsule
POWERUP_SIZE = 24
POWERUP_FALL_SPEED = 140        # pixels per second

POWERUP_COLORS = {
    "multiball": (255, 120, 190),
    "grow": (120, 220, 140),
    "shrink": (220, 90, 90),
    "fast": (250, 200, 60),
}
POWERUP_LABELS = {
    "multiball": "M",
    "grow": "+",
    "shrink": "-",
    "fast": ">>",
}
POWERUP_NAMES = {
    "multiball": "MULTI-BALL!",
    "grow": "BIG PADDLE!",
    "shrink": "SMALL PADDLE!",
    "fast": "SPEED UP!",
}

PADDLE_GROW_WIDTH = 160
PADDLE_SHRINK_WIDTH = 60
PADDLE_SIZE_EFFECT_DURATION_MS = 8000

BALL_FAST_MULTIPLIER = 1.8
BALL_FAST_DURATION_MS = 6000

# Optional sound -- drop matching .wav files next to this script to
# turn these on. No files, no audio device? The game just stays quiet.
PADDLE_HIT_SOUND_PATH = "paddle_hit.wav"
BRICK_BREAK_SOUND_PATH = "brick_break.wav"
POWERUP_CATCH_SOUND_PATH = "powerup_catch.wav"

FPS = 60

# ---------------------------------------------------------------------------
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakout -- Stage 5: Power-Ups")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 48)
popup_font = pygame.font.SysFont(None, 24)

SOUND_ENABLED = True
try:
    pygame.mixer.init()
except pygame.error:
    SOUND_ENABLED = False

paddle_hit_sound = None
brick_break_sound = None
powerup_catch_sound = None
if SOUND_ENABLED:
    if os.path.exists(PADDLE_HIT_SOUND_PATH):
        paddle_hit_sound = pygame.mixer.Sound(PADDLE_HIT_SOUND_PATH)
    if os.path.exists(BRICK_BREAK_SOUND_PATH):
        brick_break_sound = pygame.mixer.Sound(BRICK_BREAK_SOUND_PATH)
    if os.path.exists(POWERUP_CATCH_SOUND_PATH):
        powerup_catch_sound = pygame.mixer.Sound(POWERUP_CATCH_SOUND_PATH)


def draw_centered_text(surface, text, font_obj, color, y_offset=0):
    text_surface = font_obj.render(text, True, color)
    rect = text_surface.get_rect(
        center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + y_offset)
    )
    surface.blit(text_surface, rect)


class Paddle:
    """Same as Stage 4, plus the ability to temporarily change width.
    base_width is what it always reverts to; size_timer_ms counts down
    the same way flash does, just measured in whole seconds instead of
    fractions of one.
    """

    def __init__(self):
        self.base_width = PADDLE_WIDTH
        self.rect = pygame.Rect(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.rect.centerx = WINDOW_WIDTH // 2
        self.rect.bottom = WINDOW_HEIGHT - PADDLE_Y_OFFSET
        self.flash = 0.0
        self.size_timer_ms = 0.0

    def hit_flash(self):
        self.flash = 1.0

    def _resize(self, width):
        """Change width while keeping the paddle centered where it
        already was -- otherwise growing or shrinking would visibly
        yank the paddle sideways.
        """
        center = self.rect.centerx
        self.rect.width = width
        self.rect.centerx = center
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(WINDOW_WIDTH, self.rect.right)

    def apply_size_effect(self, width, duration_ms):
        self._resize(width)
        self.size_timer_ms = duration_ms

    def update(self, keys, dt):
        if keys[pygame.K_LEFT]:
            self.rect.x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT]:
            self.rect.x += PADDLE_SPEED

        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(WINDOW_WIDTH, self.rect.right)

        self.flash = max(0.0, self.flash - dt / PADDLE_FLASH_DECAY_MS)

        if self.size_timer_ms > 0:
            self.size_timer_ms -= dt
            if self.size_timer_ms <= 0:
                self.size_timer_ms = 0
                self._resize(self.base_width)

    def draw(self, surface):
        color = tuple(
            int(PADDLE_COLOR[i] + (PADDLE_FLASH_COLOR[i] - PADDLE_COLOR[i]) * self.flash)
            for i in range(3)
        )
        pygame.draw.rect(surface, color, self.rect, border_radius=4)


class Brick:
    """Unchanged from Stage 3."""

    def __init__(self, x, y, color, value):
        self.rect = pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)
        self.color = color
        self.value = value

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)


def build_bricks():
    """Unchanged from Stage 3."""
    bricks = []
    grid_width = BRICK_COLS * (BRICK_WIDTH + BRICK_PADDING) - BRICK_PADDING
    left_margin = (WINDOW_WIDTH - grid_width) // 2

    for row in range(BRICK_ROWS):
        color = BRICK_ROW_COLORS[row % len(BRICK_ROW_COLORS)]
        value = BRICK_ROW_SCORES[row % len(BRICK_ROW_SCORES)]
        y = BRICK_TOP_OFFSET + row * (BRICK_HEIGHT + BRICK_PADDING)
        for col in range(BRICK_COLS):
            x = left_margin + col * (BRICK_WIDTH + BRICK_PADDING)
            bricks.append(Brick(x, y, color, value))

    return bricks


# ---------------------------------------------------------------------------
# Particles and popups -- same dict-list pattern as Stage 4.
# spawn_score_popup is now spawn_popup_text: the exact same mechanism,
# generalized to show any short message, not just "+N".
# ---------------------------------------------------------------------------
particles = []
popups = []


def spawn_particles(center, color):
    cx, cy = center
    for _ in range(BRICK_PARTICLE_COUNT):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*BRICK_PARTICLE_SPEED_RANGE)
        particles.append({
            "x": cx,
            "y": cy,
            "vx": math.cos(angle) * speed,
            "vy": math.sin(angle) * speed,
            "age": 0.0,
            "color": color,
        })


def spawn_popup_text(center, text):
    cx, cy = center
    popups.append({"x": cx, "y": cy, "age": 0.0, "text": text})


def update_particles(dt):
    for p in particles[:]:
        p["age"] += dt
        p["x"] += p["vx"] * (dt / 1000)
        p["y"] += p["vy"] * (dt / 1000)
        if p["age"] >= BRICK_PARTICLE_LIFETIME_MS:
            particles.remove(p)


def update_popups(dt):
    for pop in popups[:]:
        pop["age"] += dt
        pop["y"] -= dt * 0.04
        if pop["age"] >= POPUP_LIFETIME_MS:
            popups.remove(pop)


def draw_particles(surface):
    for p in particles:
        remaining = max(0.0, 1 - (p["age"] / BRICK_PARTICLE_LIFETIME_MS))
        size = BRICK_PARTICLE_SIZE
        particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(particle_surface, p["color"], particle_surface.get_rect())
        particle_surface.set_alpha(int(255 * remaining))
        surface.blit(particle_surface, (p["x"] - size / 2, p["y"] - size / 2))


def draw_popups(surface):
    for pop in popups:
        remaining = max(0.0, 1 - (pop["age"] / POPUP_LIFETIME_MS))
        text_surface = popup_font.render(pop["text"], True, POPUP_COLOR)
        text_surface.set_alpha(int(255 * remaining))
        surface.blit(text_surface, (pop["x"], pop["y"]))


# ---------------------------------------------------------------------------
# Power-ups -- a falling capsule (position + kind, nothing more) and a
# handful of functions that spawn, move, and apply them. Same shape as
# everything else in this file: small pieces of state, checked and
# updated once per frame.
# ---------------------------------------------------------------------------
powerups = []

# These two get reassigned by apply_powerup() below, so the "fast ball"
# effect is declared here at module level rather than living inside
# any one ball -- every ball currently in play (and any spawned while
# it's active) shares the same boost.
ball_speed_multiplier = 1.0
ball_speed_timer_ms = 0.0


class PowerUp:
    def __init__(self, kind, center):
        self.kind = kind
        self.rect = pygame.Rect(0, 0, POWERUP_SIZE, POWERUP_SIZE)
        self.rect.center = center

    def update(self, dt):
        self.rect.y += POWERUP_FALL_SPEED * (dt / 1000)

    def draw(self, surface):
        pygame.draw.rect(surface, POWERUP_COLORS[self.kind], self.rect, border_radius=5)
        label_surface = popup_font.render(POWERUP_LABELS[self.kind], True, (25, 25, 30))
        surface.blit(label_surface, label_surface.get_rect(center=self.rect.center))


def spawn_multiball():
    """Clone whichever ball is first in the list into two more, each
    given a slightly different sideways speed so all three fan out
    instead of overlapping perfectly.
    """
    if not balls:
        return
    source = balls[0]
    for angle_offset in (-0.6, 0.6):
        clone = Ball()
        clone.x = source.x
        clone.y = source.y
        clone.dx = source.dx + angle_offset * BALL_SPEED
        clone.dy = source.dy
        balls.append(clone)


def apply_powerup(kind, catch_center):
    """Runs once, the moment a capsule touches the paddle. Everything
    here is either a one-off action (multiball) or "set a value and a
    timer," the same pattern Paddle.apply_size_effect and the fast-ball
    globals both use.
    """
    global ball_speed_multiplier, ball_speed_timer_ms

    if kind == "multiball":
        spawn_multiball()
    elif kind == "grow":
        paddle.apply_size_effect(PADDLE_GROW_WIDTH, PADDLE_SIZE_EFFECT_DURATION_MS)
    elif kind == "shrink":
        paddle.apply_size_effect(PADDLE_SHRINK_WIDTH, PADDLE_SIZE_EFFECT_DURATION_MS)
    elif kind == "fast":
        ball_speed_multiplier = BALL_FAST_MULTIPLIER
        ball_speed_timer_ms = BALL_FAST_DURATION_MS

    spawn_popup_text(catch_center, POWERUP_NAMES[kind])
    paddle.hit_flash()
    if powerup_catch_sound:
        powerup_catch_sound.play()


def on_brick_broken(center, color, value):
    """Same callback Ball has always called. The only change from
    Stage 4 is one more thing it can decide to do: roll the dice on
    dropping a power-up.
    """
    spawn_particles(center, color)
    spawn_popup_text(center, f"+{value}")
    if brick_break_sound:
        brick_break_sound.play()

    if random.random() < POWERUP_DROP_CHANCE:
        kind = random.choice(list(POWERUP_COLORS.keys()))
        powerups.append(PowerUp(kind, center))


class Ball:
    """Same physics as Stage 4. The only change: its movement each
    frame is scaled by the module-level ball_speed_multiplier, so a
    "fast ball" power-up speeds up every ball at once without any of
    them needing to know why.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = float(WINDOW_WIDTH // 2)
        self.y = float(
            WINDOW_HEIGHT - PADDLE_Y_OFFSET - PADDLE_HEIGHT - BALL_RADIUS - 1
        )
        sideways_options = [-0.6, -0.3, 0.3, 0.6]
        self.dx = random.choice(sideways_options) * BALL_SPEED
        self.dy = -BALL_SPEED

    def rect(self):
        return pygame.Rect(
            self.x - BALL_RADIUS,
            self.y - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

    def handle_brick_collision(self, bricks, on_brick_broken):
        ball_rect = self.rect()
        for brick in bricks:
            if not ball_rect.colliderect(brick.rect):
                continue

            overlap = ball_rect.clip(brick.rect)
            if overlap.width < overlap.height:
                self.dx *= -1
            else:
                self.dy *= -1

            bricks.remove(brick)
            on_brick_broken(brick.rect.center, brick.color, brick.value)
            return brick.value

        return 0

    def update(self, paddle, bricks, on_brick_broken):
        self.x += self.dx * ball_speed_multiplier
        self.y += self.dy * ball_speed_multiplier

        if self.x - BALL_RADIUS <= 0 or self.x + BALL_RADIUS >= WINDOW_WIDTH:
            self.dx *= -1

        if self.y - BALL_RADIUS <= 0:
            self.dy *= -1

        if self.dy > 0 and self.rect().colliderect(paddle.rect):
            self.dy *= -1
            # Uses the paddle's CURRENT width, not the base constant --
            # important now that grow/shrink can change it mid-game.
            offset = (self.x - paddle.rect.centerx) / (paddle.rect.width / 2)
            self.dx = offset * BALL_SPEED
            paddle.hit_flash()
            if paddle_hit_sound:
                paddle_hit_sound.play()

        points = self.handle_brick_collision(bricks, on_brick_broken)
        missed = self.y - BALL_RADIUS > WINDOW_HEIGHT

        return missed, points

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
game_state = "menu"

paddle = None
balls = []
bricks = []
score = 0
lives = STARTING_LIVES


def reset_game():
    global paddle, balls, bricks, score, lives
    global particles, popups, powerups, ball_speed_multiplier, ball_speed_timer_ms

    paddle = Paddle()
    balls = [Ball()]
    bricks = build_bricks()
    score = 0
    lives = STARTING_LIVES
    particles = []
    popups = []
    powerups = []
    ball_speed_multiplier = 1.0
    ball_speed_timer_ms = 0.0


running = True
while running:
    dt = clock.tick(FPS)

    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if game_state == "menu" and event.key == pygame.K_SPACE:
                reset_game()
                game_state = "playing"

            elif game_state in ("game_over", "win") and event.key == pygame.K_r:
                reset_game()
                game_state = "playing"

    # --- 2. Update state ------------------------------------------------------
    if game_state == "playing":
        keys = pygame.key.get_pressed()
        paddle.update(keys, dt)

        if ball_speed_timer_ms > 0:
            ball_speed_timer_ms -= dt
            if ball_speed_timer_ms <= 0:
                ball_speed_timer_ms = 0
                ball_speed_multiplier = 1.0

        # Same "loop over a copy, remove from the real list" pattern as
        # the particle/popup lists -- now applied to something that
        # actually affects whether you lose.
        for ball in balls[:]:
            missed, points = ball.update(paddle, bricks, on_brick_broken)
            score += points
            if missed:
                balls.remove(ball)

        for p in powerups[:]:
            p.update(dt)
            if p.rect.colliderect(paddle.rect):
                apply_powerup(p.kind, p.rect.center)
                powerups.remove(p)
            elif p.rect.top > WINDOW_HEIGHT:
                powerups.remove(p)

        if not balls:
            # A life is only lost once EVERY ball is gone -- the whole
            # point of Multi-Ball is buying insurance against exactly
            # this moment.
            lives -= 1
            if lives <= 0:
                game_state = "game_over"
            else:
                balls = [Ball()]
        elif not bricks:
            game_state = "win"

    update_particles(dt)
    update_popups(dt)

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    if game_state != "menu":
        for brick in bricks:
            brick.draw(screen)
        draw_particles(screen)
        for p in powerups:
            p.draw(screen)
        paddle.draw(screen)
        for ball in balls:
            ball.draw(screen)
        draw_popups(screen)

        score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_surface, (20, 20))

        remaining_surface = font.render(f"Bricks remaining: {len(bricks)}", True, TEXT_COLOR)
        screen.blit(remaining_surface, (20, 50))

        status_bits = []
        if ball_speed_multiplier > 1.0:
            status_bits.append("FAST BALL")
        if paddle.rect.width > paddle.base_width:
            status_bits.append("BIG PADDLE")
        elif paddle.rect.width < paddle.base_width:
            status_bits.append("SMALL PADDLE")
        if status_bits:
            status_surface = font.render(" | ".join(status_bits), True, POPUP_COLOR)
            screen.blit(status_surface, (20, 80))

        lives_surface = font.render(f"Lives: {lives}", True, TEXT_COLOR)
        lives_rect = lives_surface.get_rect(topright=(WINDOW_WIDTH - 20, 20))
        screen.blit(lives_surface, lives_rect)

    if game_state == "menu":
        draw_centered_text(screen, "BREAKOUT", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(screen, "Press SPACE to start", font, TEXT_COLOR, y_offset=20)

    elif game_state == "game_over":
        draw_centered_text(screen, "GAME OVER", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(
            screen, f"Score: {score}  --  Press R to restart", font, TEXT_COLOR, y_offset=20
        )

    elif game_state == "win":
        draw_centered_text(screen, "YOU WIN!", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(
            screen, f"Score: {score}  --  Press R to play again", font, TEXT_COLOR, y_offset=20
        )

    pygame.display.flip()

pygame.quit()