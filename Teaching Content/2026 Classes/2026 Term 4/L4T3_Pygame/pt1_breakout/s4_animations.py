"""
Breakout -- Stage 4: Juice (final)
=====================================

Everything from Stage 3 is still here -- states, lives, scoring, all of
it unchanged. This stage adds the same kind of "juice" Cookie Clicker's
Stage 4 added to its cursors: things that fire once and fade, instead
of anything that changes the actual rules.

Three additions, each reusing a pattern from earlier work:

  1. A paddle FLASH on hit -- the exact same "blend two colors based on
     a 0..1 value that decays over time" trick as Cookie Clicker's
     Cursor flash, just applied to a rectangle instead of an arrow.

  2. Brick-break PARTICLES and a floating "+N" score popup -- plain
     dicts in a list, no new class, same as Cookie Clicker's
     floating_texts. The particle fade reuses Session 6's SRCALPHA
     overlay trick (a translucent surface you can set_alpha() on)
     instead of just fading a text surface.

  3. An OPTIONAL sound hook for the paddle and bricks -- same
     safe-if-missing pattern as Session 6's icon button: check
     os.path.exists() first, and if there's no .wav file (or no audio
     device at all) the game just runs silently instead of crashing.

The one structural idea worth pointing out: Ball doesn't know what a
particle is, and it never will. When it destroys a brick, it calls
whatever function it was handed (`on_brick_broken`) with the brick's
position, color, and value -- the EXACT same separation as a Button's
`on_click`. Ball's job is physics; deciding what "a brick just broke"
should look like on screen belongs somewhere else.

CUSTOMIZE ME: the flash color, particle count/speed/lifetime, and popup
color are all new CONFIG entries. Drop your own paddle_hit.wav /
brick_break.wav next to this file to turn sound on for free.
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
BALL_SPEED = 5            # total pixels moved per frame

BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_WIDTH = 56
BRICK_HEIGHT = 20
BRICK_PADDING = 6         # gap between bricks, both directions
BRICK_TOP_OFFSET = 60     # distance from the top of the window to row 0

BRICK_ROW_COLORS = [
    (235, 90, 90),
    (240, 150, 70),
    (235, 210, 70),
    (120, 200, 100),
    (100, 170, 235),
]
BRICK_ROW_SCORES = [50, 40, 30, 20, 10]

BRICK_PARTICLE_COUNT = 8            # particles spawned per broken brick
BRICK_PARTICLE_SPEED_RANGE = (60, 160)   # pixels per second, min/max
BRICK_PARTICLE_LIFETIME_MS = 400
BRICK_PARTICLE_SIZE = 5

SCORE_POPUP_COLOR = (255, 230, 140)
SCORE_POPUP_LIFETIME_MS = 700

STARTING_LIVES = 3

# Optional sound -- drop matching .wav files next to this script to
# turn these on. No files, no audio device? The game just stays quiet.
PADDLE_HIT_SOUND_PATH = "paddle_hit.wav"
BRICK_BREAK_SOUND_PATH = "brick_break.wav"

FPS = 60

# ---------------------------------------------------------------------------
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakout")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 48)
popup_font = pygame.font.SysFont(None, 24)

SOUND_ENABLED = True
try:
    pygame.mixer.init()
except pygame.error:
    # Some machines (or sandboxes) don't have a working audio device --
    # the game should still run fine without sound rather than crashing.
    SOUND_ENABLED = False

paddle_hit_sound = None
brick_break_sound = None
if SOUND_ENABLED:
    if os.path.exists(PADDLE_HIT_SOUND_PATH):
        paddle_hit_sound = pygame.mixer.Sound(PADDLE_HIT_SOUND_PATH)
    if os.path.exists(BRICK_BREAK_SOUND_PATH):
        brick_break_sound = pygame.mixer.Sound(BRICK_BREAK_SOUND_PATH)


def draw_centered_text(surface, text, font_obj, color, y_offset=0):
    text_surface = font_obj.render(text, True, color)
    rect = text_surface.get_rect(
        center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + y_offset)
    )
    surface.blit(text_surface, rect)


class Paddle:
    """Same rectangle and movement as Stage 1-3, plus a `flash` value
    (0..1) that spikes to 1 on a paddle hit and decays back to 0, the
    same shape of idea as Cookie Clicker's Cursor.flash.
    """

    def __init__(self):
        self.rect = pygame.Rect(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.rect.centerx = WINDOW_WIDTH // 2
        self.rect.bottom = WINDOW_HEIGHT - PADDLE_Y_OFFSET
        self.flash = 0.0

    def hit_flash(self):
        self.flash = 1.0

    def update(self, keys, dt):
        if keys[pygame.K_LEFT]:
            self.rect.x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT]:
            self.rect.x += PADDLE_SPEED

        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(WINDOW_WIDTH, self.rect.right)

        self.flash = max(0.0, self.flash - dt / PADDLE_FLASH_DECAY_MS)

    def draw(self, surface):
        # Blend channel-by-channel toward the flash color -- identical
        # technique to Cookie Clicker's Cursor color interpolation.
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
# Particles and score popups -- plain dicts in a list, same pattern as
# Cookie Clicker's floating_texts. Nothing here needs its own class
# since a particle doesn't do anything but exist, drift, and fade.
# ---------------------------------------------------------------------------
particles = []
score_popups = []


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


def spawn_score_popup(center, value):
    cx, cy = center
    score_popups.append({"x": cx, "y": cy, "age": 0.0, "text": f"+{value}"})


def on_brick_broken(center, color, value):
    """Handed to Ball as a callback -- the exact same separation of
    concerns as a Button's on_click. Ball just calls this; it has no
    idea particles or popups exist.
    """
    spawn_particles(center, color)
    spawn_score_popup(center, value)
    if brick_break_sound:
        brick_break_sound.play()


def update_particles(dt):
    for p in particles[:]:
        p["age"] += dt
        p["x"] += p["vx"] * (dt / 1000)
        p["y"] += p["vy"] * (dt / 1000)
        if p["age"] >= BRICK_PARTICLE_LIFETIME_MS:
            particles.remove(p)


def update_score_popups(dt):
    for pop in score_popups[:]:
        pop["age"] += dt
        pop["y"] -= dt * 0.04  # drift upward, same rate as Cookie Clicker's popup
        if pop["age"] >= SCORE_POPUP_LIFETIME_MS:
            score_popups.remove(pop)


def draw_particles(surface):
    for p in particles:
        remaining = max(0.0, 1 - (p["age"] / BRICK_PARTICLE_LIFETIME_MS))
        size = BRICK_PARTICLE_SIZE
        # Same SRCALPHA-surface trick as Session 6's hover overlay:
        # draw fully opaque onto its own surface, then fade THAT
        # surface's alpha before blitting it.
        particle_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(particle_surface, p["color"], particle_surface.get_rect())
        particle_surface.set_alpha(int(255 * remaining))
        surface.blit(particle_surface, (p["x"] - size / 2, p["y"] - size / 2))


def draw_score_popups(surface):
    for pop in score_popups:
        remaining = max(0.0, 1 - (pop["age"] / SCORE_POPUP_LIFETIME_MS))
        text_surface = popup_font.render(pop["text"], True, SCORE_POPUP_COLOR)
        text_surface.set_alpha(int(255 * remaining))
        surface.blit(text_surface, (pop["x"], pop["y"]))


class Ball:
    """Same physics as Stage 3. update() takes one new argument --
    on_brick_broken -- and calls it exactly once, right where a brick
    is destroyed, instead of the game loop having to guess when that
    happened.
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
        self.x += self.dx
        self.y += self.dy

        if self.x - BALL_RADIUS <= 0 or self.x + BALL_RADIUS >= WINDOW_WIDTH:
            self.dx *= -1

        if self.y - BALL_RADIUS <= 0:
            self.dy *= -1

        if self.dy > 0 and self.rect().colliderect(paddle.rect):
            self.dy *= -1
            offset = (self.x - paddle.rect.centerx) / (PADDLE_WIDTH / 2)
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
# Game state -- same menu/playing/game_over/win pattern as Stage 3.
# ---------------------------------------------------------------------------
game_state = "menu"

paddle = None
ball = None
bricks = []
score = 0
lives = STARTING_LIVES


def reset_game():
    """Same reset discipline as Stage 3 -- every variable a round can
    change gets reset here, including the new particle/popup lists so
    a fresh game doesn't start with leftover debris on screen.
    """
    global paddle, ball, bricks, score, lives, particles, score_popups

    paddle = Paddle()
    ball = Ball()
    bricks = build_bricks()
    score = 0
    lives = STARTING_LIVES
    particles = []
    score_popups = []


running = True
while running:
    dt = clock.tick(FPS)  # milliseconds since the last frame

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

        missed, points = ball.update(paddle, bricks, on_brick_broken)
        score += points

        if missed:
            lives -= 1
            if lives <= 0:
                game_state = "game_over"
            else:
                ball.reset()
        elif not bricks:
            game_state = "win"

    # Particles and popups keep animating even after the round ends, so
    # the last brick's burst gets to finish instead of cutting off.
    update_particles(dt)
    update_score_popups(dt)

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    if game_state != "menu":
        for brick in bricks:
            brick.draw(screen)
        draw_particles(screen)
        paddle.draw(screen)
        ball.draw(screen)
        draw_score_popups(screen)

        score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_surface, (20, 20))

        remaining_surface = font.render(f"Bricks remaining: {len(bricks)}", True, TEXT_COLOR)
        screen.blit(remaining_surface, (20, 50))

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