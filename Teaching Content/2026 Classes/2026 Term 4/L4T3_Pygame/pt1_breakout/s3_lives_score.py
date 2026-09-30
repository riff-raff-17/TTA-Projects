"""
Breakout -- Stage 3: Lives, Score, and Game States
=====================================================

Everything from Stage 2 is still here -- Paddle, Brick, and
build_bricks() are all unchanged. This stage wraps the game in the same
three-state pattern Session 5 used for Snake: `game_state` holds
"menu", "playing", "game_over", or "win" (one extra option here, same
idea), and that single variable decides which update/draw logic runs
each frame, exactly like Session 5's menu/playing/game_over routing.

The one real behaviour change is in Ball. Stage 2's ball reset itself
for free every time it fell off the bottom -- fine for testing
physics, but not an actual game. Now missing the ball means losing a
life, and only the GAME LOOP knows whether any lives are left, so
reset() moved out of Ball.update() and into the loop: update() just
reports "did I get missed this frame?" (and how many points were just
scored) and lets the caller decide what that means.

Scoring reuses the row grouping bricks already had for color: each row
now also has a point value, top rows worth more -- the same
"index into a list with `row % len(list)`" trick BRICK_ROW_COLORS used,
just for numbers instead of colors.

CUSTOMIZE ME: STARTING_LIVES and BRICK_ROW_SCORES are new. Change how
forgiving the game is, or how much a top-row brick is worth relative to
a bottom-row one, without touching anything below CONFIG.
"""

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
PADDLE_SPEED = 7  # pixels moved per frame while a key is held
PADDLE_Y_OFFSET = 40  # distance from the bottom edge

BALL_RADIUS = 9
BALL_SPEED = 5  # total pixels moved per frame

BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_WIDTH = 56
BRICK_HEIGHT = 20
BRICK_PADDING = 6  # gap between bricks, both directions
BRICK_TOP_OFFSET = 60  # distance from the top of the window to row 0

# One color per row, top to bottom. If BRICK_ROWS is longer than this
# list, colors repeat -- change these to reskin the whole grid.
BRICK_ROW_COLORS = [
    (235, 90, 90),
    (240, 150, 70),
    (235, 210, 70),
    (120, 200, 100),
    (100, 170, 235),
]

# Points for a brick in each row, top to bottom -- top rows are worth
# more, same idea as classic Breakout. Same repeat-if-shorter rule as
# BRICK_ROW_COLORS above.
BRICK_ROW_SCORES = [50, 40, 30, 20, 10]

STARTING_LIVES = 3

FPS = 60

# ---------------------------------------------------------------------------
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakout -- Stage 3: Lives, Score, and Game States")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 48)


def draw_centered_text(surface, text, font_obj, color, y_offset=0):
    """Same helper as Session 5 -- centers text using the rendered
    surface's own size instead of guessing pixel coordinates by hand.
    """
    text_surface = font_obj.render(text, True, color)
    rect = text_surface.get_rect(
        center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + y_offset)
    )
    surface.blit(text_surface, rect)


class Paddle:
    """Unchanged from Stage 1."""

    def __init__(self):
        self.rect = pygame.Rect(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.rect.centerx = WINDOW_WIDTH // 2
        self.rect.bottom = WINDOW_HEIGHT - PADDLE_Y_OFFSET

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT]:
            self.rect.x += PADDLE_SPEED

        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(WINDOW_WIDTH, self.rect.right)

    def draw(self, surface):
        pygame.draw.rect(surface, PADDLE_COLOR, self.rect, border_radius=4)


class Brick:
    """Unchanged from Stage 2 except for one new attribute: `value`,
    the score earned for destroying this particular brick.
    """

    def __init__(self, x, y, color, value):
        self.rect = pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)
        self.color = color
        self.value = value

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)


def build_bricks():
    """Unchanged in shape from Stage 2 -- just also looks up each row's
    point value alongside its color.
    """
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


class Ball:
    """Same physics as Stage 2. The difference is what update() reports
    back instead of handling on its own: it now returns
    (missed, points_scored) every frame instead of silently
    resetting itself when it falls off the bottom.
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

    def handle_brick_collision(self, bricks):
        """Same overlap-width-vs-height bounce logic as Stage 2. Now
        returns the value of whichever brick was destroyed (0 if none
        was hit this frame), so the game loop can add it to the score.
        """
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
            return brick.value

        return 0

    def update(self, paddle, bricks):
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

        points = self.handle_brick_collision(bricks)

        # No more auto-reset here -- the game loop checks this return
        # value and decides whether that costs a life or ends the game.
        missed = self.y - BALL_RADIUS > WINDOW_HEIGHT

        return missed, points

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
# game_state holds "menu", "playing", "game_over", or "win" -- the same
# single-variable-routes-everything pattern as Session 5, just with a
# fourth option since Breakout can be WON as well as lost.
game_state = "menu"

paddle = None
ball = None
bricks = []
score = 0
lives = STARTING_LIVES


def reset_game():
    """Reinitialise every variable that changes during play -- same
    discipline as Session 5's reset_game(): every piece of state that a
    round can change gets reset here, not just some of them.
    """
    global paddle, ball, bricks, score, lives

    paddle = Paddle()
    ball = Ball()
    bricks = build_bricks()
    score = 0
    lives = STARTING_LIVES


running = True
while running:
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
        paddle.update(keys)

        missed, points = ball.update(paddle, bricks)
        score += points

        if missed:
            lives -= 1
            if lives <= 0:
                game_state = "game_over"
            else:
                ball.reset()
        elif not bricks:
            game_state = "win"

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    if game_state != "menu":
        # Draw the board underneath the message on game_over/win, same
        # as Session 5 still drawing the snake beneath "GAME OVER".
        for brick in bricks:
            brick.draw(screen)
        paddle.draw(screen)
        ball.draw(screen)

        score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_surface, (20, 20))

        remaining_surface = font.render(
            f"Bricks remaining: {len(bricks)}", True, TEXT_COLOR
        )
        screen.blit(remaining_surface, (20, 50))

        lives_surface = font.render(f"Lives: {lives}", True, TEXT_COLOR)
        lives_rect = lives_surface.get_rect(topright=(WINDOW_WIDTH - 20, 20))
        screen.blit(lives_surface, lives_rect)

    if game_state == "menu":
        draw_centered_text(screen, "BREAKOUT", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(
            screen, "Press SPACE to start", font, TEXT_COLOR, y_offset=20
        )

    elif game_state == "game_over":
        draw_centered_text(screen, "GAME OVER", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(
            screen,
            f"Score: {score}  --  Press R to restart",
            font,
            TEXT_COLOR,
            y_offset=20,
        )

    elif game_state == "win":
        draw_centered_text(screen, "YOU WIN!", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(
            screen,
            f"Score: {score}  --  Press R to play again",
            font,
            TEXT_COLOR,
            y_offset=20,
        )

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
