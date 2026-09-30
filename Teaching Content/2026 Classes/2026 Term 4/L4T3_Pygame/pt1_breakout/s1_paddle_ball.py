"""
Breakout -- Stage 1: Paddle and Ball
=====================================

New project, same foundations. The paddle is just a rectangle you move
with the keyboard -- the same Rect + clamp() idea from Session 2's
circle, just constrained to one axis instead of two. The ball is a
circle with a velocity, the same shape of idea as Snake's direction
tuple, except it's pixels-per-frame instead of grid-cells-per-tick, and
it changes sign on a bounce instead of on a keypress.

This stage does ONE thing: get the paddle and ball feeling right. Serve
the ball, keep it in play, don't worry yet about bricks, score, or
losing -- those are next.

CUSTOMIZE ME: everything under CONFIG below is meant to be changed.
Swap the colors, resize the paddle, speed the ball up -- none of it
requires touching the logic further down.
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

FPS = 60

# ---------------------------------------------------------------------------
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakout -- Stage 1: Paddle and Ball")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)


class Paddle:
    """A rectangle that only ever moves left/right, clamped to the window.

    Structurally this is even simpler than Session 6's Button -- no
    click to detect, just a position that keyboard input nudges every
    frame, the same as the circle in Session 2.
    """

    def __init__(self):
        self.rect = pygame.Rect(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.rect.centerx = WINDOW_WIDTH // 2
        self.rect.bottom = WINDOW_HEIGHT - PADDLE_Y_OFFSET

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT]:
            self.rect.x += PADDLE_SPEED

        # Session 2's clamp idea, applied through Rect's own bounds
        # instead of by hand with min()/max() on a bare x value.
        self.rect.left = max(0, self.rect.left)
        self.rect.right = min(WINDOW_WIDTH, self.rect.right)

    def draw(self, surface):
        pygame.draw.rect(surface, PADDLE_COLOR, self.rect, border_radius=4)


class Ball:
    """A circle with a velocity (dx, dy) -- bouncing is just "flip the
    sign of one component when you'd go out of bounds," the same
    comparison logic behind every clamp() and wall check so far.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Put the ball back on the paddle's launch line, moving up and
        slightly sideways. Called at the start, and again any time the
        ball is lost off the bottom -- Stage 3 turns "lost" into
        something that actually costs the player a life instead of a
        free do-over.
        """
        self.x = float(WINDOW_WIDTH // 2)
        self.y = float(
            WINDOW_HEIGHT - PADDLE_Y_OFFSET - PADDLE_HEIGHT - BALL_RADIUS - 1
        )
        sideways_options = [-0.6, -0.3, 0.3, 0.6]
        self.dx = random.choice(sideways_options) * BALL_SPEED
        self.dy = -BALL_SPEED

    def update(self, paddle):
        self.x += self.dx
        self.y += self.dy

        # Side walls: reflect instead of clamp.
        if self.x - BALL_RADIUS <= 0 or self.x + BALL_RADIUS >= WINDOW_WIDTH:
            self.dx *= -1

        # Ceiling.
        if self.y - BALL_RADIUS <= 0:
            self.dy *= -1

        # Paddle -- close-enough circle-vs-rect check using the ball's
        # bounding box, the same Rect-collision thinking as every
        # button so far, just checking the ball's box instead of the
        # mouse position.
        ball_rect = pygame.Rect(
            self.x - BALL_RADIUS,
            self.y - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )
        if self.dy > 0 and ball_rect.colliderect(paddle.rect):
            self.dy *= -1
            # Where the ball hit the paddle nudges its sideways speed,
            # so a return isn't identical every time -- catching it
            # near an edge sends it off at more of an angle.
            offset = (self.x - paddle.rect.centerx) / (PADDLE_WIDTH / 2)
            self.dx = offset * BALL_SPEED

        # Missed the paddle entirely: for now, just serve again.
        if self.y - BALL_RADIUS > WINDOW_HEIGHT:
            self.reset()

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)


paddle = Paddle()
ball = Ball()

running = True
while running:
    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- 2. Update state ------------------------------------------------------
    keys = pygame.key.get_pressed()
    paddle.update(keys)
    ball.update(paddle)

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)
    paddle.draw(screen)
    ball.draw(screen)

    hint_surface = font.render(
        "Arrow keys to move -- ball serves itself for now", True, TEXT_COLOR
    )
    screen.blit(hint_surface, (20, 20))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
