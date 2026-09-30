"""
Breakout -- Stage 2: Bricks
=============================

Everything from Stage 1 is still here, unchanged -- Paddle is a
byte-for-byte copy, and Ball keeps every bit of its wall/ceiling/paddle
bouncing exactly as it was. This stage adds one new idea on top of it:

    A GRID of small rectangles the ball can destroy.

Structurally a Brick is nothing new -- it's a Rect and a color, the
same shape as everything else in this file. What's new is building many
of them at once with a double loop (row, then column within the row)
and giving Ball one more thing to check each frame: "did I just hit any
of these rectangles?"

Brick collision uses one small trick worth calling out: when the ball's
box overlaps a brick's box, the SHORTER side of that overlap tells you
which way to bounce. A ball clipping a brick's corner from the side
overlaps a lot vertically but only a little horizontally -- so a small
horizontal overlap means "you hit a side, flip dx," and a small
vertical overlap means "you hit the top or bottom, flip dy." Same
"compare two numbers, branch on which is smaller" skill as every
conditional so far, just applied to overlap width/height instead of a
health value or a screen edge.

For now, clearing every brick just stops the ball and shows a message
-- bare-bones on purpose, the same way Session 4's `game_over` flag was
bare-bones before Session 5 built a proper state machine around it.
Stage 3 here does the same job: menu, lives, a real win/lose screen.

CUSTOMIZE ME: CONFIG has grown a "bricks" section -- row/column count,
sizing, and a color per row. Nothing below CONFIG needs to change to
reskin the grid.
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
PADDLE_SPEED = 7          # pixels moved per frame while a key is held
PADDLE_Y_OFFSET = 40      # distance from the bottom edge

BALL_RADIUS = 9
BALL_SPEED = 5            # total pixels moved per frame

BRICK_ROWS = 5
BRICK_COLS = 9
BRICK_WIDTH = 56
BRICK_HEIGHT = 20
BRICK_PADDING = 6         # gap between bricks, both directions
BRICK_TOP_OFFSET = 60     # distance from the top of the window to row 0

# One color per row, top to bottom. If BRICK_ROWS is longer than this
# list, colors repeat -- change these to reskin the whole grid.
BRICK_ROW_COLORS = [
    (235, 90, 90),
    (240, 150, 70),
    (235, 210, 70),
    (120, 200, 100),
    (100, 170, 235),
]

FPS = 60

# ---------------------------------------------------------------------------
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakout -- Stage 2: Bricks")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 48)


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
    """A rectangle with a color. No behaviour of its own -- Ball is the
    one that notices when it's been hit, the same way Cookie Clicker's
    floating-text particles never did anything but exist and get
    checked against by something else.
    """

    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, BRICK_WIDTH, BRICK_HEIGHT)
        self.color = color

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect, border_radius=3)


def build_bricks():
    """Lay out a full grid of bricks, centered horizontally in the
    window. Same "figure out the total size first, then center it"
    idea as fit_text() sizing a button label -- work out how wide the
    whole grid is before deciding where column 0 starts.
    """
    bricks = []
    grid_width = BRICK_COLS * (BRICK_WIDTH + BRICK_PADDING) - BRICK_PADDING
    left_margin = (WINDOW_WIDTH - grid_width) // 2

    for row in range(BRICK_ROWS):
        color = BRICK_ROW_COLORS[row % len(BRICK_ROW_COLORS)]
        y = BRICK_TOP_OFFSET + row * (BRICK_HEIGHT + BRICK_PADDING)
        for col in range(BRICK_COLS):
            x = left_margin + col * (BRICK_WIDTH + BRICK_PADDING)
            bricks.append(Brick(x, y, color))

    return bricks


class Ball:
    """Everything from Stage 1 is unchanged except update(), which now
    also takes the brick list and checks for a hit after checking the
    paddle.
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
        """The ball's current bounding box -- used for every collision
        check below, so it's pulled into its own method instead of
        being rebuilt with slightly different code three times.
        """
        return pygame.Rect(
            self.x - BALL_RADIUS,
            self.y - BALL_RADIUS,
            BALL_RADIUS * 2,
            BALL_RADIUS * 2,
        )

    def handle_brick_collision(self, bricks):
        """Check the ball against every remaining brick. At most one
        brick is removed per frame -- resolving just one collision at
        a time keeps the bounce direction unambiguous even if the ball
        is touching two bricks at once.
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
            break

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

        self.handle_brick_collision(bricks)

        if self.y - BALL_RADIUS > WINDOW_HEIGHT:
            self.reset()

    def draw(self, surface):
        pygame.draw.circle(surface, BALL_COLOR, (int(self.x), int(self.y)), BALL_RADIUS)


paddle = Paddle()
ball = Ball()
bricks = build_bricks()
cleared = False  # bare-bones win flag -- Stage 3 replaces this with real states

running = True
while running:
    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- 2. Update state ------------------------------------------------------
    keys = pygame.key.get_pressed()
    paddle.update(keys)

    if not cleared:
        ball.update(paddle, bricks)
        if not bricks:
            cleared = True

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)

    for brick in bricks:
        brick.draw(screen)

    paddle.draw(screen)
    ball.draw(screen)

    remaining_surface = font.render(f"Bricks remaining: {len(bricks)}", True, TEXT_COLOR)
    screen.blit(remaining_surface, (20, 20))

    if cleared:
        win_surface = big_font.render("BOARD CLEARED!", True, TEXT_COLOR)
        screen.blit(win_surface, win_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()