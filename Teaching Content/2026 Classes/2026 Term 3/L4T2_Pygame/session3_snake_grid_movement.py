"""
Session 3 -- Starting Snake: Grid Movement and the Snake's Body
=================================================================

What makes Snake different from the free-roaming circle of Session 2:

    1. Movement happens on a fixed GRID, one cell at a time -- not smooth
       pixel-by-pixel motion.
    2. The snake is not one shape -- it's a growing LIST of positions that
       all move together.

This script gets a multi-segment snake moving correctly around the grid,
turning on arrow-key presses, and refusing to reverse into itself. No
food or collisions yet -- that's Session 4. The goal here is just to get
movement rock solid before adding anything else (build-and-test-one-piece
-at-a-time, same habit as Session 1 and 2).
"""

import pygame

# ---------------------------------------------------------------------------
# Setup (same boilerplate as Session 2)
# ---------------------------------------------------------------------------
pygame.init()

# Designing a grid on top of pixel coordinates.
# We pick a cell size, then convert between grid coords (column, row) and
# pixel coords (column * CELL_SIZE, row * CELL_SIZE) only at draw time.
# Deciding this up front avoids mixing the two systems throughout the
# movement logic.
CELL_SIZE = 20
GRID_WIDTH = 30  # cells across
GRID_HEIGHT = 20  # cells down

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Session 3 -- Snake Grid Movement")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (10, 10, 10)
GRID_LINE_COLOR = (40, 40, 40)
SNAKE_COLOR = (0, 200, 0)
HEAD_COLOR = (0, 255, 100)


def draw_grid(surface):
    """Draw faint grid lines so each cell is visible."""
    for col in range(GRID_WIDTH + 1):
        x = col * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE_COLOR, (x, 0), (x, WINDOW_HEIGHT))

    for row in range(GRID_HEIGHT + 1):
        y = row * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE_COLOR, (0, y), (WINDOW_WIDTH, y))


# ---------------------------------------------------------------------------
# Representing the snake as a list of segments
# ---------------------------------------------------------------------------
# Each segment is an (x, y) GRID-coordinate tuple (not pixels). The first
# element is the head.
snake = [(10, 10), (9, 10), (8, 10)]  # head first

# Choosing a direction with a single variable.
# We store direction once and update it only on a key press -- NOT by
# reading the keyboard fresh every frame. Reading fresh every frame would
# let the snake reverse into itself the instant you tapped the opposite key.
direction = (1, 0)  # (dx, dy) -- moving right


# ---------------------------------------------------------------------------
# Decoupling movement speed from the frame rate
# ---------------------------------------------------------------------------
# The screen redraws at 60 FPS, but the snake should only advance one grid
# cell every MOVE_INTERVAL milliseconds. Without this separate timer, the
# snake would move 60 cells per second -- far too fast to see, let alone
# play. This builds on the Session 2 idea of the frame as the unit of time,
# and why grid-based movement needs its own pacing.
move_timer = 0
MOVE_INTERVAL = 150  # milliseconds per grid step

running = True

while running:
    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            # Preventing an immediate reversal.
            # Each check rejects the new direction if it's the exact
            # opposite of the current one, so you can't turn left
            # while already moving right.
            if event.key == pygame.K_UP and direction != (0, 1):
                direction = (0, -1)
            elif event.key == pygame.K_DOWN and direction != (0, -1):
                direction = (0, 1)
            elif event.key == pygame.K_LEFT and direction != (1, 0):
                direction = (-1, 0)
            elif event.key == pygame.K_RIGHT and direction != (-1, 0):
                direction = (1, 0)

    # --- 2. Update state ------------------------------------------------------
    # Advance the movement timer by however many milliseconds the last
    # frame took (clock.get_time() reports that), rather than counting
    # frames -- this keeps movement speed consistent even if the frame
    # rate fluctuates.
    move_timer += clock.get_time()

    if move_timer >= MOVE_INTERVAL:
        move_timer = 0

        # Moving the whole snake on a fixed tick
        # "Moving" a multi-segment object is really "add a new head,
        # drop the tail" -- not shifting every segment individually.
        # The head is treated as special; the rest of the body follows.
        head_x, head_y = snake[0]
        new_head = (head_x + direction[0], head_y + direction[1])

        snake.insert(0, new_head)
        snake.pop()

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)
    draw_grid(screen)

    for index, segment in enumerate(snake):
        seg_x, seg_y = segment
        rect = (seg_x * CELL_SIZE, seg_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = HEAD_COLOR if index == 0 else SNAKE_COLOR
        pygame.draw.rect(screen, color, rect)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
