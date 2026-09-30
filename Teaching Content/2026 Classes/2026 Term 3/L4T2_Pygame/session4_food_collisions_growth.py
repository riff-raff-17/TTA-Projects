"""
Session 4 -- Food, Collisions, and Growth
============================================

With a moving snake in place (Session 3), this session adds the mechanics
that turn it into an actual game:

    1. something to eat (food, spawned at a random grid cell)
    2. ways to lose (hitting a wall or the snake's own body)
    3. a score that tracks progress

Every new check here is just a careful comparison -- same skill as
Session 1's conditionals, repeated for each kind of overlap that matters:
has the head reached the food? hit a wall? hit itself?
"""

import random
import pygame

# --- Setup ---
pygame.init()

# Designing a grid on top of pixel coordinates
CELL_SIZE = 20  # pixels
GRID_WIDTH = 30  # cells across
GRID_HEIGHT = 20  # cells down

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Session 4 -- Food, Collisions, and Growth")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (10, 10, 10)
GRID_LINE_COLOR = (40, 40, 40)
SNAKE_COLOR = (0, 200, 0)
HEAD_COLOR = (0, 255, 100)
FOOD_COLOR = (220, 60, 60)
TEXT_COLOR = (255, 255, 255)

font = pygame.font.SysFont(None, 28)


def draw_grid(surface):
    """Draw faint grid lines so each cell is visible."""
    for col in range(GRID_WIDTH + 1):
        x = col * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE_COLOR, (x, 0), (x, WINDOW_HEIGHT))

    for row in range(GRID_HEIGHT + 1):
        y = row * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE_COLOR, (0, y), (WINDOW_WIDTH, y))


def random_empty_cell(occupied_cells):
    """Pick a random grid cell that isn't currently occupied by the snake.

    Bundling this into its own function (Session 1's "functions organise
    repeated actions" idea) means both the initial food spawn and every
    later respawn-after-eating can call the same logic.
    """
    while True:
        cell = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if cell not in occupied_cells:
            return cell


def hits_wall(head):
    """True if the head has gone outside the grid's valid range.

    Same bounds-checking comparison from Session 1's clamp() discussion --
    but here an out-of-bounds value ends the game rather than being
    clamped back in.
    """
    x, y = head
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT


def hits_self(snake_body):
    """True if the head's position already appears elsewhere in the body.

    Checking membership in a list (`in`) is a natural way to ask "has this
    position been visited by my own body."
    """
    return snake_body[0] in snake_body[1:]


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
snake = [(10, 10), (9, 10), (8, 10)]  # head first

# Choosing a direction with a single variable.
direction = (1, 0)  # (dx, dy) -- moving right

# --- Decoupling movement speed from the frame rate ---
move_timer = 0
MOVE_INTERVAL = 150  # milliseconds per grid step

score = 0
food = random_empty_cell(snake)

# A simple "is the game still going" flag. A proper game-over SCREEN with
# a restart key is Session 5's job -- for now we just stop updating once
# this is False, which is enough to decide what "game over" means before
# building the polished version of it.
game_over = False

running = True

while running:
    # --- 1. Handle events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and direction != (0, 1):
                direction = (0, -1)
            elif event.key == pygame.K_DOWN and direction != (0, -1):
                direction = (0, 1)
            elif event.key == pygame.K_LEFT and direction != (1, 0):
                direction = (-1, 0)
            elif event.key == pygame.K_RIGHT and direction != (-1, 0):
                direction = (1, 0)

    # --- 2. Update state ------------------------------------------------------
    if not game_over:
        move_timer += clock.get_time()

        if move_timer >= MOVE_INTERVAL:
            move_timer = 0

            head_x, head_y = snake[0]
            new_head = (head_x + direction[0], head_y + direction[1])
            snake.insert(0, new_head)

            # Checking collisions in a sensible order: food first.
            # Eating food and continuing to play matters more immediately
            # than checking for a loss that hasn't happened yet.
            if new_head == food:
                score += 1
                food = random_empty_cell(snake)
                # No pop() this step -- skipping it is what makes the
                # snake grow. This is the direct payoff of choosing a
                # list to represent the snake back in Session 3.
            else:
                snake.pop()

            # Wall and self checks come after the food check, and are
            # each their own small, clearly named function rather than
            # one large tangled condition.
            if hits_wall(new_head) or hits_self(snake):
                game_over = True

    # --- 3. Draw the frame ---
    screen.fill(BACKGROUND_COLOR)
    draw_grid(screen)

    # Food
    food_rect = (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, FOOD_COLOR, food_rect)

    # Snake
    for index, segment in enumerate(snake):
        seg_x, seg_y = segment
        rect = (seg_x * CELL_SIZE, seg_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        color = HEAD_COLOR if index == 0 else SNAKE_COLOR
        pygame.draw.rect(screen, color, rect)

    # Score -- extends the Session 2 drawing skills to render text for the
    # first time, and doubles as a debugging signal: a live, visible
    # number confirms food collision is actually being detected reliably.
    score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
    screen.blit(score_surface, (10, 10))

    if game_over:
        # Deliberately bare-bones -- Session 5 builds the real game-over
        # screen (centred message, restart key, etc.). For now this just
        # confirms the loss condition is firing correctly.
        over_surface = font.render("GAME OVER", True, TEXT_COLOR)
        screen.blit(over_surface, (WINDOW_WIDTH // 2 - 50, WINDOW_HEIGHT // 2 - 14))

    pygame.display.flip()
    clock.tick(FPS)


pygame.quit()
