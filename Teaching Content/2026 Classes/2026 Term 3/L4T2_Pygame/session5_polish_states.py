"""
Session 5 -- Polishing Snake: Game Over, Restart, and Difficulty
====================================================================

Session 4 left Snake playable but rough: it ends abruptly with no message
and no way to play again without restarting the program. This session adds
the layer of polish that turns a working prototype into a real, replayable
game:

    1. a proper game-over screen (and a start menu)
    2. a restart key
    3. a difficulty curve that rewards a good run
"""

import random

import pygame

# ---------------------------------------------------------------------------
# Setup (same as Sessions 2-4)
# ---------------------------------------------------------------------------
pygame.init()

CELL_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Session 5 -- Polishing Snake")

clock = pygame.time.Clock()
FPS = 60

BACKGROUND_COLOR = (10, 10, 10)
GRID_LINE_COLOR = (40, 40, 40)
SNAKE_COLOR = (0, 200, 0)
HEAD_COLOR = (0, 255, 100)
FOOD_COLOR = (220, 60, 60)
TEXT_COLOR = (255, 255, 255)

font = pygame.font.SysFont(None, 28)
big_font = pygame.font.SysFont(None, 48)

STARTING_MOVE_INTERVAL = 150  # milliseconds per grid step at the start of a run
MIN_MOVE_INTERVAL = 60  # the fastest the snake is ever allowed to move
SPEEDUP_PER_FOOD = 5  # how many ms faster the snake gets per food eaten


def draw_grid(surface):
    """Draw faint grid lines so each cell is visible (from Session 3)."""
    for col in range(GRID_WIDTH + 1):
        x = col * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE_COLOR, (x, 0), (x, WINDOW_HEIGHT))

    for row in range(GRID_HEIGHT + 1):
        y = row * CELL_SIZE
        pygame.draw.line(surface, GRID_LINE_COLOR, (0, y), (WINDOW_WIDTH, y))


def random_empty_cell(occupied_cells):
    """Pick a random grid cell that isn't currently occupied by the snake."""
    while True:
        cell = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if cell not in occupied_cells:
            return cell


def hits_wall(head):
    """True if `head` has gone outside the grid's valid range."""
    x, y = head
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT


def hits_self(snake_body):
    """True if the head's position already appears elsewhere in the body."""
    return snake_body[0] in snake_body[1:]


def draw_centered_text(surface, text, font_obj, color, y_offset=0):
    """Render `text` centred horizontally (and vertically, plus an offset).

    Computes position from the rendered text surface's own size, rather
    than guessing pixel coordinates by hand -- the Session 5 technique
    for centred messages.
    """
    text_surface = font_obj.render(text, True, color)
    rect = text_surface.get_rect(
        center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + y_offset)
    )
    surface.blit(text_surface, rect)


# ---------------------------------------------------------------------------
# Game state variables
# ---------------------------------------------------------------------------
# game_state holds one of three values: "menu", "playing", "game_over".
# Routing the loop's update and draw steps through this single variable is
# the same "stored value decides which logic should run" pattern as the
# direction variable in Session 3 -- just with three options instead of
# four, and driving which BLOCK of code runs instead of which way a
# segment moves.
game_state = "menu"

snake = []
direction = (1, 0)
score = 0
food = (0, 0)
move_timer = 0
move_interval = STARTING_MOVE_INTERVAL


def reset_game():
    """Reinitialise every variable that changes during play.

    Resetting state cleanly means re-setting EVERY variable that changed
    while playing -- not just some of them. Forgetting one (e.g. leaving
    move_interval sped-up from the previous run) is exactly the kind of
    bug that lives at a state transition.
    """
    global snake, direction, score, food, move_timer, move_interval

    snake = [(10, 10), (9, 10), (8, 10)]
    direction = (1, 0)
    score = 0
    food = random_empty_cell(snake)
    move_timer = 0
    move_interval = STARTING_MOVE_INTERVAL


running = True

while running:
    # --- 1. Handle events ---------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if game_state == "menu":
                if event.key == pygame.K_SPACE:
                    reset_game()
                    game_state = "playing"

            elif game_state == "playing":
                if event.key == pygame.K_UP and direction != (0, 1):
                    direction = (0, -1)
                elif event.key == pygame.K_DOWN and direction != (0, -1):
                    direction = (0, 1)
                elif event.key == pygame.K_LEFT and direction != (1, 0):
                    direction = (-1, 0)
                elif event.key == pygame.K_RIGHT and direction != (-1, 0):
                    direction = (1, 0)

            elif game_state == "game_over" and event.key == pygame.K_r:
                reset_game()
                game_state = "playing"

    # --- 2. Update state ------------------------------------------------------
    # Only the "playing" state has anything to update each frame. menu and
    # game_over are static until a key press changes game_state above --
    # exactly the state-transition design students sketch out before coding.
    if game_state == "playing":
        move_timer += clock.get_time()

        if move_timer >= move_interval:
            move_timer = 0

            head_x, head_y = snake[0]
            new_head = (head_x + direction[0], head_y + direction[1])
            snake.insert(0, new_head)

            if new_head == food:
                score += 1
                food = random_empty_cell(snake)
                # Increasing speed as the score grows: shave a little off
                # the move interval every time food is eaten, with a
                # floor (MIN_MOVE_INTERVAL) so the game never becomes
                # literally unplayable.
                move_interval = max(MIN_MOVE_INTERVAL, move_interval - SPEEDUP_PER_FOOD)
            else:
                snake.pop()

            if hits_wall(new_head) or hits_self(snake):
                game_state = "game_over"

    # --- 3. Draw the frame ------------------------------------------------------
    screen.fill(BACKGROUND_COLOR)
    draw_grid(screen)

    if game_state == "menu":
        draw_centered_text(screen, "SNAKE", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(screen, "Press SPACE to start", font, TEXT_COLOR, y_offset=20)

    elif game_state == "playing":
        food_rect = (food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, FOOD_COLOR, food_rect)

        for index, segment in enumerate(snake):
            seg_x, seg_y = segment
            rect = (seg_x * CELL_SIZE, seg_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = HEAD_COLOR if index == 0 else SNAKE_COLOR
            pygame.draw.rect(screen, color, rect)

        score_surface = font.render(f"Score: {score}", True, TEXT_COLOR)
        screen.blit(score_surface, (10, 10))

    elif game_state == "game_over":
        # Still show the final board underneath, same as Session 4, but
        # now with a proper centred message instead of a bare corner label.
        for index, segment in enumerate(snake):
            seg_x, seg_y = segment
            rect = (seg_x * CELL_SIZE, seg_y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = HEAD_COLOR if index == 0 else SNAKE_COLOR
            pygame.draw.rect(screen, color, rect)

        draw_centered_text(screen, "GAME OVER", big_font, TEXT_COLOR, y_offset=-30)
        draw_centered_text(
            screen,
            f"Score: {score}  --  Press R to restart",
            font,
            TEXT_COLOR,
            y_offset=20,
        )

    pygame.display.flip()
    clock.tick(FPS)


pygame.quit()
