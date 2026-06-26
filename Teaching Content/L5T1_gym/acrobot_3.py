import pygame
import math

# --- Setup ---
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Acrobot")

clock = pygame.time.Clock()
FPS = 60

WHITE = (255, 255, 255)
LINK_TEAL = (92, 201, 202)
JOINT_YELLOW = (204, 204, 66)
GOAL_GRAY = (96, 96, 96)

# --- Acrobot geometry ---
PIVOT = (WIDTH // 2, HEIGHT // 3)   # fixed point the whole system hangs from
LINK_LENGTH = 120                    # pixels, same for both links
LINK_WIDTH = 14                      # thickness of each link
JOINT_RADIUS = 8
GOAL_Y = PIVOT[1] - LINK_LENGTH      # goal line: one link-length above the pivot

# Starting angles (radians). theta1 from straight down, ccw positive.
# theta2 is relative to link 1 (the "elbow bend").
# These are now updated every frame below - just oscillating with sine waves
# for now (no real physics yet), to test that the render loop handles
# continuously changing angles smoothly.
theta1 = math.radians(30)
theta2 = math.radians(45)

elapsed_time = 0.0   # seconds since start, drives the test animation


def get_joint_positions(theta1, theta2):
    """Compute pixel positions of the elbow and free end from the two angles."""
    # theta measured from straight down; pygame y-axis points down,
    # so "down" is +y and we rotate from there.
    x0, y0 = PIVOT

    # Elbow joint (end of link 1)
    x1 = x0 + LINK_LENGTH * math.sin(theta1)
    y1 = y0 + LINK_LENGTH * math.cos(theta1)

    # Free end (end of link 2), angle is theta1 + theta2 in world frame
    x2 = x1 + LINK_LENGTH * math.sin(theta1 + theta2)
    y2 = y1 + LINK_LENGTH * math.cos(theta1 + theta2)

    return (x0, y0), (x1, y1), (x2, y2)


def draw_acrobot(surface, theta1, theta2):
    pivot, elbow, tip = get_joint_positions(theta1, theta2)

    pygame.draw.line(surface, LINK_TEAL, pivot, elbow, LINK_WIDTH)
    pygame.draw.line(surface, LINK_TEAL, elbow, tip, LINK_WIDTH)

    pygame.draw.circle(surface, JOINT_YELLOW, (int(pivot[0]), int(pivot[1])), JOINT_RADIUS)
    pygame.draw.circle(surface, JOINT_YELLOW, (int(elbow[0]), int(elbow[1])), JOINT_RADIUS)


# --- Main loop ---
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update angles (test animation only - real physics comes in Step 4)
    theta1 = math.radians(60) * math.sin(elapsed_time)
    theta2 = math.radians(90) * math.sin(elapsed_time * 1.7)

    # Draw
    screen.fill(WHITE)
    pygame.draw.line(screen, GOAL_GRAY, (0, GOAL_Y), (WIDTH, GOAL_Y), 2)
    draw_acrobot(screen, theta1, theta2)
    pygame.display.flip()

    # Cap framerate, advance our own clock by how long this frame took
    dt_ms = clock.tick(FPS)
    elapsed_time += dt_ms / 1000.0

pygame.quit()