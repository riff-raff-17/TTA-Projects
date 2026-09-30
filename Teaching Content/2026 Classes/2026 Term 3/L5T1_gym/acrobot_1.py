import pygame

# --- Setup ---
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Acrobot")

clock = pygame.time.Clock()
FPS = 60

WHITE = (255, 255, 255)

# --- Main loop ---
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw
    screen.fill(WHITE)
    pygame.display.flip()

    # Cap framerate
    clock.tick(FPS)

pygame.quit()