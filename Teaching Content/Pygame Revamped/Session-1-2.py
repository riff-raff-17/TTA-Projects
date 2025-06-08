import sys
import pygame

# Constants
WIDTH, HEIGHT   = 800, 600
FPS             = 60
BG_COLOR        = (30, 30, 30)
SHAPE_COLOR     = (200, 200,  60)
TEXT_COLOR      = (255, 255, 255)

# Initialization
def init():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Extended Pygame Window")
    clock  = pygame.time.Clock()
    # Create a simple Font object (None = default font, size 24)
    font   = pygame.font.SysFont(None, 24)
    return screen, clock, font

# Event Handling
def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            # Close on Escape key, too
            if event.key == pygame.K_ESCAPE:
                return False
    return True

# Game Update
def update(x_pos, dt):
    # Move the square at 120 pixels per second to the right
    x_pos += 120 * dt
    # Wrap around when it leaves the screen
    if x_pos > WIDTH:
        x_pos = -50
    return x_pos

# Rendering
def draw(screen, font, x_pos, clock):
    screen.fill(BG_COLOR)

    # Draw a moving square
    square_rect = pygame.Rect(int(x_pos), HEIGHT//2 - 25, 50, 50)
    pygame.draw.rect(screen, SHAPE_COLOR, square_rect)

    # Render the FPS in the top-left corner
    fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, TEXT_COLOR)
    screen.blit(fps_text, (10, 10))

    # Flip the back buffer to the display
    pygame.display.flip()

# Main Loop
def main():
    screen, clock, font = init()
    running = True
    x_pos   = 0.0

    while running:
        # dt = seconds elapsed since last tick
        dt = clock.tick(FPS) / 1000.0

        running = handle_events()
        x_pos   = update(x_pos, dt)
        draw(screen, font, x_pos, clock)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
