import sys
import pygame

def main():
    # Initialize Pygame
    pygame.init()

    # Window settings
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("My Pygame Window")

    # Frame rate control
    FPS = 60
    clock = pygame.time.Clock()

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update game state here 
        # (e.g., move sprites, check collisions, etc.)

        # Rendering
        screen.fill((30, 30, 30))   # clear screen to dark gray
        # (draw the game objects here)
        
        pygame.display.flip()       # update the display

        # Maintain target frame rate
        clock.tick(FPS)

    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
