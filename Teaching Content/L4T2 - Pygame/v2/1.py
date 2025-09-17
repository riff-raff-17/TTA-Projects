import pygame
import math

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Hello, Pygame!")

# Colors
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Fill screen with white
    screen.fill(WHITE)
    
    # Rectangle
    pygame.draw.rect(screen, RED, (100, 100, 200, 100), 0)
    
    # Circle
    pygame.draw.circle(screen, BLUE, (400, 300), 50, 0)
    
    # Polygon (triangle here)
    pygame.draw.polygon(screen, GREEN, [(600, 100), (700, 200), (500, 200)], 0)
    
    # Ellipse (drawn inside a bounding rectangle)
    pygame.draw.ellipse(screen, ORANGE, (100, 300, 200, 100), 0)
    
    # Arc (part of an ellipse) — angles in radians
    pygame.draw.arc(screen, BLACK, (350, 400, 200, 100), 0, math.pi, 3)  # half-circle arc
    
    # Single line
    pygame.draw.line(screen, BLACK, (50, 500), (200, 550), 5)
    
    # Multiple connected lines (polyline)
    pygame.draw.lines(screen, BLUE, False, [(300, 500), (400, 550), (500, 520), (600, 580)], 3)

    # Update display
    pygame.display.flip()

pygame.quit()
