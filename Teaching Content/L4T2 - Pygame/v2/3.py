import pygame
import math
import random

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Every Other Frame Colors")

# Function to get a random color
def random_color():
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255))

# Initial colors (so we don’t start blank)
rect_color = random_color()
circle_color = random_color()
poly_color = random_color()
ellipse_color = random_color()
arc_color = random_color()
line_color = random_color()
polyline_color = random_color()

# Game loop
running = True
clock = pygame.time.Clock()
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update colors only every other frame
    if frame_count % 2 == 0:
        rect_color = random_color()
        circle_color = random_color()
        poly_color = random_color()
        ellipse_color = random_color()
        arc_color = random_color()
        line_color = random_color()
        polyline_color = random_color()
    
    # Fill screen with white
    screen.fill((255, 255, 255))
    
    # Rectangle
    pygame.draw.rect(screen, rect_color, (100, 100, 200, 100), 0)
    
    # Circle
    pygame.draw.circle(screen, circle_color, (400, 300), 50, 0)
    
    # Polygon (triangle here)
    pygame.draw.polygon(screen, poly_color, [(600, 100), (700, 200), (500, 200)], 0)
    
    # Ellipse
    pygame.draw.ellipse(screen, ellipse_color, (100, 300, 200, 100), 0)
    
    # Arc
    pygame.draw.arc(screen, arc_color, (350, 400, 200, 100), 0, math.pi, 3)
    
    # Single line
    pygame.draw.line(screen, line_color, (50, 500), (200, 550), 5)
    
    # Multiple connected lines
    pygame.draw.lines(screen, polyline_color, False, [(300, 500), (400, 550), (500, 520), (600, 580)], 3)

    # Update display
    pygame.display.flip()
    clock.tick(30)  # 30 FPS
    frame_count += 1

pygame.quit()
