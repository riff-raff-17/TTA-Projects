import pygame

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Draw Name Example")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Set up font (size 72)
font = pygame.font.SysFont("arial", 72)

# Render the text
text_surface = font.render("Caden", True, BLUE)  
# Parameters:
# 1. The text string
# 2. Anti-aliasing (True = smooth edges, False = jagged pixels)
# 3. Text color
# 4. (Optional) background color

# Get rectangle of the text to center it
text_rect = text_surface.get_rect(center=(400, 300))

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill screen
    screen.fill(WHITE)

    # Draw the text on screen
    screen.blit(text_surface, text_rect)

    # Update display
    pygame.display.flip()

pygame.quit()
