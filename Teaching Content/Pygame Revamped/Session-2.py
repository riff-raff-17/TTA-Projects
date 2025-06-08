#!/usr/bin/env python3
import sys
import pygame

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Shapes, Surfaces & Sprites")
clock = pygame.time.Clock()
FPS = 60

# Optional Background Color
# To change the background color, uncomment the line below and adjust the RGB values.
# BG_COLOR = (50, 50, 100)

# Load and scale sprite image
SPRITE_MAX_SIZE = (100, 100)  # max width, max height
sprite_surf = pygame.image.load("sprite.png").convert_alpha()
orig_w, orig_h = sprite_surf.get_size()
max_w, max_h = SPRITE_MAX_SIZE
scale = min(max_w / orig_w, max_h / orig_h)
new_size = (int(orig_w * scale), int(orig_h * scale))
sprite_surf = pygame.transform.smoothscale(sprite_surf, new_size)
sprite_rect = sprite_surf.get_rect(topleft=(50, 50))

# Create an off-screen Surface for shapes
shape_surf = pygame.Surface((200, 200), pygame.SRCALPHA)
shape_surf.fill((0, 0, 0, 0))  # fully transparent

# Draw a rectangle: (surface, color, Rect(left, top, width, height))
pygame.draw.rect(
    shape_surf,
    (200, 50, 50),
    pygame.Rect(10, 10, 80, 60)
)

# Draw a circle: (surface, color, center_pos, radius)
pygame.draw.circle(
    shape_surf,
    (50, 200, 50),
    (150, 50),
    30
)

# Draw a line: (surface, color, start_pos, end_pos, width)
pygame.draw.line(
    shape_surf,
    (50, 50, 200),
    (0, 199),
    (199, 0),
    4
)

shape_rect = shape_surf.get_rect(topright=(WIDTH - 50, 50))

# Movement vector (pixels per second)
speed = pygame.math.Vector2(120, 80)

running = True
while running:
    dt = clock.tick(FPS) / 1000.0

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update sprite position and bounce off edges
    sprite_rect.x += speed.x * dt
    sprite_rect.y += speed.y * dt
    if sprite_rect.left < 0 or sprite_rect.right > WIDTH:
        speed.x *= -1
    if sprite_rect.top < 0 or sprite_rect.bottom > HEIGHT:
        speed.y *= -1

    # Rendering (clear -> shapes -> line -> sprite)
    # Default background fill:
    screen.fill((30, 30, 30))
    # To use custom background color instead, uncomment the next line:
    # screen.fill(BG_COLOR)

    screen.blit(shape_surf, shape_rect)
    pygame.draw.line(
        screen,
        (255, 255, 0),
        (0, HEIGHT - 50),
        (WIDTH, HEIGHT - 50),
        2
    )
    screen.blit(sprite_surf, sprite_rect)
    pygame.display.flip()

pygame.quit()
sys.exit()
