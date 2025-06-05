import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 400, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Control")

WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)

BUTTON_SIZE = 60
BUTTON_GAP = 50
UP_POS = (WIDTH // 2 - BUTTON_SIZE // 2, HEIGHT // 2 - BUTTON_SIZE - BUTTON_GAP)
DOWN_POS = (WIDTH // 2 - BUTTON_SIZE // 2, HEIGHT // 2 + BUTTON_GAP)
LEFT_POS = (WIDTH // 2 - BUTTON_SIZE - BUTTON_GAP, HEIGHT // 2)
RIGHT_POS = (WIDTH // 2 + BUTTON_GAP, HEIGHT // 2)

def forward():
    print("Moving forward")

def backward():
    print("Moving backward")

def left():
    print("Turning left")

def right():
    print("Turning right")

def draw_button(pos, text):
    rect = pygame.Rect(pos[0], pos[1], BUTTON_SIZE, BUTTON_SIZE)
    pygame.draw.rect(screen, GRAY, rect, border_radius=5)
    font = pygame.font.SysFont(None, 40)
    text_surf = font.render(text, True, BLACK)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)
    return rect

running = True
while running:
    screen.fill(WHITE)


    up_rect = draw_button(UP_POS, '↑')
    down_rect = draw_button(DOWN_POS, '↓')
    left_rect = draw_button(LEFT_POS, '←')
    right_rect = draw_button(RIGHT_POS, '→')

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if up_rect.collidepoint(event.pos):
                forward()
            elif down_rect.collidepoint(event.pos):
                backward()
            elif left_rect.collidepoint(event.pos):
                left()
            elif right_rect.collidepoint(event.pos):
                right()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                forward()
            elif event.key == pygame.K_DOWN:
                backward()
            elif event.key == pygame.K_LEFT:
                left()
            elif event.key == pygame.K_RIGHT:
                right()

    pygame.display.flip()

pygame.quit()
sys.exit()
