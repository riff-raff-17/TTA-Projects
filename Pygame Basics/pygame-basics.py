import pygame
import random
import sys

# --- Constants ---
WIDTH, HEIGHT = 480, 600
FPS = 60
PLAYER_SPEED = 5
BLOCK_SPEED_START = 4
BLOCK_ACCELERATION = 0.005  # how much faster blocks get each frame
BLOCK_SPAWN_RATE = 30       # frames between spawns

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dodge the Falling Blocks")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 36)

# --- Sprite Classes ---
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 30))
        self.image.fill((50, 200, 50))
        self.rect = self.image.get_rect(midbottom=(WIDTH // 2, HEIGHT - 10))

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.rect.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            self.rect.x += PLAYER_SPEED
        # keep on screen
        self.rect.x = max(0, min(self.rect.x, WIDTH - self.rect.width))


class Block(pygame.sprite.Sprite):
    def __init__(self, speed):
        super().__init__()
        size = random.randint(20, 50)
        self.image = pygame.Surface((size, size))
        self.image.fill((200, 50, 50))
        self.rect = self.image.get_rect(
            x=random.randrange(0, WIDTH - size),
            y=-size
        )
        self.speed = speed

    def update(self):
        self.rect.y += self.speed
        if self.rect.top > HEIGHT:
            self.kill()


# --- Groups & Game State ---
all_sprites = pygame.sprite.Group()
blocks = pygame.sprite.Group()
player = Player()
all_sprites.add(player)

block_speed = BLOCK_SPEED_START
frame_count = 0
start_ticks = pygame.time.get_ticks()  # to calculate survival time

# --- Main Game Loop ---
running = True
while running:
    dt = clock.tick(FPS)
    frame_count += 1
    block_speed += BLOCK_ACCELERATION  # gradually speed up

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # --- Spawn Blocks ---
    if frame_count % BLOCK_SPAWN_RATE == 0:
        b = Block(block_speed)
        all_sprites.add(b)
        blocks.add(b)

    # --- Updates ---
    all_sprites.update()

    # --- Collision Check ---
    if pygame.sprite.spritecollideany(player, blocks):
        running = False  # end game on hit

    # --- Draw ---
    screen.fill((30, 30, 30))
    all_sprites.draw(screen)
    elapsed_sec = (pygame.time.get_ticks() - start_ticks) // 1000
    score_surf = font.render(f"Time: {elapsed_sec}s", True, (255, 255, 255))
    screen.blit(score_surf, (10, 10))
    pygame.display.flip()

# --- Game Over Screen ---
game_over_surf = font.render(f"Game Over! You survived {elapsed_sec}s", True, (255, 200, 0))
rect = game_over_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
screen.blit(game_over_surf, rect)
pygame.display.flip()

# wait for a keypress or quit
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type in (pygame.QUIT, pygame.KEYDOWN):
            waiting = False

pygame.quit()
