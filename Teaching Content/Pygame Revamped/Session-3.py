import sys
import random
import pygame

# Initialization
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sprites & Collision Demo")
clock = pygame.time.Clock()
FPS = 60

# Sprite Classes
class Player(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        # create a 50×50 blue square
        self.image = pygame.Surface((50, 50))
        self.image.fill((0, 128, 255))
        self.rect = self.image.get_rect(center=pos)

    def update(self, dt):
        # keyboard movement (arrow keys)
        keys = pygame.key.get_pressed()
        speed = 200  # pixels per second
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * speed * dt
        dy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])   * speed * dt
        self.rect.x += dx
        self.rect.y += dy
        # keep inside screen
        self.rect.clamp_ip(screen.get_rect())

class Enemy(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        # create a 30×30 red square
        self.image = pygame.Surface((30, 30))
        self.image.fill((255,  50,  50))
        self.rect = self.image.get_rect(center=pos)
        # random velocity vector
        self.velocity = pygame.math.Vector2(
            random.choice([-1, 1]) * random.uniform(50, 150),
            random.choice([-1, 1]) * random.uniform(50, 150)
        )

    def update(self, dt):
        # move and bounce off walls
        self.rect.x += self.velocity.x * dt
        self.rect.y += self.velocity.y * dt
        if self.rect.left < 0 or self.rect.right > WIDTH:
            self.velocity.x *= -1
        if self.rect.top  < 0 or self.rect.bottom > HEIGHT:
            self.velocity.y *= -1

# Sprite Groups
all_sprites = pygame.sprite.Group()
enemies    = pygame.sprite.Group()

player = Player((WIDTH // 2, HEIGHT // 2))
all_sprites.add(player)

# Main Loop
running = True
while running:
    dt = clock.tick(FPS) / 1000.0  # seconds since last frame

    # -- Event handling --
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # spawn enemy where you click
        elif event.type == pygame.MOUSEBUTTONDOWN:
            enemy = Enemy(event.pos)
            all_sprites.add(enemy)
            enemies.add(enemy)

        # press SPACE to spawn enemy at player
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                enemy = Enemy(player.rect.center)
                all_sprites.add(enemy)
                enemies.add(enemy)

    # Update
    for sprite in all_sprites:
        sprite.update(dt)

    # Collision detection
    # remove any enemy touching the player
    hits = pygame.sprite.spritecollide(
        player, enemies, True
    )
    if hits:
        print(f"Collision! {len(hits)} enemy removed.")

    # -- Draw --
    screen.fill((30, 30, 30))
    all_sprites.draw(screen)
    pygame.display.flip()

pygame.quit()
sys.exit()
