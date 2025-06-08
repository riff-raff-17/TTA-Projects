import sys
import random
import pygame

# Settings
WIDTH, HEIGHT = 800, 600
FPS = 60
INITIAL_ITEMS = 5
ITEM_RESPAWN_COUNT = 5
NUM_BOUNCERS = 5

MENU, PLAYING, GAME_OVER = "MENU", "PLAYING", "GAME_OVER"

# Asset Loading
def load_assets():
    pygame.mixer.init()
    pygame.mixer.music.load("background.mp3")
    pygame.mixer.music.set_volume(0.5)
    return {
        'collect_sfx': pygame.mixer.Sound("collect.wav"),
        'font':        pygame.font.SysFont(None, 36)
    }

# Sprite Classes
class Player(pygame.sprite.Sprite):
    def __init__(self, pos, obstacles):
        super().__init__()
        self.image = pygame.Surface((50,50))
        self.image.fill((0,128,255))
        self.rect = self.image.get_rect(center=pos)
        self.speed = 250
        self.obstacles = obstacles

    def update(self, dt):
        old = self.rect.copy()
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * self.speed * dt
        dy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])   * self.speed * dt
        self.rect.x += dx
        self.rect.y += dy
        self.rect.clamp_ip(pygame.Rect(0,0,WIDTH,HEIGHT))
        for obs in self.obstacles:
            if self.rect.colliderect(obs):
                self.rect = old
                break

class Item(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((30,30), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (50,200,50), (15,15), 15)
        self.rect = self.image.get_rect(center=pos)

class Chaser(pygame.sprite.Sprite):
    COLOR = (255,50,50)  # red
    def __init__(self, pos):
        super().__init__()
        self.image = pygame.Surface((30,30))
        self.image.fill(self.COLOR)
        self.rect = self.image.get_rect(center=pos)
        self.speed = 120
    def update(self, dt, player):
        dir_vec = pygame.math.Vector2(player.rect.center) - self.rect.center
        if dir_vec.length():
            dir_vec = dir_vec.normalize()
        move = dir_vec * self.speed * dt
        self.rect.x += move.x
        self.rect.y += move.y

class Bouncer(pygame.sprite.Sprite):
    COLOR = (255,200,0)  # yellow
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((25,25))
        self.image.fill(self.COLOR)
        x = random.randint(50, WIDTH-50)
        y = random.randint(50, HEIGHT-50)
        self.rect = self.image.get_rect(center=(x,y))
        angle = random.uniform(0, 2*3.14159)
        speed = random.uniform(80,160)
        self.velocity = pygame.math.Vector2(speed, 0).rotate_rad(angle)
    def update(self, dt):
        self.rect.x += self.velocity.x * dt
        self.rect.y += self.velocity.y * dt
        # bounce off screen edges
        if self.rect.left < 0 or self.rect.right > WIDTH:
            self.velocity.x *= -1
        if self.rect.top < 0 or self.rect.bottom > HEIGHT:
            self.velocity.y *= -1


# State Handlers
def show_menu(screen, font):
    screen.fill((20,20,20))
    screen.blit(font.render("CHASE & COLLECT", True, (255,255,255)),
                (WIDTH//2-140, HEIGHT//2-40))
    screen.blit(font.render("ENTER to start", True, (200,200,200)),
                (WIDTH//2-100, HEIGHT//2+10))
    pygame.display.flip()

def show_game_over(screen, font, score):
    screen.fill((20,20,20))
    screen.blit(font.render("GAME OVER", True, (255,50,50)),
                (WIDTH//2-80, HEIGHT//2-60))
    screen.blit(font.render(f"Score: {score}", True, (255,255,255)),
                (WIDTH//2-70, HEIGHT//2))
    screen.blit(font.render("ENTER=replay   ESC=quit", True, (200,200,200)),
                (WIDTH//2-140, HEIGHT//2+60))
    pygame.display.flip()


# Main Loop

def run_game(screen, clock, assets):
    pygame.mixer.music.play(-1)
    # static obstacles
    obstacles = [
        pygame.Rect(150,150,100,20),
        pygame.Rect(350,300,20,150),
        pygame.Rect(500,100,150,20),
        pygame.Rect(600,400,20,120),
    ]
    player = Player((WIDTH//2,HEIGHT//2), obstacles)

    items = pygame.sprite.Group(
        *(Item((random.randint(50,WIDTH-50), random.randint(50,HEIGHT-50)))
          for _ in range(INITIAL_ITEMS))
    )

    # one chaser, from a corner
    chaser = Chaser((50,50))
    # five bouncing enemies
    bouncers = pygame.sprite.Group(Bouncer() for _ in range(NUM_BOUNCERS))

    score = 0
    next_spawn = INITIAL_ITEMS

    while True:
        dt = clock.tick(FPS) / 1000.0

        # events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.mixer.music.stop()
                return GAME_OVER, score

        # updates
        player.update(dt)
        chaser.update(dt, player)
        for b in bouncers:
            b.update(dt)

        # collect items
        hits = pygame.sprite.spritecollide(player, items, dokill=True)
        for _ in hits:
            assets['collect_sfx'].play()
            score += 1
        if score >= next_spawn:
            for _ in range(ITEM_RESPAWN_COUNT):
                items.add(Item((random.randint(50,WIDTH-50),
                                random.randint(50,HEIGHT-50))))
            next_spawn += ITEM_RESPAWN_COUNT

        # collisions mean game over
        if pygame.sprite.spritecollide(player, bouncers, dokill=False) \
        or player.rect.colliderect(chaser.rect):
            pygame.mixer.music.stop()
            return GAME_OVER, score

        # draw
        screen.fill((30,30,30))
        for obs in obstacles:
            pygame.draw.rect(screen, (100,100,100), obs)
        items.draw(screen)
        bouncers.draw(screen)
        screen.blit(chaser.image, chaser.rect)
        screen.blit(player.image, player.rect)

        # HUD
        score_surf = assets['font'].render(f"Score: {score}", True, (255,255,255))
        screen.blit(score_surf, (10,10))
        pygame.display.flip()


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption("Chaser & Bouncers")
    clock = pygame.time.Clock()
    assets = load_assets()
    state, score = MENU, 0

    while True:
        if state == MENU:
            show_menu(screen, assets['font'])
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_RETURN:
                    state = PLAYING

        elif state == PLAYING:
            state, score = run_game(screen, clock, assets)

        elif state == GAME_OVER:
            show_game_over(screen, assets['font'], score)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_RETURN:
                        state, score = MENU, 0
                    elif ev.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()
