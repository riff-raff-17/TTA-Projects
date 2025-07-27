import sys
import random
import pygame


# Game Settings & Constants

WIDTH, HEIGHT = 800, 600
FPS = 60
ITEM_COUNT = 5

# Define game states
MENU, PLAYING, GAME_OVER = "MENU", "PLAYING", "GAME_OVER"


# Asset Loading

def load_assets():
    """Load images, sounds, music, fonts."""
    assets = {}
    # initialize mixer for sounds
    pygame.mixer.init()
    # background music (loops indefinitely)
    pygame.mixer.music.load("background.mp3")
    pygame.mixer.music.set_volume(0.5)
    # sound effect for collecting an item
    assets['collect_sfx'] = pygame.mixer.Sound("collect.wav")
    # font for UI text
    assets['font'] = pygame.font.SysFont(None, 36)
    return assets


# Sprite Classes

class Player(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        # blue square player
        self.image = pygame.Surface((50, 50))
        self.image.fill((0, 128, 255))
        self.rect = self.image.get_rect(center=pos)
        self.speed = 250  # px/sec

    def update(self, dt):
        keys = pygame.key.get_pressed()
        dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * self.speed * dt
        dy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])   * self.speed * dt
        self.rect.x += dx
        self.rect.y += dy
        # keep inside window
        self.rect.clamp_ip(pygame.Rect(0, 0, WIDTH, HEIGHT))

class Item(pygame.sprite.Sprite):
    def __init__(self, pos):
        super().__init__()
        # small green circle item
        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (50, 200, 50), (15, 15), 15)
        self.rect = self.image.get_rect(center=pos)


# State Handlers

def show_menu(screen, font):
    screen.fill((20, 20, 20))
    title_surf = font.render("COLLECT-EM ALL!", True, (255,255,255))
    prompt_surf = font.render("Press ENTER to start", True, (200,200,200))
    screen.blit(title_surf,  title_surf.get_rect(center=(WIDTH//2, HEIGHT//2 - 40)))
    screen.blit(prompt_surf, prompt_surf.get_rect(center=(WIDTH//2, HEIGHT//2 + 10)))
    pygame.display.flip()

def show_game_over(screen, font, score):
    screen.fill((20, 20, 20))
    over_surf  = font.render("GAME OVER", True, (255, 50, 50))
    score_surf = font.render(f"Score: {score}", True, (255,255,255))
    retry_surf = font.render("Press ENTER to retry or ESC to quit", True, (200,200,200))
    screen.blit(over_surf,  over_surf.get_rect(center=(WIDTH//2, HEIGHT//2 - 60)))
    screen.blit(score_surf, score_surf.get_rect(center=(WIDTH//2, HEIGHT//2)))
    screen.blit(retry_surf, retry_surf.get_rect(center=(WIDTH//2, HEIGHT//2 + 60)))
    pygame.display.flip()

def run_game(screen, clock, assets):
    # start background music
    pygame.mixer.music.play(-1)

    # create player & items
    player = Player((WIDTH//2, HEIGHT//2))
    items = pygame.sprite.Group(
        *(Item((random.randint(50, WIDTH-50), random.randint(50, HEIGHT-50)))
          for _ in range(ITEM_COUNT))
    )
    all_sprites = pygame.sprite.Group(player, *items)

    score = 0
    state = PLAYING

    while state == PLAYING:
        dt = clock.tick(FPS) / 1000.0

        #  Event Handling 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return GAME_OVER, score
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return GAME_OVER, score

        #  Update 
        all_sprites.update(dt)

        #  Collision Detection 
        hits = pygame.sprite.spritecollide(player, items, dokill=True)
        for _ in hits:
            assets['collect_sfx'].play()
            score += 1

        if not items:
            # collected all items → end game
            pygame.mixer.music.stop()
            state = GAME_OVER

        #  Draw 
        screen.fill((30, 30, 30))
        all_sprites.draw(screen)
        # draw score
        score_surf = assets['font'].render(f"Score: {score}", True, (255,255,255))
        screen.blit(score_surf, (10,10))
        pygame.display.flip()

    return GAME_OVER, score


# Main Function

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pygame Demo: States & Sound")
    clock = pygame.time.Clock()

    assets = load_assets()
    state  = MENU
    score  = 0

    while True:
        if state == MENU:
            show_menu(screen, assets['font'])
            # wait for ENTER or QUIT
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    state = PLAYING

        elif state == PLAYING:
            state, score = run_game(screen, clock, assets)

        elif state == GAME_OVER:
            show_game_over(screen, assets['font'], score)
            # wait for ENTER or ESC
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        state = PLAYING
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

if __name__ == "__main__":
    main()
