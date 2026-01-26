import sys
import pygame

class Player:
    def __init__(self, x, y, size = 40, speed = 300.0):
        self.rect = pygame.Rect(x, y, size, size)
        self.speed = speed  # pixels per second

    def update(self, dt, keys, bounds_rect):
        dx = 0.0
        dy = 0.0

        # Arrow keys or WASD
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= self.speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += self.speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= self.speed * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += self.speed * dt

        # Move (Rect stores ints, so we accumulate into x/y via floats if needed later
        self.rect.x += int(dx)
        self.rect.y += int(dy)

        # Keep player inside the window
        self.rect.clamp_ip(bounds_rect)

    def draw(self, surface):
        pygame.draw.rect(surface, (80, 200, 120), self.rect)


class Game:
    def __init__(self, width=800, height=450, caption="Session 2 - Player Movement"):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0.0

        # NEW: Create a bounds rect and a player
        self.bounds = self.screen.get_rect()
        start_x = self.bounds.centerx - 20
        start_y = self.bounds.centery - 20
        self.player = Player(start_x, start_y)

    def run(self):
        while self.running:
            self.dt = self.clock.tick(60) / 1000.0

            self.handle_events()
            self.update(self.dt)
            self.draw()

        self.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self, dt: float):
        # NEW: read held keys + update player
        keys = pygame.key.get_pressed()
        self.player.update(dt, keys, self.bounds)

    def draw(self):
        self.screen.fill((25, 25, 35))

        # NEW: draw the player
        self.player.draw(self.screen)

        pygame.display.flip()

    def quit(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
