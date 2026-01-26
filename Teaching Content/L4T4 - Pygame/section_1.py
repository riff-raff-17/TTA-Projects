import sys
import pygame


class Game:
    def __init__(self, width=800, height=450, caption="Session 1 - Pygame OOP Skeleton"):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.running = True

        # We'll use dt (delta time) in later sessions for smooth movement.
        self.dt = 0.0

    def run(self):
        """Main game loop."""
        while self.running:
            # dt in seconds (e.g., 0.016 at ~60 FPS)
            self.dt = self.clock.tick(60) / 1000.0

            self.handle_events()
            self.update(self.dt)
            self.draw()

        self.quit()

    def handle_events(self):
        """Handle all input/events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self, dt):
        """Update game state. (Nothing yet in Session 1.)"""
        pass

    def draw(self):
        """Draw everything each frame."""
        self.screen.fill((25, 25, 35))  # Background color

        # In later sessions we’ll draw entities here.

        pygame.display.flip()

    def quit(self):
        """Clean shutdown."""
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
