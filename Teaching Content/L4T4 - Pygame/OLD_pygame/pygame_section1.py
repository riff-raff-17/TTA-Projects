import pygame

class Game:
    def __init__(self, width=800, height=450, fps=60):
        # Initialize pygame (must be done before using most pygame features)
        pygame.init()

        # Store basic settings
        self.width = width
        self.height = height
        self.fps = fps

        # Create the window (display surface)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Game Loop")

        # Clock controls how fast the game loop runs
        self.clock = pygame.time.Clock()

        # Controls whether the main loop keeps running
        self.running = True

        # Background color (RGB)
        self.background_color = (25, 25, 35)

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()

            # Limit the loop to self.fps frames per second
            self.clock.tick(self.fps)

        # Cleanly shut down pygame when the loop ends
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        pass

    def draw(self):
        # Clear the screen
        self.screen.fill(self.background_color)

        # Update the display
        pygame.display.flip()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
