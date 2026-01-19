"""
Session 1: Pygame Foundations + OOP Game Loop

Key ideas:
- The entire game lives inside a Game class
- The game loop runs continuously until the user quits
- Each loop follows the same structure:
    handle events -> update -> draw -> limit FPS
"""

import pygame


class Game:
    """
    The Game class is responsible for:
    - setting up pygame
    - creating the window
    - running the main loop

    Later, this class will also manage players, enemies, and game states.
    """

    def __init__(self, width=800, height=450, fps=60):
        # Initialize pygame (must be done before using most pygame features)
        pygame.init()

        # Store basic settings
        self.width = width
        self.height = height
        self.fps = fps

        # Create the window (display surface)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Session 1 - OOP Game Loop")

        # Clock controls how fast the game loop runs
        self.clock = pygame.time.Clock()

        # Controls whether the main loop keeps running
        self.running = True

        # Background color (RGB)
        self.background_color = (25, 25, 35)

    def run(self):
        """
        The main game loop.
        This loop keeps running until self.running becomes False.
        """
        while self.running:
            self.handle_events()
            self.update()
            self.draw()

            # Limit the loop to self.fps frames per second
            self.clock.tick(self.fps)

        # Cleanly shut down pygame when the loop ends
        pygame.quit()

    def handle_events(self):
        """
        Handle all pending pygame events.
        Right now, we only care about closing the window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        """
        Update game logic.
        Session 1 has no game objects yet, so this is empty.
        """
        pass

    def draw(self):
        """
        Draw everything for the current frame.
        """
        # Clear the screen
        self.screen.fill(self.background_color)

        # Update the display
        pygame.display.flip()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
