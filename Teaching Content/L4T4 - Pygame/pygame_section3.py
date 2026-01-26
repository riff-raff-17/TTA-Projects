"""
Session 3: Introducing a Player Class (Game Objects)

Key ideas:
- A game object is something with state and behavior
- The Player is no longer just a rectangle — it is an object
- Game owns and manages objects, but objects manage themselves
"""

import pygame


class Player:
    """
    The Player represents a game entity.

    It knows:
    - where it is (position)
    - what it looks like (size, color)
    - how to draw itself

    It does NOT know about the game loop or the window.
    """

    def __init__(self, x, y, width, height, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def update(self):
        """
        Update player logic.
        (No movement yet — coming next session)
        """
        pass

    def draw(self, surface):
        """
        Draw the player onto the given surface.
        """
        pygame.draw.rect(
            surface,
            self.color,
            (self.x, self.y, self.width, self.height)
        )


class Game:
    def __init__(self, width=800, height=450, fps=60):
        pygame.init()

        self.width = width
        self.height = height
        self.fps = fps

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Session 3 - Player Class")

        self.clock = pygame.time.Clock()
        self.running = True

        self.background_color = (20, 20, 30)

        # Create a Player object
        self.player = Player(
            x=100,
            y=150,
            width=120,
            height=60,
            color=(80, 160, 220)
        )

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        # Update all game objects
        self.player.update()

    def draw(self):
        self.screen.fill(self.background_color)

        # Draw all game objects
        self.player.draw(self.screen)

        pygame.display.flip()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
