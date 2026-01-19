"""
Session 2: Drawing, Coordinates, and Screen Space

Key ideas:
- The screen is a 2D coordinate system with (0, 0) at the top-left
- Drawing happens every frame inside draw()
- Game owns the screen; objects do not draw globally
"""

import pygame


class Game:
    def __init__(self, width=800, height=450, fps=60):
        pygame.init()

        self.width = width
        self.height = height
        self.fps = fps

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Session 2 - Drawing & Coordinates")

        self.clock = pygame.time.Clock()
        self.running = True

        # Colors
        self.background_color = (20, 20, 30)
        self.rectangle_color = (80, 160, 220)
        self.text_color = (240, 240, 240)

        # Rectangle properties
        self.rect_x = 100
        self.rect_y = 80
        self.rect_width = 120
        self.rect_height = 60

        # Font setup (None means default system font)
        self.font = pygame.font.Font(None, 32)

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
        # No movement or logic yet
        pass

    def draw(self):
        # Clear screen
        self.screen.fill(self.background_color)

        # Draw a rectangle
        pygame.draw.rect(
            self.screen,
            self.rectangle_color,
            (self.rect_x, self.rect_y, self.rect_width, self.rect_height)
        )

        # Draw a circle (center x, center y, radius)
        pygame.draw.circle(
            self.screen,
            (200, 100, 100),
            (400, 200),
            40
        )

        # Draw text
        text_surface = self.font.render(
            "Coordinates start at the top-left (0, 0)",
            True,
            self.text_color
        )
        self.screen.blit(text_surface, (20, 20))

        # Present frame
        pygame.display.flip()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
