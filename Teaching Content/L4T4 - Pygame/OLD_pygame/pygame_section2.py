import pygame


class Game:
    def __init__(self, width=800, height=450, fps=60):
        pygame.init()

        self.width = width
        self.height = height
        self.fps = fps

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Drawing & Coordinates")

        self.clock = pygame.time.Clock()
        self.running = True

        # Colors
        self.background_color = (20, 20, 30)

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
        # (self.rect_x = 100, self.rect_y = 80, self.rect_width = 120, self.rect_height = 60)
        pygame.draw.rect(
            surface=self.screen,
            color=self.rectangle_color,
            rect=(100, 80, 120, 60)
        )

        # Draw a circle 
        # #(center x, center y)
        pygame.draw.circle(
            surface=self.screen,
            color=(200, 100, 100),
            center=(400, 200),
            radius=40
        )

        # Draw text
        text_surface = self.font.render(
            "Coordinates start at the top-left (0, 0)",
            True,
            (240, 240, 240)
        )
        self.screen.blit(text_surface, (20, 20))

        # Present frame
        pygame.display.flip()


def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
