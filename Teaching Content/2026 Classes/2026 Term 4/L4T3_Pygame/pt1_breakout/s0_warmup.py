"""
Breakout -- Stage 0: Pygame Refresher
========================================

Not part of the numbered progression -- just a warm-up before Stage 1,
in case it's been a while since Session 2. Everything here is
something Stage 1 uses immediately, just without "paddle" or "ball"
attached to it yet:

    - the window + game loop skeleton (open it, keep it alive, close
      it cleanly) is IDENTICAL to every stage that follows.
    - drawing a rectangle is what a paddle (and every brick) will be.
    - drawing a circle is what the ball will be.
    - moving a shape with the arrow keys and clamping it to the
      window is exactly what Paddle.update() does in Stage 1 -- just
      constrained to one axis there instead of two.

Nothing here is graded or kept -- change the numbers, break it, see
what happens. If this all feels familiar, you're ready for Stage 1.
"""

import pygame

pygame.init()

# Same window size and background Stage 1 uses, so nothing changes
# visually when you open that file next.
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
BACKGROUND_COLOR = (18, 18, 24)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Breakout -- Stage 0: Pygame Refresher")

clock = pygame.time.Clock()
FPS = 60

# A couple of colors just for this refresher -- Stage 1 introduces its
# own PADDLE_COLOR / BALL_COLOR config instead.
SQUARE_COLOR = (80, 200, 255)
CIRCLE_COLOR = (255, 210, 80)
LINE_COLOR = (90, 90, 110)

# A free-roaming shape, moved with the arrow keys -- the direct
# ancestor of Stage 1's Paddle, before it gets locked to one axis and
# wrapped in a class.
circle_x, circle_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
circle_radius = 20
move_speed = 5  # pixels moved per frame while a key is held

running = True
while running:
    # --- 1. Handle events ---------------------------------------------------
    # Every pygame program drains this queue every frame -- skip it and
    # the window stops responding to anything, including the close button.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- 2. Update state ------------------------------------------------------
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        circle_x -= move_speed
    if keys[pygame.K_RIGHT]:
        circle_x += move_speed
    if keys[pygame.K_UP]:
        circle_y -= move_speed
    if keys[pygame.K_DOWN]:
        circle_y += move_speed

    # Clamp: never let the circle's edge pass the window's edge. Stage
    # 1's Paddle does this exact max()/min() pairing, just on
    # rect.left/rect.right instead of a bare x value.
    circle_x = max(circle_radius, min(WINDOW_WIDTH - circle_radius, circle_x))
    circle_y = max(circle_radius, min(WINDOW_HEIGHT - circle_radius, circle_y))

    # --- 3. Draw the frame ------------------------------------------------------
    # fill() must run first every frame -- pygame draws to an
    # off-screen buffer, so without this you'd see smearing as old
    # frames stay on screen underneath the new one.
    screen.fill(BACKGROUND_COLOR)

    # A static rectangle and line, just to recap the other primitives
    # -- a paddle (Stage 1) and every brick (Stage 2) are both nothing
    # more than a pygame.draw.rect() call like this one.
    pygame.draw.rect(screen, SQUARE_COLOR, (40, 40, 120, 50), border_radius=6)
    pygame.draw.line(
        screen, LINE_COLOR, (0, WINDOW_HEIGHT - 60), (WINDOW_WIDTH, WINDOW_HEIGHT - 60), 2
    )

    # The moving circle -- same pygame.draw.circle() call the ball
    # will use in Stage 1, just driven by keys here instead of a
    # stored velocity.
    pygame.draw.circle(screen, CIRCLE_COLOR, (circle_x, circle_y), circle_radius)

    # pygame.display.flip() is what actually shows everything drawn
    # above -- nothing is visible on screen until this call happens.
    pygame.display.flip()

    # Caps the loop at FPS frames per second -- without this the loop
    # runs as fast as the CPU allows, and movement speed stops being
    # consistent across different machines.
    clock.tick(FPS)

pygame.quit()