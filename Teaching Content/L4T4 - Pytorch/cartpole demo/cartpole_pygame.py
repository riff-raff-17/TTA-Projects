'''This replicates the cartpole game in pygame so you can try it out'''

import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("CartPole - User Controlled")

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 50    # frames per second
TAU = 1.0 / FPS  # time step for physics integration

# CartPole parameters (similar to OpenAI Gym)
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5    # half the pole's length
polemass_length = masspole * length
force_mag = 5.0

theta_threshold_radians = 30 * (math.pi / 180)  # 30° in radians
x_threshold = 2.4  # cart position limit for game over

# Conversion from physics state to screen pixels
def physics_x_to_screen(x_phys):
    return int((x_phys + x_threshold) * (SCREEN_WIDTH / (2 * x_threshold)))

# Fixed y-position for the cart on the screen
cart_y = SCREEN_HEIGHT * 0.75

# Cart dimensions (in pixels)
CART_WIDTH = 80
CART_HEIGHT = 40
POLE_WIDTH = 10  # thickness of the pole when drawing

# Font for displaying text
font = pygame.font.SysFont(None, 48)

# Game state variables (initialized later)
x = 0.0         # cart position
x_dot = 0.0     # cart velocity
theta = 0.0     # pole angle (0 = perfectly vertical)
theta_dot = 0.0 # pole angular velocity

game_over = False
time_steps = 0

# “Waiting for SPACE” flag
in_countdown = False


def reset_environment():
    global x, x_dot, theta, theta_dot
    global game_over, time_steps
    global in_countdown

    # Small random initial state in ±0.05 for each variable
    x = random.uniform(-0.05, 0.05)
    x_dot = random.uniform(-0.05, 0.05)
    theta = random.uniform(-0.05, 0.05)
    theta_dot = random.uniform(-0.05, 0.05)

    game_over = False
    time_steps = 0

    # Wait for SPACE to start (freeze rendering)
    in_countdown = True


# Initial reset: randomize state and show “Press SPACE to Start”
reset_environment()

while True:
    # Handle Pygame events (quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # If we are waiting for SPACE to start, draw the frozen CartPole + text overlay
    if in_countdown:
        # 1) Draw white background
        screen.fill((255, 255, 255))

        # 2) Draw the track (horizontal line)
        pygame.draw.line(
            screen,
            (0, 0, 0),
            (0, cart_y + CART_HEIGHT // 2),
            (SCREEN_WIDTH, cart_y + CART_HEIGHT // 2),
            2
        )

        # 3) Draw the cart at its current x, y
        cart_x_pix = physics_x_to_screen(x) - CART_WIDTH // 2
        cart_rect = pygame.Rect(
            cart_x_pix,
            cart_y - CART_HEIGHT // 2,
            CART_WIDTH,
            CART_HEIGHT
        )
        pygame.draw.rect(screen, (0, 0, 255), cart_rect)

        # 4) Draw the pole in its current angle θ
        pole_len_pix = length * (SCREEN_HEIGHT * 0.5)
        pole_x_end = cart_x_pix + CART_WIDTH // 2 + pole_len_pix * math.sin(theta)
        pole_y_end = cart_y - CART_HEIGHT // 2 - pole_len_pix * math.cos(theta)
        pygame.draw.line(
            screen,
            (255, 0, 0),
            (cart_x_pix + CART_WIDTH // 2, cart_y - CART_HEIGHT // 2),
            (pole_x_end, pole_y_end),
            POLE_WIDTH
        )

        # 5) Overlay “Press SPACE to Start” in the center
        start_text = font.render("Press SPACE to Start", True, (0, 0, 0))
        screen.blit(
            start_text,
            (
                SCREEN_WIDTH // 2 - start_text.get_width() // 2,
                SCREEN_HEIGHT // 2 - start_text.get_height() // 2
            )
        )

        # 6) Flip display and wait for user to hit SPACE
        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            in_countdown = False

        clock.tick(FPS)
        continue  # skip physics/drawing until SPACE is pressed

    # --- Physics update (runs only when NOT in_countdown and NOT game_over) ---
    if not game_over:
        # Read user input: left/right arrows to apply force
        keys = pygame.key.get_pressed()
        force = 0.0
        if keys[pygame.K_LEFT]:
            force = -force_mag
        elif keys[pygame.K_RIGHT]:
            force = force_mag

        # CartPole dynamics (Euler integration)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - (masspole * costheta * costheta) / total_mass)
        )
        xacc = temp - (polemass_length * thetaacc * costheta) / total_mass

        # Update state
        x += TAU * x_dot
        x_dot += TAU * xacc
        theta += TAU * theta_dot
        theta_dot += TAU * thetaacc

        time_steps += 1

        # Check for failure (cart out of bounds or pole angle too large)
        if abs(x) > x_threshold or abs(theta) > theta_threshold_radians:
            game_over = True

    # --- Drawing (screen cleared each frame) ---
    screen.fill((255, 255, 255))

    # Draw the track (horizontal line) at cart_y + half cart height
    pygame.draw.line(
        screen,
        (0, 0, 0),
        (0, cart_y + CART_HEIGHT // 2),
        (SCREEN_WIDTH, cart_y + CART_HEIGHT // 2),
        2
    )

    # Draw the cart
    cart_x_pix = physics_x_to_screen(x) - CART_WIDTH // 2
    cart_rect = pygame.Rect(
        cart_x_pix,
        cart_y - CART_HEIGHT // 2,
        CART_WIDTH,
        CART_HEIGHT
    )
    pygame.draw.rect(screen, (0, 0, 255), cart_rect)

    # Draw the pole
    pole_len_pix = length * (SCREEN_HEIGHT * 0.5)
    pole_x_end = cart_x_pix + CART_WIDTH // 2 + pole_len_pix * math.sin(theta)
    pole_y_end = cart_y - CART_HEIGHT // 2 - pole_len_pix * math.cos(theta)
    pygame.draw.line(
        screen,
        (255, 0, 0),
        (cart_x_pix + CART_WIDTH // 2, cart_y - CART_HEIGHT // 2),
        (pole_x_end, pole_y_end),
        POLE_WIDTH
    )

    # Display the time survived (as score)
    elapsed_time = time_steps * TAU
    score_text = font.render(f"Time: {elapsed_time:.2f}s", True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # If game over, show “Game Over! Press R to Restart”
    if game_over:
        over_text = font.render("Game Over! Press R to Restart", True, (255, 0, 0))
        screen.blit(
            over_text,
            (SCREEN_WIDTH // 2 - over_text.get_width() // 2, SCREEN_HEIGHT // 2)
        )

        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            reset_environment()  # reinitialize state and freeze for SPACE

    # Flip the display and cap the frame rate
    pygame.display.flip()
    clock.tick(FPS)
