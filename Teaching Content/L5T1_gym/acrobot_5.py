import pygame
import math

# --- Setup ---
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Acrobot")

clock = pygame.time.Clock()
FPS = 60

WHITE = (255, 255, 255)
LINK_TEAL = (92, 201, 202)
JOINT_YELLOW = (204, 204, 66)
GOAL_GRAY = (96, 96, 96)

# --- Acrobot geometry ---
PIVOT = (WIDTH // 2, HEIGHT // 3)   # fixed point the whole system hangs from
LINK_LENGTH = 120                    # pixels, same for both links
LINK_WIDTH = 14                      # thickness of each link
JOINT_RADIUS = 8
GOAL_Y = PIVOT[1] - LINK_LENGTH      # goal line: one link-length above the pivot

# --- Physics constants (matching Gymnasium's real Acrobot-v1 values) ---
# These describe the physical system in "physics units" (meters, kg, etc.),
# completely separate from the pixel units used for drawing above.
LINK_MASS_1 = 1.0     # mass of link 1 [kg]
LINK_MASS_2 = 1.0     # mass of link 2 [kg]
LINK_COM_POS_1 = 0.5  # position of link 1's center of mass [m]
LINK_COM_POS_2 = 0.5  # position of link 2's center of mass [m]
LINK_MOI = 1.0        # moment of inertia for both links
PHYS_LINK_LENGTH = 1.0  # link length in physics units (not pixels!)
GRAVITY = 9.8
DT = 0.02             # physics timestep in seconds (small = more stable)
TORQUE_MAGNITUDE = 1.0  # matches Gymnasium's AVAIL_TORQUE = [-1, 0, +1]

# Acrobot state (radians, rad/s). theta1 from straight down, ccw positive.
# theta2 is relative to link 1 (the "elbow bend").
# Starting near hanging straight down, like the real Gymnasium environment.
theta1 = 0.0
theta2 = 0.0
theta1_dot = 0.0   # angular velocity of joint 1
theta2_dot = 0.0   # angular velocity of joint 2


def get_joint_positions(theta1, theta2):
    """Compute pixel positions of the elbow and free end from the two angles."""
    # theta measured from straight down; pygame y-axis points down,
    # so "down" is +y and we rotate from there.
    x0, y0 = PIVOT

    # Elbow joint (end of link 1)
    x1 = x0 + LINK_LENGTH * math.sin(theta1)
    y1 = y0 + LINK_LENGTH * math.cos(theta1)

    # Free end (end of link 2), angle is theta1 + theta2 in world frame
    x2 = x1 + LINK_LENGTH * math.sin(theta1 + theta2)
    y2 = y1 + LINK_LENGTH * math.cos(theta1 + theta2)

    return (x0, y0), (x1, y1), (x2, y2)


def compute_accelerations(theta1, theta2, theta1_dot, theta2_dot, torque):
    """The acrobot's equations of motion: given the current state and the
    applied torque, return the angular accelerations (theta1_dotdot,
    theta2_dotdot). These come from Lagrangian mechanics for a double
    pendulum with a motor at the second joint - same equations Gymnasium
    uses internally."""
    m1, m2 = LINK_MASS_1, LINK_MASS_2
    l1 = PHYS_LINK_LENGTH
    lc1, lc2 = LINK_COM_POS_1, LINK_COM_POS_2
    I1, I2 = LINK_MOI, LINK_MOI
    g = GRAVITY

    # Effective inertia terms
    d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(theta2)) + I1 + I2
    d2 = m2 * (lc2**2 + l1 * lc2 * math.cos(theta2)) + I2

    # Gravity / Coriolis terms
    phi2 = m2 * lc2 * g * math.cos(theta1 + theta2 - math.pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * theta2_dot**2 * math.sin(theta2)
        - 2 * m2 * l1 * lc2 * theta2_dot * theta1_dot * math.sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * math.cos(theta1 - math.pi / 2)
        + phi2
    )

    theta2_dotdot = (
        torque + d2 / d1 * phi1 - m2 * l1 * lc2 * theta1_dot**2 * math.sin(theta2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    theta1_dotdot = -(d2 * theta2_dotdot + phi1) / d1

    return theta1_dotdot, theta2_dotdot


def step_physics(theta1, theta2, theta1_dot, theta2_dot, torque, dt):
    """Advance the system state by one timestep using semi-implicit Euler:
    update velocities first, then use the *new* velocities to update angles.
    This is much simpler than RK4 and plenty stable for this system at a
    small dt."""
    theta1_dotdot, theta2_dotdot = compute_accelerations(
        theta1, theta2, theta1_dot, theta2_dot, torque
    )

    theta1_dot += theta1_dotdot * dt
    theta2_dot += theta2_dotdot * dt

    theta1 += theta1_dot * dt
    theta2 += theta2_dot * dt

    return theta1, theta2, theta1_dot, theta2_dot


def draw_acrobot(surface, theta1, theta2):
    pivot, elbow, tip = get_joint_positions(theta1, theta2)

    pygame.draw.line(surface, LINK_TEAL, pivot, elbow, LINK_WIDTH)
    pygame.draw.line(surface, LINK_TEAL, elbow, tip, LINK_WIDTH)

    pygame.draw.circle(surface, JOINT_YELLOW, (int(pivot[0]), int(pivot[1])), JOINT_RADIUS)
    pygame.draw.circle(surface, JOINT_YELLOW, (int(elbow[0]), int(elbow[1])), JOINT_RADIUS)


# --- Main loop ---
running = True
dt_ms = 0               # milliseconds the previous frame took
time_to_simulate = 0.0  # leftover physics time to catch up on
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Read keyboard input once per frame. Left/right arrows apply torque at
    # the actuated joint (magnitude matches Gymnasium's discrete action
    # space). Note the mapping below is swapped relative to the sign you
    # might expect: with our angle convention (x = pivot_x + L*sin(theta)),
    # positive torque visually swings the arm toward screen-left, so we map
    # the LEFT arrow to +1 and RIGHT to -1 to match what the player sees.
    # Holding nothing (or both) applies zero torque.
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
        applied_torque = TORQUE_MAGNITUDE
    elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
        applied_torque = -TORQUE_MAGNITUDE
    else:
        applied_torque = 0.0

    # Update physics. We step with a fixed-size DT (not the variable frame
    # time) because physics simulations are more stable and predictable with
    # a constant timestep - we just run as many small steps as needed to
    # cover however long the frame took. The torque read above is held
    # constant across all of this frame's physics substeps.
    time_to_simulate += dt_ms / 1000.0
    while time_to_simulate >= DT:
        theta1, theta2, theta1_dot, theta2_dot = step_physics(
            theta1, theta2, theta1_dot, theta2_dot, applied_torque, DT
        )
        time_to_simulate -= DT

    # Draw
    screen.fill(WHITE)
    pygame.draw.line(screen, GOAL_GRAY, (0, GOAL_Y), (WIDTH, GOAL_Y), 2)
    draw_acrobot(screen, theta1, theta2)
    pygame.display.flip()

    # Cap framerate
    dt_ms = clock.tick(FPS)

pygame.quit()