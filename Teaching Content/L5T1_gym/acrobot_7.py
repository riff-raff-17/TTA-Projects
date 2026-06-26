import pygame
import math

# --- Setup ---
pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Acrobot")

clock = pygame.time.Clock()
FPS = 60
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 24)

WHITE = (255, 255, 255)
LINK_TEAL = (92, 201, 202)
JOINT_YELLOW = (204, 204, 66)
GOAL_GRAY = (96, 96, 96)
TEXT_GRAY = (140, 140, 140)
TORQUE_ARROW = (230, 100, 60)

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
GOAL_HEIGHT = 1.0       # win when -cos(theta1) - cos(theta1+theta2) > this
GYM_STEP_DT = 0.2       # one Gymnasium "step" = this many seconds of physics
MAX_STEPS = 500         # matches Gymnasium's Acrobot-v1 episode time limit

# Acrobot state (radians, rad/s). theta1 from straight down, ccw positive.
# theta2 is relative to link 1 (the "elbow bend").
# Starting near hanging straight down, like the real Gymnasium environment.
theta1 = 0.0
theta2 = 0.0
theta1_dot = 0.0   # angular velocity of joint 1
theta2_dot = 0.0   # angular velocity of joint 2

step_count = 0          # number of Gymnasium-equivalent steps elapsed
episode_over = False    # True once goal is reached or time runs out
result_text = ""        # what to display once the episode ends
time_since_gym_step = 0.0  # tracks progress toward the next counted step

best_steps = None        # fastest solve this session (None until first win)


def reset_episode():
    """Reset the acrobot to hanging straight down and clear episode state,
    so pressing R feels like starting a fresh attempt. best_steps is
    intentionally NOT touched here, since it should persist across resets."""
    global theta1, theta2, theta1_dot, theta2_dot
    global step_count, episode_over, result_text, time_since_gym_step

    theta1 = 0.0
    theta2 = 0.0
    theta1_dot = 0.0
    theta2_dot = 0.0

    step_count = 0
    episode_over = False
    result_text = ""
    time_since_gym_step = 0.0


def tip_height(theta1, theta2):
    """Height of the free end above the pivot, in physics units (link
    lengths). This is Gymnasium's exact termination check:
    -cos(theta1) - cos(theta1 + theta2) > 1.0 means "goal reached"."""
    return -math.cos(theta1) - math.cos(theta1 + theta2)


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


def draw_torque_arrow(surface, theta1, theta2, torque):
    """Small curved arrow at the elbow showing which way torque is
    currently being applied. Purely cosmetic - makes the otherwise
    invisible 'applied_torque' variable visible to the player."""
    if torque == 0:
        return

    _, elbow, _ = get_joint_positions(theta1, theta2)
    ex, ey = elbow
    radius = 30
    direction = -1 if torque > 0 else 1

    # Build a short arc out of small line segments, then add an
    # arrowhead at the leading end (in the direction of rotation).
    span = math.radians(130)
    n_segments = 20
    points = []
    for i in range(n_segments + 1):
        a = direction * (-span / 2 + span * (i / n_segments))
        x = ex + radius * math.sin(a)
        y = ey - radius * math.cos(a)
        points.append((x, y))
    pygame.draw.lines(surface, TORQUE_ARROW, False, points, 3)

    tip, prev = points[-1], points[-3]
    angle = math.atan2(tip[1] - prev[1], tip[0] - prev[0])
    head_size = 9
    left = (tip[0] - head_size * math.cos(angle - 0.5), tip[1] - head_size * math.sin(angle - 0.5))
    right = (tip[0] - head_size * math.cos(angle + 0.5), tip[1] - head_size * math.sin(angle + 0.5))
    pygame.draw.line(surface, TORQUE_ARROW, left, tip, 3)
    pygame.draw.line(surface, TORQUE_ARROW, right, tip, 3)


# --- Main loop ---
running = True
dt_ms = 0                     # milliseconds the previous frame took
time_to_simulate = 0.0        # leftover physics time to catch up on
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            reset_episode()

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
    # Once the episode is over (goal reached or time limit hit) we stop
    # stepping the physics, so the final pose stays frozen on screen.
    if not episode_over:
        time_to_simulate += dt_ms / 1000.0
        while time_to_simulate >= DT:
            theta1, theta2, theta1_dot, theta2_dot = step_physics(
                theta1, theta2, theta1_dot, theta2_dot, applied_torque, DT
            )
            time_to_simulate -= DT
            time_since_gym_step += DT

            # Count a "step" every GYM_STEP_DT seconds of simulated time,
            # matching how Gymnasium counts one step per env.step() call
            # (each of which advances physics by 0.2s internally).
            while time_since_gym_step >= GYM_STEP_DT:
                time_since_gym_step -= GYM_STEP_DT
                step_count += 1

                if tip_height(theta1, theta2) > GOAL_HEIGHT:
                    episode_over = True
                    result_text = f"Solved in {step_count} steps!"
                    if best_steps is None or step_count < best_steps:
                        best_steps = step_count
                elif step_count >= MAX_STEPS:
                    episode_over = True
                    result_text = "Time's up - try again"

            if episode_over:
                break

    # Draw
    screen.fill(WHITE)
    pygame.draw.line(screen, GOAL_GRAY, (0, GOAL_Y), (WIDTH, GOAL_Y), 2)
    draw_acrobot(screen, theta1, theta2)
    if not episode_over:
        draw_torque_arrow(screen, theta1, theta2, applied_torque)

    step_text = font.render(f"Step: {step_count}/{MAX_STEPS}", True, (0, 0, 0))
    screen.blit(step_text, (10, 10))

    if best_steps is not None:
        best_text = small_font.render(f"Best: {best_steps} steps", True, TEXT_GRAY)
        screen.blit(best_text, (10, 46))

    instructions = small_font.render(
        "Left/Right arrows to swing - reach the line!", True, TEXT_GRAY
    )
    screen.blit(instructions, (10, HEIGHT - 28))

    if episode_over:
        msg = font.render(result_text, True, (0, 0, 0))
        msg_rect = msg.get_rect(center=(WIDTH // 2, 40))
        screen.blit(msg, msg_rect)

        hint = small_font.render("Press R to try again", True, TEXT_GRAY)
        hint_rect = hint.get_rect(center=(WIDTH // 2, 70))
        screen.blit(hint, hint_rect)

    pygame.display.flip()

    # Cap framerate
    dt_ms = clock.tick(FPS)

pygame.quit()