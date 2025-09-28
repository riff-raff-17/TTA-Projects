# mountain_car_pygame.py
# pip install pygame
import math
import random
import sys
from dataclasses import dataclass

import pygame

# -----------------------
# Physics (Gym-style)
# -----------------------
# State: position x in [-1.2, 0.6], velocity v in [-0.07, 0.07]
# Dynamics:
#   v <- v + 0.001*action - 0.0025*cos(3*x), where action ∈ {-1, 0, +1}
#   x <- x + v
# Constraints:
#   if x <= -1.2 and v < 0: v = 0
# Goal: x >= 0.5

X_MIN, X_MAX = -1.2, 0.6
V_MIN, V_MAX = -0.07, 0.07
GOAL_X = 0.5

def hill_y(x):
    """Height profile, classic: y = sin(3x). Return in model units."""
    return math.sin(3 * x)

def slope_angle(x):
    """Angle of terrain (radians) from derivative y' = 3*cos(3x)."""
    dy_dx = 3.0 * math.cos(3 * x)
    return math.atan(dy_dx)

@dataclass
class MCState:
    x: float
    v: float

class MountainCar:
    def __init__(self, start_random=True):
        self.reset(start_random=start_random)

    def reset(self, start_random=True):
        x0 = random.uniform(-0.6, -0.4) if start_random else -0.5
        self.state = MCState(x=x0, v=0.0)

    def step(self, action):
        """
        action ∈ {-1, 0, +1}
        returns (state, done)
        """
        x, v = self.state.x, self.state.v
        force = 0.001 * action
        gravity = -0.0025 * math.cos(3 * x)
        v = v + force + gravity
        v = max(V_MIN, min(V_MAX, v))
        x = x + v
        if x < X_MIN:
            x = X_MIN
            v = 0.0
        done = x >= GOAL_X
        self.state = MCState(x, v)
        return self.state, done

# -----------------------
# Pygame View
# -----------------------
WIDTH, HEIGHT = 900, 500
FPS = 60

# World-to-screen mapping
# x ∈ [X_MIN, X_MAX] -> screen x in [pad, WIDTH - pad]
# y = sin(3x) ∈ [-1, 1] -> screen y flipped & scaled to [HEIGHT - ground_pad .. top_pad]
PAD_X = 60
PAD_Y_TOP = 70
PAD_Y_BOTTOM = 90

def world_to_screen(x, y):
    sx = PAD_X + (x - X_MIN) / (X_MAX - X_MIN) * (WIDTH - 2 * PAD_X)
    # map y=-1..1 to screen y=HEIGHT-PAD_Y_BOTTOM .. PAD_Y_TOP (inverted)
    y0 = (y + 1) / 2  # 0..1
    sy = (HEIGHT - PAD_Y_BOTTOM) - y0 * (HEIGHT - PAD_Y_TOP - PAD_Y_BOTTOM)
    return int(sx), int(sy)

def draw_track(surface):
    pts = []
    steps = 800
    for i in range(steps + 1):
        x = X_MIN + (X_MAX - X_MIN) * i / steps
        y = hill_y(x)
        pts.append(world_to_screen(x, y))
    pygame.draw.lines(surface, (40, 120, 40), False, pts, 3)

def draw_goal(surface):
    gx, gy = world_to_screen(GOAL_X, hill_y(GOAL_X))
    pygame.draw.line(surface, (255, 215, 0), (gx, gy - 100), (gx, gy + 100), 3)
    font = pygame.font.SysFont(None, 24)
    txt = font.render("GOAL", True, (255, 215, 0))
    surface.blit(txt, (gx + 8, gy - 20))

def draw_car(surface, state: MCState):
    x, v = state.x, state.v
    y = hill_y(x)
    angle = slope_angle(x)

    cx, cy = world_to_screen(x, y)

    # Car body (a rounded rect) oriented to slope
    body_len = 64
    body_ht = 28

    # Build a small surface and rotate
    car_surf = pygame.Surface((body_len, body_ht), pygame.SRCALPHA)
    pygame.draw.rect(car_surf, (70, 130, 180), pygame.Rect(0, 0, body_len, body_ht), border_radius=10)
    # windows
    pygame.draw.rect(car_surf, (200, 230, 255), pygame.Rect(10, 6, 20, 12), border_radius=4)
    pygame.draw.rect(car_surf, (200, 230, 255), pygame.Rect(36, 6, 18, 12), border_radius=4)
    # wheels
    pygame.draw.circle(car_surf, (40, 40, 40), (14, body_ht - 2), 7)
    pygame.draw.circle(car_surf, (40, 40, 40), (50, body_ht - 2), 7)

    # Rotate so the car aligns with slope (add small tilt so it "leans uphill")
    deg = math.degrees(-angle)
    car_rot = pygame.transform.rotate(car_surf, deg)
    rect = car_rot.get_rect(center=(cx, cy - 12))  # slight lift so wheels sit on line
    surface.blit(car_rot, rect)

    # Velocity arrow
    vx = 5000 * v  # scale for visibility
    pygame.draw.line(surface, (255, 80, 80), (cx, cy - 40), (cx + int(vx), cy - 40), 4)
    pygame.draw.circle(surface, (255, 80, 80), (cx + int(vx), cy - 40), 4)

def draw_hud(surface, state: MCState, action, episode_time, done):
    font = pygame.font.SysFont(None, 24)
    info = [
        f"x: {state.x:+.3f}",
        f"v: {state.v:+.3f}",
        f"action: {action:+d}  (-1 left, 0 coast, +1 right)",
        f"time: {episode_time:.1f}s",
        "R to reset, Down=coast, Esc to quit",
    ]
    if done:
        info.append("🎉 Reached the goal! Press R to play again.")
    for i, line in enumerate(info):
        txt = font.render(line, True, (240, 240, 240))
        surface.blit(txt, (12, 12 + i * 22))

def main():
    pygame.init()
    pygame.display.set_caption("Mountain Car — Pygame Edition")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    env = MountainCar(start_random=True)
    action = 0
    done = False
    t = 0.0

    while True:
        dt = clock.tick(FPS) / 1000.0
        if not done:   # only keep time if episode is still active
            t += dt


        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    env.reset(start_random=True)
                    action = 0
                    done = False
                    t = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            action = -1
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            action = +1
        elif keys[pygame.K_DOWN]:
            action = 0  # coast
        else:
            # no arrow keys -> gentle decay to coast (optional)
            action = 0

        # Physics step
        if not done:
            _, done = env.step(action)

        # Render
        screen.fill((20, 20, 30))
        draw_track(screen)
        draw_goal(screen)
        draw_car(screen, env.state)
        draw_hud(screen, env.state, action, t, done)

        pygame.display.flip()

if __name__ == "__main__":
    main()
