"""
=============================================================
Hand-Controlled Shooting Game (MediaPipe + Pygame)
=============================================================
Your character sits at the CENTER of the screen.

HAND CONTROLS:
  - Index finger tip (landmark 8) sets your AIM DIRECTION
    (the angle from the screen center to your fingertip).
  - Pinch (thumb tip + index tip close together) FIRES a bullet
    in the aimed direction. Release and re-pinch to fire again.

ENEMIES  (red circles)  — drift toward you from the edges.
  ✗ Let one reach you   → lose 1 life
  ✓ Shoot one           → +10 points

FRIENDLIES  (green circles) — also drift toward you from the edges.
  ✓ Let one reach you   → +5 points  (you "saved" them)
  ✗ Shoot one           → lose 1 life

Lives: 5  — game over when lives reach 0.
Difficulty ramps up every 10 seconds (more spawns, faster).

Depends on: hand_common.py  (in the same directory)

Run:
    pip install pygame opencv-python mediapipe
    python hand_shooting_game.py

Controls:
    ESC / close window — quit
    R (on game-over screen) — restart
=============================================================
"""

import cv2
import mediapipe as mp
import pygame
import math
import random
import time

from hand_common import (
    make_detector,
    INDEX_TIP,
    THUMB_TIP,
    FPSCounter,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 960, 640
CAM_W, CAM_H = 640, 480
CENTER = (SCREEN_W // 2, SCREEN_H // 2)

SMOOTHING = 0.70          # finger-tip smoothing (0=raw, 1=frozen)
PINCH_THRESHOLD = 0.05    # normalised distance for pinch detection

# Character
CHAR_RADIUS = 30

# Bullets
BULLET_SPEED = 10
BULLET_RADIUS = 6
BULLET_LIFETIME = 1.2      # seconds before bullet disappears off-screen

# Entities
ENEMY_RADIUS = 18
FRIENDLY_RADIUS = 16
BASE_ENTITY_SPEED = 2.4    # pixels per frame at wave 1
SPAWN_INTERVAL = 2.0       # seconds between spawns at wave 1
ENEMY_RATIO = 0.65         # fraction of spawns that are enemies

REACH_RADIUS = CHAR_RADIUS + 4   # how close = "reached" the character

# Shooting
FIRE_RATE = 0.12   # seconds between bullets while pinching (≈8 bullets/sec)

# Lives
MAX_LIVES = 5

# Aim line
AIM_LINE_LEN = 80

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG           = (12, 14, 22)
CHAR_COLOR   = (80, 200, 255)
CHAR_PINCH   = (255, 180, 60)
AIM_COLOR    = (255, 255, 255)
BULLET_COLOR = (255, 230, 80)
ENEMY_COLOR  = (220, 50, 50)
FRIEND_COLOR = (60, 200, 100)
HUD_COLOR    = (220, 220, 220)
LIFE_COLOR   = (220, 50, 80)
WARN_COLOR   = (255, 100, 100)
SCORE_COLOR  = (255, 220, 60)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def norm_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def lerp(a, b, t):
    return a + (b - a) * t

def angle_to(cx, cy, tx, ty):
    """Angle in radians from (cx,cy) toward (tx,ty)."""
    return math.atan2(ty - cy, tx - cx)

def spawn_edge_position():
    """Return a random (x, y) just outside the screen edges."""
    side = random.randint(0, 3)
    margin = 30
    if side == 0:   # top
        return random.randint(0, SCREEN_W), -margin
    elif side == 1: # bottom
        return random.randint(0, SCREEN_W), SCREEN_H + margin
    elif side == 2: # left
        return -margin, random.randint(0, SCREEN_H)
    else:           # right
        return SCREEN_W + margin, random.randint(0, SCREEN_H)

def circle_collide(ax, ay, ar, bx, by, br):
    return math.hypot(ax - bx, ay - by) < ar + br

# ---------------------------------------------------------------------------
# Entity classes
# ---------------------------------------------------------------------------
class Entity:
    def __init__(self, x, y, speed, radius, color, is_enemy):
        self.x, self.y = float(x), float(y)
        self.speed = speed
        self.radius = radius
        self.color = color
        self.is_enemy = is_enemy
        self.alive = True
        # flash effect when spawned
        self.spawn_time = time.time()

    def update(self):
        cx, cy = CENTER
        angle = angle_to(self.x, self.y, cx, cy)
        self.x += math.cos(angle) * self.speed
        self.y += math.sin(angle) * self.speed

    def draw(self, surface):
        age = time.time() - self.spawn_time
        # Brief white flash on spawn
        if age < 0.15:
            t = age / 0.15
            r = int(lerp(255, self.color[0], t))
            g = int(lerp(255, self.color[1], t))
            b = int(lerp(255, self.color[2], t))
            color = (r, g, b)
        else:
            color = self.color

        ix, iy = int(self.x), int(self.y)
        pygame.draw.circle(surface, color, (ix, iy), self.radius)
        pygame.draw.circle(surface, (255, 255, 255), (ix, iy), self.radius, 2)

        # Label
        label = "E" if self.is_enemy else "F"
        font_small = pygame.font.SysFont("consolas", 13, bold=True)
        lbl = font_small.render(label, True, (255, 255, 255))
        surface.blit(lbl, (ix - lbl.get_width() // 2, iy - lbl.get_height() // 2))

    def reached_center(self):
        cx, cy = CENTER
        return math.hypot(self.x - cx, self.y - cy) < REACH_RADIUS + self.radius


class Bullet:
    def __init__(self, x, y, angle):
        self.x, self.y = float(x), float(y)
        self.vx = math.cos(angle) * BULLET_SPEED
        self.vy = math.sin(angle) * BULLET_SPEED
        self.alive = True
        self.born = time.time()

    def update(self):
        self.x += self.vx
        self.y += self.vy
        age = time.time() - self.born
        if age > BULLET_LIFETIME:
            self.alive = False
        # out of screen
        pad = 40
        if not (-pad < self.x < SCREEN_W + pad and -pad < self.y < SCREEN_H + pad):
            self.alive = False

    def draw(self, surface):
        pygame.draw.circle(surface, BULLET_COLOR,
                           (int(self.x), int(self.y)), BULLET_RADIUS)
        # glow
        glow_surf = pygame.Surface((BULLET_RADIUS * 4, BULLET_RADIUS * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*BULLET_COLOR, 60),
                           (BULLET_RADIUS * 2, BULLET_RADIUS * 2), BULLET_RADIUS * 2)
        surface.blit(glow_surf,
                     (int(self.x) - BULLET_RADIUS * 2, int(self.y) - BULLET_RADIUS * 2))


# ---------------------------------------------------------------------------
# Particle effect
# ---------------------------------------------------------------------------
class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = float(x), float(y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.life = 1.0  # 1.0 -> 0.0

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.1  # gravity
        self.life -= dt * 2.5
        return self.life > 0

    def draw(self, surface):
        alpha = int(max(0, self.life) * 220)
        r = max(2, int(self.life * 7))
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (r, r), r)
        surface.blit(s, (int(self.x) - r, int(self.y) - r))


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
class Game:
    def __init__(self):
        self.reset()

    def reset(self):
        self.score = 0
        self.lives = MAX_LIVES
        self.entities = []
        self.bullets = []
        self.particles = []
        self.wave = 1
        self.wave_start = time.time()
        self.last_spawn = time.time()
        self.game_over = False
        self.aim_angle = 0.0  # radians
        self.pinching = False
        self.prev_pinching = False
        # Smoothed fingertip position (screen coords)
        self.finger_x = float(CENTER[0]) + 100
        self.finger_y = float(CENTER[1])
        # Hit flash timer
        self.hit_flash = 0.0
        self.last_shot = 0.0   # timestamp of last bullet fired

    # ---- Difficulty ----
    def current_wave(self):
        elapsed = time.time() - self.wave_start
        return int(elapsed / 10) + 1

    def entity_speed(self):
        return BASE_ENTITY_SPEED + (self.current_wave() - 1) * 0.5

    def spawn_interval(self):
        return max(0.6, SPAWN_INTERVAL - (self.current_wave() - 1) * 0.15)

    # ---- Spawn ----
    def maybe_spawn(self):
        now = time.time()
        if now - self.last_spawn < self.spawn_interval():
            return
        self.last_spawn = now
        x, y = spawn_edge_position()
        is_enemy = random.random() < ENEMY_RATIO
        color = ENEMY_COLOR if is_enemy else FRIEND_COLOR
        radius = ENEMY_RADIUS if is_enemy else FRIENDLY_RADIUS
        e = Entity(x, y, self.entity_speed(), radius, color, is_enemy)
        self.entities.append(e)

    # ---- Shooting ----
    def try_shoot(self):
        cx, cy = CENTER
        b = Bullet(cx, cy, self.aim_angle)
        self.bullets.append(b)

    # ---- Update ----
    def update(self, dt):
        if self.game_over:
            return

        self.wave = self.current_wave()
        self.maybe_spawn()

        # Aim angle from finger position
        cx, cy = CENTER
        self.aim_angle = angle_to(cx, cy, self.finger_x, self.finger_y)

        # Continuous fire while pinching, rate-limited by FIRE_RATE
        now = time.time()
        if self.pinching and now - self.last_shot >= FIRE_RATE:
            self.try_shoot()
            self.last_shot = now

        # Update bullets
        for b in self.bullets:
            b.update()
        self.bullets = [b for b in self.bullets if b.alive]

        # Update entities & collision
        for e in self.entities:
            e.update()

            # Did entity reach center?
            if e.reached_center():
                e.alive = False
                if e.is_enemy:
                    self.lives -= 1
                    self.hit_flash = 0.5
                    self._explode(e.x, e.y, ENEMY_COLOR)
                else:
                    self.score += 5
                    self._explode(e.x, e.y, FRIEND_COLOR)
                continue

            # Bullet collision
            for b in self.bullets:
                if not b.alive:
                    continue
                if circle_collide(b.x, b.y, BULLET_RADIUS, e.x, e.y, e.radius):
                    b.alive = False
                    e.alive = False
                    if e.is_enemy:
                        self.score += 10
                        self._explode(e.x, e.y, ENEMY_COLOR)
                    else:
                        self.lives -= 1
                        self.hit_flash = 0.5
                        self._explode(e.x, e.y, FRIEND_COLOR)
                    break

        self.entities = [e for e in self.entities if e.alive]
        self.bullets = [b for b in self.bullets if b.alive]

        # Particles
        self.particles = [p for p in self.particles if p.update(dt)]

        # Hit flash decay
        self.hit_flash = max(0.0, self.hit_flash - dt * 2)

        if self.lives <= 0:
            self.lives = 0
            self.game_over = True

    def _explode(self, x, y, color):
        for _ in range(18):
            self.particles.append(Particle(x, y, color))

    # ---- Draw ----
    def draw(self, surface, font, font_big, font_med):
        # Background
        surface.fill(BG)

        # Screen-edge red flash when hit
        if self.hit_flash > 0:
            alpha = int(self.hit_flash * 120)
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((220, 30, 30, alpha))
            surface.blit(overlay, (0, 0))

        # Subtle grid
        for gx in range(0, SCREEN_W, 60):
            pygame.draw.line(surface, (25, 28, 40), (gx, 0), (gx, SCREEN_H))
        for gy in range(0, SCREEN_H, 60):
            pygame.draw.line(surface, (25, 28, 40), (0, gy), (SCREEN_W, gy))

        # Entities
        for e in self.entities:
            e.draw(surface)

        # Bullets
        for b in self.bullets:
            b.draw(surface)

        # Particles
        for p in self.particles:
            p.draw(surface)

        # Aim line
        cx, cy = CENTER
        ax = cx + math.cos(self.aim_angle) * AIM_LINE_LEN
        ay = cy + math.sin(self.aim_angle) * AIM_LINE_LEN
        pygame.draw.line(surface, (255, 255, 255, 120), (cx, cy), (int(ax), int(ay)), 2)
        # Arrowhead
        for off in (-0.4, 0.4):
            hx = ax + math.cos(self.aim_angle + math.pi + off) * 12
            hy = ay + math.sin(self.aim_angle + math.pi + off) * 12
            pygame.draw.line(surface, AIM_COLOR, (int(ax), int(ay)), (int(hx), int(hy)), 2)

        # Character
        color = CHAR_PINCH if self.pinching else CHAR_COLOR
        radius = CHAR_RADIUS + 6 if self.pinching else CHAR_RADIUS
        pygame.draw.circle(surface, color, CENTER, radius)
        pygame.draw.circle(surface, (255, 255, 255), CENTER, radius, 2)
        # Inner dot
        pygame.draw.circle(surface, (255, 255, 255), CENTER, 5)

        # HUD — top left
        lines = [
            f"SCORE : {self.score}",
            f"WAVE  : {self.wave}",
        ]
        for i, line in enumerate(lines):
            surf = font.render(line, True, HUD_COLOR)
            surface.blit(surf, (12, 10 + i * 22))

        # Lives — top right as hearts
        for i in range(MAX_LIVES):
            color = LIFE_COLOR if i < self.lives else (50, 50, 60)
            pygame.draw.circle(surface, color,
                               (SCREEN_W - 20 - i * 28, 22), 10)

        lives_lbl = font.render("LIVES", True, HUD_COLOR)
        surface.blit(lives_lbl, (SCREEN_W - 20 - MAX_LIVES * 28 - lives_lbl.get_width() - 6, 14))

        # Legend bottom
        legend = [
            ("RED = Enemy  → shoot to score (+10)",  ENEMY_COLOR),
            ("GREEN = Friendly  → let them reach you (+5) / don't shoot!", FRIEND_COLOR),
        ]
        for i, (text, col) in enumerate(legend):
            surf = font.render(text, True, col)
            surface.blit(surf, (12, SCREEN_H - 14 - (len(legend) - i) * 20))

        # Game Over overlay
        if self.game_over:
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            surface.blit(overlay, (0, 0))

            go = font_big.render("GAME OVER", True, (220, 50, 50))
            surface.blit(go, (SCREEN_W // 2 - go.get_width() // 2, SCREEN_H // 2 - 80))

            sc = font_med.render(f"Final Score: {self.score}", True, SCORE_COLOR)
            surface.blit(sc, (SCREEN_W // 2 - sc.get_width() // 2, SCREEN_H // 2 - 10))

            restart = font_med.render("Press  R  to restart", True, HUD_COLOR)
            surface.blit(restart, (SCREEN_W // 2 - restart.get_width() // 2, SCREEN_H // 2 + 50))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Hand Shooting Game")
    clock = pygame.time.Clock()

    font     = pygame.font.SysFont("consolas", 16)
    font_med = pygame.font.SysFont("consolas", 28, bold=True)
    font_big = pygame.font.SysFont("consolas", 56, bold=True)

    # Webcam + MediaPipe
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    detector = make_detector(num_hands=1)
    fps_counter = FPSCounter()

    game = Game()
    prev_time = time.time()

    running = True
    while running:
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # ---- Webcam frame ----
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = detector.detect(mp_image)

        hand_visible = False
        if result.hand_landmarks:
            hand_visible = True
            lms = result.hand_landmarks[0]
            index_tip = lms[INDEX_TIP]
            thumb_tip = lms[THUMB_TIP]

            # Smooth finger position
            raw_fx = index_tip.x * SCREEN_W
            raw_fy = index_tip.y * SCREEN_H
            game.finger_x = lerp(game.finger_x, raw_fx, 1 - SMOOTHING)
            game.finger_y = lerp(game.finger_y, raw_fy, 1 - SMOOTHING)

            game.pinching = norm_distance(index_tip, thumb_tip) < PINCH_THRESHOLD

            # Draw fingertip on camera preview
            cv2.circle(frame,
                       (int(index_tip.x * img_w), int(index_tip.y * img_h)),
                       8, (0, 255, 0), -1)
        else:
            game.pinching = False

        # ---- Pygame events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r and game.game_over:
                    game = Game()

        # ---- Update ----
        game.update(dt)

        # ---- Render ----
        game.draw(screen, font, font_big, font_med)

        # No hand warning
        if not hand_visible:
            msg = font_med.render("Show your hand to the camera!", True, WARN_COLOR)
            screen.blit(msg, (SCREEN_W // 2 - msg.get_width() // 2, 30))

        # FPS counter (tiny, top center)
        fps_surf = font.render(f"{clock.get_fps():.0f} fps", True, (80, 80, 100))
        screen.blit(fps_surf, (SCREEN_W // 2 - fps_surf.get_width() // 2, 4))

        pygame.display.flip()
        fps_counter.tick()

        # Optional camera preview
        cv2.imshow("Camera (press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False

        clock.tick(60)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    pygame.quit()


if __name__ == "__main__":
    main()