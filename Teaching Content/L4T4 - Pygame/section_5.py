import sys
import random
import pygame


def circles_collide(pos_a, r_a, pos_b, r_b):
    return pos_a.distance_squared_to(pos_b) <= (r_a + r_b) ** 2


class Bullet:
    def __init__(self, pos, vel, lifetime=1.2):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.lifetime = lifetime
        self.radius = 3

    def update(self, dt, screen_size):
        self.pos += self.vel * dt
        self.lifetime -= dt

        w, h = screen_size
        if self.pos.x < 0:
            self.pos.x += w
        elif self.pos.x >= w:
            self.pos.x -= w

        if self.pos.y < 0:
            self.pos.y += h
        elif self.pos.y >= h:
            self.pos.y -= h

        return self.lifetime > 0

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 240, 120),
                           (int(self.pos.x), int(self.pos.y)), self.radius)

    def get_collision_circle(self):
        return self.pos, float(self.radius)


class Asteroid:
    SIZES = {
        "big": 40,
        "medium": 26,
        "small": 16,
    }

    def __init__(self, pos, vel, size_name="big"):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.size_name = size_name
        self.radius = self.SIZES[size_name]

        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-90, 90)

    def update(self, dt, screen_size):
        self.pos += self.vel * dt
        self.angle = (self.angle + self.spin * dt) % 360

        w, h = screen_size
        if self.pos.x < 0:
            self.pos.x += w
        elif self.pos.x >= w:
            self.pos.x -= w

        if self.pos.y < 0:
            self.pos.y += h
        elif self.pos.y >= h:
            self.pos.y -= h

    def draw(self, surface):
        pygame.draw.circle(surface, (160, 160, 170),
                           (int(self.pos.x), int(self.pos.y)),
                           int(self.radius), width=2)

        tip = pygame.Vector2(self.radius, 0).rotate(self.angle) + self.pos
        pygame.draw.line(surface, (160, 160, 170),
                         (int(self.pos.x), int(self.pos.y)),
                         (int(tip.x), int(tip.y)), width=2)

    def get_collision_circle(self):
        return self.pos, float(self.radius)

    def split(self):
        """
        Returns a list of new Asteroid objects (children).
        big -> 2 medium
        medium -> 2 small
        small -> []
        """
        if self.size_name == "small":
            return []

        next_size = "medium" if self.size_name == "big" else "small"

        children = []
        for _ in range(2):
            # Give each child a new direction/speed "kick"
            direction = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
            speed = random.uniform(120, 220) if next_size == "small" else random.uniform(90, 170)

            # Children inherit some of the parent's velocity too (classic feel)
            child_vel = self.vel * 0.4 + direction * speed
            children.append(Asteroid(self.pos, child_vel, size_name=next_size))

        return children


class Player:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)

        self.angle = -90.0

        self.turn_speed = 220.0
        self.thrust_accel = 520.0
        self.max_speed = 420.0
        self.damping = 0.995

        self.radius = 16

        self.fire_cooldown = 0.18
        self._fire_timer = 0.0
        self.bullet_speed = 650.0
        self.bullet_spawn_offset = self.radius + 4

    def update(self, dt, keys, screen_size):
        self._fire_timer = max(0.0, self._fire_timer - dt)

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angle -= self.turn_speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += self.turn_speed * dt

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            forward = pygame.Vector2(1, 0).rotate(self.angle)
            self.vel += forward * self.thrust_accel * dt

        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        self.pos += self.vel * dt
        self.vel *= self.damping

        w, h = screen_size
        if self.pos.x < 0:
            self.pos.x += w
        elif self.pos.x >= w:
            self.pos.x -= w

        if self.pos.y < 0:
            self.pos.y += h
        elif self.pos.y >= h:
            self.pos.y -= h

    def try_fire(self):
        if self._fire_timer > 0.0:
            return None

        forward = pygame.Vector2(1, 0).rotate(self.angle)
        spawn_pos = self.pos + forward * self.bullet_spawn_offset
        bullet_vel = self.vel + forward * self.bullet_speed

        self._fire_timer = self.fire_cooldown
        return Bullet(spawn_pos, bullet_vel)

    def _ship_points(self):
        tip = pygame.Vector2(self.radius, 0)
        left = pygame.Vector2(-self.radius * 0.8, self.radius * 0.6)
        right = pygame.Vector2(-self.radius * 0.8, -self.radius * 0.6)
        local = [tip, left, right]
        return [p.rotate(self.angle) + self.pos for p in local]

    def draw(self, surface):
        pygame.draw.polygon(surface, (220, 220, 240),
                            self._ship_points(), width=2)
        pygame.draw.circle(surface, (220, 220, 240),
                           (int(self.pos.x), int(self.pos.y)), 2)

    def get_collision_circle(self):
        return self.pos, float(self.radius)

    def respawn(self, pos):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)
        self.angle = -90.0


class Game:
    def __init__(self, width=800, height=450, caption="Session 5 - Asteroids Final"):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.running = True

        self.font_big = pygame.font.Font(None, 56)
        self.font_small = pygame.font.Font(None, 28)

        self.player = Player((self.width / 2, self.height / 2))
        self.bullets = []
        self.asteroids = []

        self.game_over = False

        # NEW: scoring + lives + waves
        self.score = 0
        self.lives = 3
        self.wave = 1

        # NEW: invulnerability after getting hit
        self.invuln_time = 2.0
        self.invuln_timer = 0.0

        self.start_wave()

    def start_wave(self):
        # Spawn more big asteroids each wave
        count = 4 + self.wave  # wave 1 => 5 asteroids, wave 2 => 6, ...
        for _ in range(count):
            self.spawn_asteroid(size_name="big")

    def spawn_asteroid(self, size_name="big"):
        w, h = self.width, self.height
        edge = random.choice(["top", "bottom", "left", "right"])

        if edge == "top":
            pos = pygame.Vector2(random.uniform(0, w), -10)
        elif edge == "bottom":
            pos = pygame.Vector2(random.uniform(0, w), h + 10)
        elif edge == "left":
            pos = pygame.Vector2(-10, random.uniform(0, h))
        else:
            pos = pygame.Vector2(w + 10, random.uniform(0, h))

        speed = random.uniform(60, 140) + (self.wave - 1) * 6
        direction = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        vel = direction * speed

        self.asteroids.append(Asteroid(pos, vel, size_name=size_name))

    def reset(self):
        self.player = Player((self.width / 2, self.height / 2))
        self.bullets = []
        self.asteroids = []

        self.game_over = False
        self.score = 0
        self.lives = 3
        self.wave = 1
        self.invuln_timer = 0.0

        self.start_wave()

    def award_points(self, size_name):
        # Smaller asteroids give more points
        if size_name == "big":
            self.score += 20
        elif size_name == "medium":
            self.score += 50
        else:
            self.score += 100

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
        self.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                if self.game_over and event.key == pygame.K_r:
                    self.reset()

    def update(self, dt):
        if self.game_over:
            return

        # Reduce invulnerability timer
        self.invuln_timer = max(0.0, self.invuln_timer - dt)

        keys = pygame.key.get_pressed()
        self.player.update(dt, keys, (self.width, self.height))

        # Hold SPACE to fire continuously
        if keys[pygame.K_SPACE]:
            bullet = self.player.try_fire()
            if bullet is not None:
                self.bullets.append(bullet)

        # Update bullets
        alive = []
        for b in self.bullets:
            if b.update(dt, (self.width, self.height)):
                alive.append(b)
        self.bullets = alive

        # Update asteroids
        for a in self.asteroids:
            a.update(dt, (self.width, self.height))

        # --- Collisions: bullets vs asteroids (with splitting) ---
        bullets_to_remove = set()
        asteroids_to_remove = set()
        new_asteroids = []

        for bi, b in enumerate(self.bullets):
            bpos, br = b.get_collision_circle()
            for ai, a in enumerate(self.asteroids):
                apos, ar = a.get_collision_circle()
                if circles_collide(bpos, br, apos, ar):
                    bullets_to_remove.add(bi)
                    asteroids_to_remove.add(ai)

                    # Score + split
                    self.award_points(a.size_name)
                    new_asteroids.extend(a.split())
                    break

        if bullets_to_remove or asteroids_to_remove:
            self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
            self.asteroids = [a for i, a in enumerate(self.asteroids) if i not in asteroids_to_remove]
            self.asteroids.extend(new_asteroids)

        # --- Collisions: player vs asteroids (lives + invuln) ---
        if self.invuln_timer <= 0.0:
            ppos, pr = self.player.get_collision_circle()
            for a in self.asteroids:
                apos, ar = a.get_collision_circle()
                if circles_collide(ppos, pr, apos, ar):
                    self.lives -= 1

                    if self.lives <= 0:
                        self.game_over = True
                    else:
                        # Respawn player in the center with invulnerability
                        self.player.respawn((self.width / 2, self.height / 2))
                        self.invuln_timer = self.invuln_time
                    break

        # --- Wave progression ---
        if not self.asteroids:
            self.wave += 1
            self.start_wave()

    def draw(self):
        self.screen.fill((25, 25, 35))

        for a in self.asteroids:
            a.draw(self.screen)

        for b in self.bullets:
            b.draw(self.screen)

        # Draw player with a blink effect if invulnerable
        if self.invuln_timer > 0.0:
            # Blink by skipping draw every few frames
            if int(self.invuln_timer * 10) % 2 == 0:
                self.player.draw(self.screen)
        else:
            self.player.draw(self.screen)

        # HUD
        hud = self.font_small.render(
            "Score: {}   Lives: {}   Wave: {}".format(self.score, self.lives, self.wave),
            True, (220, 220, 220)
        )
        self.screen.blit(hud, (10, 10))

        if self.game_over:
            text = self.font_big.render("GAME OVER", True, (240, 80, 80))
            hint = self.font_small.render("Press R to restart", True, (220, 220, 220))

            rect = text.get_rect(center=(self.width / 2, self.height / 2 - 20))
            rect2 = hint.get_rect(center=(self.width / 2, self.height / 2 + 25))

            self.screen.blit(text, rect)
            self.screen.blit(hint, rect2)

        pygame.display.flip()

    def quit(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
