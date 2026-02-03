import sys
import pygame

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


class Player:
    def __init__(self, pos):
        # Physics
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)

        # Ship orientation (degrees). 0 = pointing right.
        self.angle = -90.0  # start pointing up

        # Tuning
        self.turn_speed = 220.0 # degrees per second
        self.thrust_accel = 520.0 # pixels per second^2
        self.max_speed = 420.0 # clamp velocity magnitude
        self.damping = 0.995 # slight drift reduction per frame

        # Rendering/collision
        self.radius = 16

        # Shooting
        self.fire_cooldown = 0.18
        self._fire_timer = 0.0
        self.bullet_speed = 650.0
        self.bullet_spawn_offset = self.radius + 4

    def update(self, dt, keys, screen_size):
        self._fire_timer = max(0.0, self._fire_timer - dt)

        # Rotation
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.angle -= self.turn_speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.angle += self.turn_speed * dt

        # Thrust (acceleration in facing direction)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            forward = pygame.Vector2(1, 0).rotate(self.angle)
            self.vel += forward * self.thrust_accel * dt

        # Clamp speed
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        # Move
        self.pos += self.vel * dt

        # Mild damping to keep things controllable (can reduce/remove later)
        self.vel *= self.damping

        # Screen wrap
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
        """
        Returns 3 points (triangle) in world space.
        We'll draw a simple triangle ship.
        """
        # Define ship triangle in local space (pointing right),
        # then rotate by angle and translate by pos.
        tip = pygame.Vector2(self.radius, 0)
        left = pygame.Vector2(-self.radius * 0.8, self.radius * 0.6)
        right = pygame.Vector2(-self.radius * 0.8, -self.radius * 0.6)

        pts = [tip, left, right]
        return [p.rotate(self.angle) + self.pos for p in pts]

    def draw(self, surface):
        pygame.draw.polygon(surface, (220, 220, 240),
                            self._ship_points(), width=2)
        
        # Optional: draw a tiny center dot (helps see pos)
        pygame.draw.circle(surface, (220, 220, 240),
                           (int(self.pos.x), int(self.pos.y)), 2)

    def get_collision_circle(self):
        return self.pos, float(self.radius)


class Game:
    def __init__(self, width=800, height=450, caption="Session 3 - Shooting"):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0.0

        self.player = Player((self.width / 2, self.height / 2))
        self.bullets = []

    def run(self):
        """Main game loop."""
        while self.running:
            # dt in seconds (e.g., 0.016 at ~60 FPS)
            self.dt = self.clock.tick(60) / 1000.0

            self.handle_events()
            self.update(self.dt)
            self.draw()

        self.quit()

    def handle_events(self):
        """Handle all input/events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

    def update(self, dt):
        keys = pygame.key.get_pressed()
        self.player.update(dt, keys, (self.width, self.height))

        # NEW: hold SPACE to fire continuously (cooldown controls rate)
        if keys[pygame.K_SPACE]:
            bullet = self.player.try_fire()
            if bullet is not None:
                self.bullets.append(bullet)

        alive = []
        for b in self.bullets:
            if b.update(dt, (self.width, self.height)):
                alive.append(b)
        self.bullets = alive


    def draw(self):
        """Draw everything each frame."""
        self.screen.fill((25, 25, 35))

        for b in self.bullets:
            b.draw(self.screen)

        self.player.draw(self.screen)
        pygame.display.flip()

    def quit(self):
        """Clean shutdown."""
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    Game().run()
