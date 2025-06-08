import sys
import pygame
from functools import partial

# movement callbacks
# insert ugot here when needed
def forward():     
    print('forward')
def backward():    
    print('backward')
def strafe_left(): 
    print('strafe left')
def strafe_right():
    print('strafe right')
def turn_left():   
    print('turn left')
def turn_right():  
    print('turn right')

class Button:
    __slots__ = ('rect','text','callback','font')
    def __init__(self, rect, text, callback, font):
        self.rect, self.text, self.callback, self.font = pygame.Rect(rect), text, callback, font

    def draw(self, surf):
        pygame.draw.rect(surf, (70,70,70), self.rect)
        lbl = self.font.render(self.text, True, (255,255,255))
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 \
        and self.rect.collidepoint(ev.pos):
            self.callback()

class Slider:
    __slots__ = ('rect','min_val','max_val','value','callback','knob_radius','dragging')
    def __init__(self, rect, mn, mx, init, callback):
        self.rect, self.min_val, self.max_val = pygame.Rect(rect), mn, mx
        self.value, self.callback = init, callback
        self.knob_radius, self.dragging = 8, False

    def draw(self, surf):
        # track
        y = self.rect.centery
        pygame.draw.line(surf, (150,150,150),
                         (self.rect.x, y),
                         (self.rect.right, y), 4)
        # knob
        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        x = self.rect.x + t * self.rect.w
        pygame.draw.circle(surf, (200,200,50), (int(x), y), self.knob_radius)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            # click on knob?
            t = (self.value - self.min_val) / (self.max_val - self.min_val)
            kx = self.rect.x + t * self.rect.w
            ky = self.rect.centery
            if (ev.pos[0]-kx)**2 + (ev.pos[1]-ky)**2 <= self.knob_radius**2:
                self.dragging = True

        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            # drag
            rel = (ev.pos[0] - self.rect.x) / self.rect.w
            self.value = max(0, min(1, rel))*(self.max_val-self.min_val) + self.min_val

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.dragging:
            # release
            self.dragging = False
            self.callback(self.value)
            print(f"Speed set to: {self.value:.2f}")
            # insert ugot speed adjustment here

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Robot Control GUI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # speed state
    speed_mult = 1.0
    def on_speed_change(val):
        nonlocal speed_mult
        speed_mult = val

    slider = Slider((50, 50, 200, 10), 0.1, 1.0, speed_mult, on_speed_change)

    # layout constants
    CENTER = (450, 200)
    SPACING = 60
    BTN_SIZE = 50

    # define all controls in one place
    controls = [
        # label, (dx, dy), callback
        ("Up", (  0, -SPACING), forward),
        ("Down", (0, SPACING), backward),
        ("Left", (-SPACING, 0),   strafe_left),
        ("Right", ( SPACING, 0),   strafe_right),
        ("Turn L", (-SPACING, -SPACING), turn_left),
        ("Turn R", ( SPACING, -SPACING), turn_right),
    ]

    # build buttons in one comprehension
    btns = [
        Button(
            rect=(CENTER[0]+dx - BTN_SIZE//2,
                  CENTER[1]+dy - BTN_SIZE//2,
                  BTN_SIZE, BTN_SIZE),
            text=label,
            callback=fn,
            font=font
        )
        for label, (dx, dy), fn in controls
    ]

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            slider.handle_event(ev)
            for b in btns:
                b.handle_event(ev)

        screen.fill((30,30,30))
        # draw labels
        screen.blit(font.render("Speed:", True, (200,200,200)), (50,20))
        screen.blit(font.render(f"{int(speed_mult*100)}%", True, (200,200,50)), (260,40))
        screen.blit(font.render("Use arrows & turns to send commands", True, (200,200,200)), (300,20))

        # draw controls
        slider.draw(screen)
        for b in btns:
            b.draw(screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
