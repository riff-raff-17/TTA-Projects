import sys
import cv2
import pygame
import numpy as np
from ugot import ugot

# gripper state
gripper_open = False

# movement & helper callbacks
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

def toggle_gripper():
    global gripper_open
    gripper_open = not gripper_open
    print('gripper opened' if gripper_open else 'gripper closed')

def raise_arm():
    print('arm raised')

def lower_arm():
    print('arm lowered')

class Button:
    __slots__ = ('rect','text','callback','font')
    def __init__(self, rect, text, callback, font):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.font = font

    def draw(self, surf):
        pygame.draw.rect(surf, (70,70,70), self.rect)
        lbl = self.font.render(self.text, True, (255,255,255))
        surf.blit(lbl, lbl.get_rect(center=self.rect.center))

    def handle_event(self, ev):
        if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
        and self.rect.collidepoint(ev.pos)):
            self.callback()


class Slider:
    __slots__ = ('rect','min_val','max_val','value','callback','knob_radius','dragging')
    def __init__(self, rect, mn, mx, init, callback):
        self.rect = pygame.Rect(rect)
        self.min_val, self.max_val = mn, mx
        self.value = init
        self.callback = callback
        self.knob_radius = 8
        self.dragging = False

    def draw(self, surf):
        y = self.rect.centery
        pygame.draw.line(surf, (150,150,150),
                         (self.rect.x, y),
                         (self.rect.right, y), 4)
        t = (self.value - self.min_val) / (self.max_val - self.min_val)
        x = self.rect.x + t * self.rect.w
        pygame.draw.circle(surf, (200,200,50),
                           (int(x), y),
                           self.knob_radius)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            t = (self.value - self.min_val) / (self.max_val - self.min_val)
            kx = self.rect.x + t * self.rect.w
            ky = self.rect.centery
            if (ev.pos[0]-kx)**2 + (ev.pos[1]-ky)**2 <= self.knob_radius**2:
                self.dragging = True

        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            rel = (ev.pos[0] - self.rect.x) / self.rect.w
            rel = max(0.0, min(1.0, rel))
            self.value = self.min_val + rel*(self.max_val-self.min_val)

        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.dragging:
            self.dragging = False
            self.callback(self.value)
            print(f"Speed set to: {self.value:.2f}")


def main():
    # set up camera
    got = ugot.UGOT()
    ip_add = input("What is the UGOT IP address? > ")
    got.initialize(ip_add)
    got.open_camera()
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

    # set up pygame
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("Robot Control GUI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # speed slider
    speed_mult = 1.0
    def on_speed_change(val):
        nonlocal speed_mult
        speed_mult = val

    slider = Slider((50, 50, 200, 10), 0.1, 1.0, speed_mult, on_speed_change)

    # layout constants
    CENTER = (450, 200)
    SPACING = 60
    BTN_SIZE = 50

    # directional + turn buttons
    controls = [
        ("Up", (0, -SPACING), forward),
        ("Down", (0, SPACING), backward),
        ("Left", (-SPACING, 0), strafe_left),
        ("Right", ( SPACING, 0), strafe_right),
        ("Turn L", (-SPACING, -SPACING), turn_left),
        ("Turn R", ( SPACING, -SPACING), turn_right),
    ]

    btns = [
        Button(
            rect=(
                CENTER[0] + dx - BTN_SIZE//2,
                CENTER[1] + dy - BTN_SIZE//2,
                BTN_SIZE, BTN_SIZE
            ),
            text=label,
            callback=fn,
            font=font
        )
        for label, (dx, dy), fn in controls
    ]

    # helper buttons under the slider, stacked vertically
    helper_defs = [
        ("Arm Up", raise_arm),
        ("Gripper", toggle_gripper),
        ("Arm Down", lower_arm),
    ]
    helper_h = 30
    helper_sp = 10
    start_y = slider.rect.bottom + 20  # 20px below slider
    for i, (label, fn) in enumerate(helper_defs):
        x    = slider.rect.x
        y    = start_y + i*(helper_h + helper_sp)
        w    = slider.rect.w
        btns.append(Button((x, y, w, helper_h), label, fn, font))

    # main loop
    running = True
    while running:
        # handle events
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            slider.handle_event(ev)
            for b in btns:
                b.handle_event(ev)

        # draw GUI
        screen.fill((30,30,30))
        screen.blit(font.render("Speed:", True, (200,200,200)), (50,20))
        screen.blit(font.render(f"{int(speed_mult*100)}%", True, (200,200,50)), (260,40))
        screen.blit(font.render("Use arrows & turns to send commands", True, (200,200,200)), (300,20))
        slider.draw(screen)
        for b in btns:
            b.draw(screen)
        pygame.display.flip()

        # grab & show camera frame
        frame_data = got.read_camera_data()
        if frame_data is None:
            continue
        nparr = np.frombuffer(frame_data, np.uint8)
        data  = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cam_f = cv2.flip(data, 1)
        cv2.imshow("Camera", cam_f)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        clock.tick(60)

    # teardown
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
