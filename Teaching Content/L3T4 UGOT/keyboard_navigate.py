# Importing Libraries 
import cv2 
import numpy as np 
import pygame
from ugot import ugot

got = ugot.UGOT()

# ——— robot API stubs ———
def forward():
    print("Robot → forward")
    got.mecanum_move_speed(0, 30)

def backward():
    print("Robot → backward")
    got.mecanum_move_speed(1, 30)

def turn_left():
    print("Robot → left")
    got.mecanum_turn_speed(2, 30)

def turn_right():
    print("Robot → right")
    got.mecanum_turn_speed(3, 30)

def stop():
    print("Robot → stop")
    got.mecanum_stop()

def scan_qr():
    print(got.get_apriltag_total_info())

# ————————————————————————————————————————————————

def main():
    # — Initialize UGOT camera —
    ip_add = '192.168.1.' + input("What are the LAST 3 numbers of the UGOT IP address? > ")
    got.initialize(ip_add)
    got.open_camera()
    got.load_models(['apriltag_qrcode'])


    # — Initialize pygame for display + input —
    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    running = True

    while running:
        # 1) Grab frame from UGOT
        frame = got.read_camera_data()
        if not frame:
            break
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        flipped = cv2.flip(data, 1)
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        # 2) On first frame, create pygame window of matching size
        h, w = frame_rgb.shape[:2]
        if screen is None:
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("UGOT Camera + Robot Control")

        # 3) Convert to pygame Surface and display
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
        screen.blit(surface, (0,0))
        pygame.display.flip()

        # 4) Handle pygame events
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

            # Key pressed down → start movement
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_UP:
                    forward()
                elif evt.key == pygame.K_DOWN:
                    backward()
                elif evt.key == pygame.K_LEFT:
                    turn_left()
                elif evt.key == pygame.K_RIGHT:
                    turn_right()
                elif evt.key == pygame.K_w:
                    scan_qr()
                elif evt.key == pygame.K_q:
                    running = False

            # Key released → stop movement
            elif evt.type == pygame.KEYUP:
                if evt.key in (pygame.K_UP, pygame.K_DOWN,
                               pygame.K_LEFT, pygame.K_RIGHT):
                    stop()

        # 5) Cap your frame‐rate
        clock.tick(30)

    # Clean up
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
