import cv2              # cv2: for camera vision
import numpy as np      # numpy: for data transformation
import pygame           # pygame: for driving your robot
from ugot import ugot   # ugot: robot commands

got = ugot.UGOT()

# ——— robot api stubs ———
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

def talk():
    user_input = input("What to say? >")
    got.play_audio_tts(user_input, 1, True)

# ————————————————————————————————————————————————

def main():
    # Initialize UGOT camera
    ip_add = '192.168.1.' + input("What are the LAST numbers of the UGOT IP address? > ")
    got.initialize(ip_add)
    got.open_camera()
    got.load_models(['apriltag_qrcode'])

    # Initialize pygame for display and input
    pygame.init()
    screen = None
    clock = pygame.time.Clock()
    running = True
    detect = False

    while running:
        # 1) Grab frame from UGOT
        frame = got.read_camera_data()
        if not frame:
            break
        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        flipped = cv2.flip(data, 1)

        # 1.5) Draw bounding boxes around QR codes
        if detect:
            image_width = data.shape[1]
            qr_info = got.get_apriltag_total_info()

            for tag in qr_info:
                qrcode, cx, cy, h, w, *rest = tag
                flipped_cx = image_width - cx
                x1 = int(flipped_cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(flipped_cx + w / 2)
                y2 = int(cy + h / 2)

                # Draw rectangle and label
                cv2.rectangle(flipped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(flipped, str(qrcode), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX,
                            1, (0, 255, 0), 2)

        cv2.putText(flipped, "Detecting: " + str(detect), (10, 40), cv2.FONT_HERSHEY_DUPLEX,
                            1, (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        # 2) On first frame, create a pygame window of matching size
        h, w = frame_rgb.shape[:2]
        if screen is None:
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("UGOT Camera + Robot Control")

        # 3) Convert to pygame Surface and display
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
        screen.blit(surface, (0, 0))
        pygame.display.flip()

        # 4) Handle pygame events
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

            # Key pressed down -> start moving
            elif evt.type == pygame.KEYDOWN:
                if evt.key == pygame.K_w:
                    forward()
                elif evt.key == pygame.K_s:
                    backward()
                elif evt.key == pygame.K_a:
                    turn_left()
                elif evt.key == pygame.K_d:
                    turn_right()
                elif evt.key == pygame.K_t:
                    talk()
                elif evt.key == pygame.K_q:
                    detect = not detect
                
            elif evt.type == pygame.KEYUP:
                if evt.key in (pygame.K_w, pygame.K_s, 
                               pygame.K_a, pygame.K_d):
                    stop()

        # 5) Cap the frame rate at 30 FPS
        clock.tick(30)
    
    # Cleanup
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()