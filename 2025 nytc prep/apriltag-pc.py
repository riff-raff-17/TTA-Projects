from ugot import ugot
import cv2
import numpy as np

got = ugot.UGOT()

def main():
    # --------------------
    # 1. Robot initialization
    # --------------------
    got.initialize('192.168.1.29')            # connect to robot
    got.open_camera()                          # start camera
    got.load_models(['apriltag_qrcode'])       # enable tag/QR detection

    # --------------------
    # 2. Seek-and-align loop
    # --------------------
    while True:
        frame = got.read_camera_data()
        if not frame:
            break
        frame = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)

        tags = got.get_apriltag_total_info()
        if tags:
            # use first detection
            # cx: center x position
            # distance: distance in meters to target
            _, cx, cy, h, w, _, distance, *_ = tags[0]
            if distance < 0.22:
                break
            # steer toward center
            if cx > 340:
                cv2.putText(frame, 'Go right',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif cx < 300:
                cv2.putText(frame, 'Go left',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Go straight',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            # Display stats
            cv2.putText(frame, f'Distance: {distance} | Center X: {cx} / 320', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2  = int(cx + w / 2)
            y2 = int(cy + h / 2)

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        else:
            distance = "NaN"
            cx = "NaN"
            cv2.putText(frame, 'No tag detected: stop',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        # Display stats
        cv2.putText(frame, f'Distance: {distance} | Center X: {cx} / 320', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --------------------
    # 3. Grasp & finish
    # --------------------
    cv2.destroyAllWindows()
    got.play_audio_tts("Finished", 0, True)    # announce

if __name__ == "__main__":
    main()
