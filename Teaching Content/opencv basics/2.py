import cv2
import numpy as np
from ugot import ugot

got = ugot.UGOT()
got.initialize('192.168.88.1')
got.open_camera()

def main():

    while True:
        frame = got.read_camera_data()
        if not frame:
            print("Failed to grab frame")
            break

        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr,cv2.IMREAD_COLOR)

        # Draw a red rectangle
        cv2.rectangle(data, (50, 50), (200, 200), (0, 0, 255), 2)

        # Draw text
        cv2.putText(data, "Hello OpenCV", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Webcam Feed", data)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
