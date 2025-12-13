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

        cv2.imshow("Webcam Feed", data)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
