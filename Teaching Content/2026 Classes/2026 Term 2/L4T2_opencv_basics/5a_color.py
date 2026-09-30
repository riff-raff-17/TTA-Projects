import cv2
import numpy as np
from ugot import ugot

# --- Connect to the robot and open the camera ---
got = ugot.UGOT()
got.initialize("192.168.1.94")
got.open_camera()

print("Camera opened. Press 'q' to quit.")

while True:
    # Grab a raw frame from the robot's camera
    frame = got.read_camera_data()

    # Check 1: did we actually receive any data?
    if frame is None or len(frame) == 0:
        print("Failed to grab frame")
        break

    # Decode the raw bytes into an image we can work with
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Check 2: did the decode succeed?
    if data is None:
        print("Failed to decode frame")
        break

    # Display the live camera feed in a window
    cv2.imshow("Camera Feed", data)

    # Wait 1ms for a keypress; quit if the user presses 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
print("Camera closed. Goodbye!")