# Importing Libraries 
import cv2 
from ugot import ugot
import numpy as np 

# Connect to UGOT and open camera
got = ugot.UGOT()
ip_add = input("What is the UGOT IP address? >")
got.initialize(ip_add)
got.open_camera()

while True: 
    # Read video frame by frame 
    frame = got.read_camera_data()

    # If no camera, breaks the program
    if not frame:
        break

    # Turns it into a numpy array
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  
    # Flip image 
    frame = cv2.flip(data, 1)
  
    # Convert BGR image to RGB image 
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            
    # Display Video and when 'q' is entered, destroy  
    # the window 
    cv2.imshow('Image', frame) 
    if cv2.waitKey(1) & 0xff == ord('q'): 
        cv2.destroyAllWindows() 
        break
