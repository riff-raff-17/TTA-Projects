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

    # Convert data into a numpy array
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Flip image 
    frame = cv2.flip(data, 1)

    # Draw boxes around faces
    face_info = got.get_face_recognition_total_info()
    for face in face_info:
        name, center_x, center_y, height, width, area = face
        
        # Flip center_x because of the frame flip
        fixed_center_x = frame.shape[1] - center_x
        
        top_left = (int(fixed_center_x - width / 2), int(center_y - height / 2))
        bottom_right = (int(fixed_center_x + width / 2), int(center_y + height / 2))
        
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, name, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Show video frame and exit on 'q'
    cv2.imshow('Image', frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) & 0xff == ord('w'):
        face_name = input('What name? >')
        got.face_recognition_add_name(face_name)

    if cv2.waitKey(1) & 0xff == ord('e'):
        print(got.face_recognition_get_all_names())

    if cv2.waitKey(1) & 0xff == ord('r'):
        print(got.get_face_recognition_total_info())

    if cv2.waitKey(1) & 0xff == ord('t'):
        for name in got.face_recognition_get_all_names():
            got.face_recognition_delete_name(name)
        
        print('Done!')

cv2.destroyAllWindows()