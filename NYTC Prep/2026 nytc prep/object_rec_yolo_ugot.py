import cv2
import numpy as np
from ugot import ugot
got = ugot.UGOT()
got.initialize('192.168.1.217')
got.open_camera()

# Load class names 
classes = []
with open("./yolo_config/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO network
net = cv2.dnn.readNet("./yolo_config/yolov3.weights", "./yolo_config/yolov3.cfg")

# Get the output layer names from YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Open webcam 

while True:
    frame_data = got.read_camera_data()
    if frame_data is not None:
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr,cv2.IMREAD_COLOR)

    height, width = frame.shape[:2]

    # Create blob from input frame
    # YOLOv3 usually uses 416x416
    blob = cv2.dnn.blobFromImage(frame, 
                                 scalefactor=1/255.0, 
                                 size=(416, 416), 
                                 swapRB=True, 
                                 crop=False)
    net.setInput(blob)

    # Run forward pass (get detections)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:] # class scores
            class_id = np.argmax(scores) # best class
            confidence = scores[class_id] # its confidence

            if confidence > 0.5: # threshold
                # detection[0:4] = center_x, center_y, w, h (relative to image size)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Convert to top-left x, y
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Draw bounding boxes and labels
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} | {center_x, center_y}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("YOLO Real-Time Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
