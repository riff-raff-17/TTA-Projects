import cv2
import os

LEFT_DIR = "dataset/left"
RIGHT_DIR = "dataset/right"

os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count_left = len(os.listdir(LEFT_DIR))
count_right = len(os.listdir(RIGHT_DIR))
TARGET = 100  # Number of images to save

BAR_WIDTH = 200
BAR_HEIGHT = 20
BAR_X = 10
BAR_Y_LEFT = 50
BAR_Y_RIGHT = 80

print("Press 'a' to save as LEFT, 'd' for RIGHT. Press 'q' to quit.")

while count_left < TARGET or count_right < TARGET:
    ret, frame = cap.read()
    if not ret:
        break

    # compute centered square ROI 
    h, w, _ = frame.shape
    side = min(h, w) // 2
    cx, cy = w // 2, h // 2
    half = side // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side

    # draw bounding box showing the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Overlay count text
    text = f"LEFT: {count_left}/{TARGET}  RIGHT: {count_right}/{TARGET}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw LEFT progress bar background and fill
    progress_left = int((count_left / TARGET) * BAR_WIDTH)
    cv2.rectangle(frame, (BAR_X, BAR_Y_LEFT), (BAR_X + BAR_WIDTH, BAR_Y_LEFT + BAR_HEIGHT), (50, 50, 50), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y_LEFT), (BAR_X + progress_left, BAR_Y_LEFT + BAR_HEIGHT), (0, 255, 0), -1)
    cv2.putText(frame, "LEFT", (BAR_X + BAR_WIDTH + 10, BAR_Y_LEFT + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw RIGHT progress bar background and fill
    progress_right = int((count_right / TARGET) * BAR_WIDTH)
    cv2.rectangle(frame, (BAR_X, BAR_Y_RIGHT), (BAR_X + BAR_WIDTH, BAR_Y_RIGHT + BAR_HEIGHT), (50, 50, 50), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y_RIGHT), (BAR_X + progress_right, BAR_Y_RIGHT + BAR_HEIGHT), (255, 0, 0), -1)
    cv2.putText(frame, "RIGHT", (BAR_X + BAR_WIDTH + 10, BAR_Y_RIGHT + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show live frame with overlay
    cv2.imshow("Arrow Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and count_left < TARGET:
        # crop and save only the square ROI
        roi = frame[y1:y2, x1:x2]
        path = os.path.join(LEFT_DIR, f"left_{count_left:03d}.jpg")
        cv2.imwrite(path, roi)
        print(f"Saved LEFT arrow ROI: {path}")
        count_left += 1

    elif key == ord('d') and count_right < TARGET:
        roi = frame[y1:y2, x1:x2]
        path = os.path.join(RIGHT_DIR, f"right_{count_right:03d}.jpg")
        cv2.imwrite(path, roi)
        print(f"Saved RIGHT arrow ROI: {path}")
        count_right += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
