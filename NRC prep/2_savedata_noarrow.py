import cv2
import os

NOARROW_DIR = "dataset/no_arrow"

os.makedirs(NOARROW_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
count = len(os.listdir(NOARROW_DIR))  # In case some images already exist
TARGET = 100  # Number of images to save

# Progress bar configuration
BAR_WIDTH = 300
BAR_HEIGHT = 20
BAR_X = 10
BAR_Y = 50

print("Press space to save as NO ARROW. Press 'q' to quit.")

while count < TARGET:
    ret, frame = cap.read()
    if not ret:
        break

    # compute centered square ROI dynamically
    h, w, _ = frame.shape
    side = min(h, w) // 2
    cx, cy = w // 2, h // 2
    half = side // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + side, y1 + side

    # draw bounding box showing the ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Overlay count text
    text = f"NO ARROW: {count}/{TARGET}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw progress bar background and fill
    progress = int((count / TARGET) * BAR_WIDTH)
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + BAR_WIDTH, BAR_Y + BAR_HEIGHT), (50, 50, 50), -1)
    cv2.rectangle(frame, (BAR_X, BAR_Y), (BAR_X + progress, BAR_Y + BAR_HEIGHT), (0, 255, 255), -1)
    cv2.putText(frame, "NO ARROW", (BAR_X + BAR_WIDTH + 10, BAR_Y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("No-Arrow Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Save ROI on SPACE
        roi = frame[y1:y2, x1:x2]
        path = os.path.join(NOARROW_DIR, f"no_arrow_{count:03d}.jpg")
        cv2.imwrite(path, roi)
        print(f"Saved NO ARROW ROI: {path}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 