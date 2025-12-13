import cv2
import numpy as np
from ugot import ugot
import time

def main():
    got = ugot.UGOT()
    got.initialize('192.168.1.29')
    got.open_camera()

    # Haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("Error: could not load Haar cascade for face detection.")
        return

    prev_time = time.time()
    fps = 0.0

    # --- Minimum size thresholds (tune these) ---
    MIN_FACE_WIDTH  = 80    # pixels
    MIN_FACE_HEIGHT = 80    # pixels
    # or use area instead, e.g. MIN_FACE_AREA = 8000

    while True:
        frame = got.read_camera_data()
        if frame is None or len(frame) == 0:
            print("Failed to grab frame")
            break

        # Decode JPEG bytes
        nparr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print("Failed to decode frame")
            break

        # FPS calculation
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect all faces (no minimum here except 30x30)
        detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Filter by size
        faces = []
        for (x, y, w, h) in detected:
            if w >= MIN_FACE_WIDTH and h >= MIN_FACE_HEIGHT:
                faces.append((x, y, w, h))

        # Draw accepted faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.putText(img, f"{w}x{h}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)

        # Overlay summary
        info = f"Faces: {len(faces)}   FPS: {fps:.1f}"
        cv2.putText(img, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Face Detection (min size)", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
