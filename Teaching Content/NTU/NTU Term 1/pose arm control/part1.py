"""
PART 1 — Raw webcam feed
========================
Goal: get a mirrored webcam window open. Every subsequent part builds on this loop.

What you'll see: your webcam feed, mirrored like a mirror.
Press Q to quit.
"""
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # mirror so left/right feel natural
    cv2.imshow("Part 1 — Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()