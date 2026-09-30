import cv2
import numpy as np
from ugot import ugot

# --- List of filters in order ---
# Each entry is a display name and a function that takes a frame and returns a frame
def apply_normal(frame):
    return frame


def apply_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # convert back so imshow is happy


def apply_blur(frame):
    return cv2.GaussianBlur(frame, (21, 21), 0)


def apply_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_cartoon(frame):
    # Smooth the colors, then overlay strong edges
    color = cv2.bilateralFilter(frame, 9, 250, 250)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.medianBlur(gray, 7),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        2,
    )
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_bgr)


def apply_invert(frame):
    return cv2.bitwise_not(frame)


def apply_sepia(frame):
    kernel = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    sepia = cv2.transform(frame, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def apply_pixelate(frame):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_sharpen(frame):
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    return cv2.filter2D(frame, -1, kernel)

# --- Constants ---
FILTERS = [
    ("Normal", apply_normal),
    ("Grayscale", apply_grayscale),
    ("Blur", apply_blur),
    ("Edges", apply_edges),
    ("Cartoon", apply_cartoon),
    ("Invert", apply_invert),
    ("Sepia", apply_sepia),
    ("Pixelate", apply_pixelate),
    ("Sharp", apply_sharpen)
]


def main():
    current = 0  # index into FILTERS

    got = ugot.UGOT()
    got.initialize("192.168.1.251")  # <-- change to your robot's IP
    got.open_camera()


    while True:
        frame = got.read_camera_data()
        if frame is None or len(frame) == 0:
            print("Failed to grab frame")
            break

        nparr = np.frombuffer(frame, np.uint8)
        data = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if data is None:
            print("Failed to decode frame")
            break

        # Apply the current filter
        name, fn = FILTERS[current]
        output = fn(data)

        # HUD - show current filter and controls
        cv2.putText(
            output,
            f"Filter: {name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            output,
            "n = next | p = prev | q = quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Filter Switcher", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):  # next filter
            current = (current + 1) % len(FILTERS)
        elif key == ord("p"):  # previous filter
            current = (current - 1) % len(FILTERS)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
