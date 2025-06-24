import cv2
import threading
import queue
import torch
import torch.nn as nn
import torch.nn.functional as F
from ugot import ugot
import numpy as np

# set up device and cuDNN autotuner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Define the model class
class ArrowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128), nn.ReLU(), nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Load the model onto the selected device
model = ArrowCNN().to(device)
model.load_state_dict(torch.load("arrow_model.pt", map_location=device))
model.eval()

# Class labels
labels = ['left', 'no_arrow', 'right']  # Match the ImageFolder class order!

# Confidence threshold 
CONFIDENCE_THRESHOLD = 0.8

# ugot connection
got = ugot.UGOT()
got.initialize('192.168.88.1')
got.open_camera()

# Start webcam 
print("Press 'q' to quit.")

# Asynchronous capture: producer thread + size-1 queue
frame_q = queue.Queue(maxsize=1)

def grab_frames():
    while True:
        raw = got.read_camera_data()
        if not raw:
            break
        # only keep the most recent frame
        if not frame_q.full():
            frame_q.put(raw)

# start the frame-grabber as a daemon so it exits with the main thread
grab_thread = threading.Thread(target=grab_frames, daemon=True)
grab_thread.start()


while True:
    try:
        raw = frame_q.get(timeout=0.1)
    except queue.Empty:
        # no frame ready yet
        continue

    # Decode JPEG buffer into OpenCV image
    nparr = np.frombuffer(raw, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Select a centered square region of interest
    h, w, _ = frame.shape
    side = min(h, w) // 2
    cx, cy = w // 2, h // 2
    crop = frame[cy - side//2 : cy + side//2, cx - side//2 : cx + side//2]

    #  Preprocess with OpenCV + NumPy
    # Grayscale & resize
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

    # Normalize to [0,1] float32
    arr = resized.astype(np.float32) / 255.0

    # Make tensor of shape [1,1,64,64] and move to device
    input_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    # Predict with softmax
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        conf = conf.item()
        pred_label = labels[pred_idx.item()]

    # Format display text
    if conf >= CONFIDENCE_THRESHOLD:
        if pred_label == 'no_arrow':
            text = "No arrow detected"
        else:
            text = f"Arrow: {pred_label} ({conf:.2f})"
    else:
        text = "Prediction: Uncertain"


    # Draw text and crop box
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.rectangle(frame, (cx - side // 2, cy - side // 2), (cx + side // 2, cy + side // 2), (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Arrow Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
