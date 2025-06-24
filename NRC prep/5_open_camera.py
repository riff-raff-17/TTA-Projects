import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

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

# Load the model
model = ArrowCNN()
model.load_state_dict(torch.load("arrow_model.pt", map_location="cpu"))
model.eval()

# Class labels
labels = ['left', 'no_arrow', 'right']  # Match the ImageFolder class order!

# Transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Confidence threshold 
CONFIDENCE_THRESHOLD = 0.8

# Start webcam 
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Select a centered square region of interest
    h, w, _ = frame.shape
    side = min(h, w) // 2
    cx, cy = w // 2, h // 2
    crop = frame[cy - side//2 : cy + side//2, cx - side//2 : cx + side//2]

    # Convert to PIL and transform
    pil_img = Image.fromarray(crop)
    input_tensor = transform(pil_img).unsqueeze(0)

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
    cv2.rectangle(frame, (cx - side//2, cy - side//2), (cx + side//2, cy + side//2), (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Arrow Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
