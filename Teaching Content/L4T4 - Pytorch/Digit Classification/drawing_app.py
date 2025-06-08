import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from train_model import DigitCNN

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        tk.Button(self.button_frame, text="Predict", command=self.predict_digit).pack(side='left')
        tk.Button(self.button_frame, text="Clear", command=self.clear_canvas).pack(side='left')

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.label = tk.Label(root, text="Draw a digit!", font=("Helvetica", 20))
        self.label.pack()

        self.model = DigitCNN()
        self.model.load_state_dict(torch.load("digit_model.pth", map_location=torch.device('cpu')))
        self.model.eval()

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.label.config(text="Draw a digit!")

    def predict_digit(self):
        resized = self.image.resize((28, 28), Image.LANCZOS)
        inverted = ImageOps.invert(resized)
        tensor = torch.tensor(np.array(inverted) / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)
            digit = pred.argmax(dim=1).item()
            self.label.config(text=f"Prediction: {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
