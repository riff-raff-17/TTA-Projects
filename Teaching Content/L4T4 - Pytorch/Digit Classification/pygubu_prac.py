'''Call pygubu with pygubu-designer'''

import tkinter as tk
import pygubu
from PIL import Image, ImageDraw, ImageOps
import torch
import numpy as np
from train_model import DigitCNN

class DrawingApp:
    def __init__(self, root):
        # Load .ui
        self.builder = pygubu.Builder()
        self.builder.add_from_file('digitapp.ui')
        # This creates the Frame (id="frame") as a child of root
        self.mainframe = self.builder.get_object('frame', root)
        root.title("Draw a Digit")

        # Grab widgets to interact with
        self.canvas = self.builder.get_object('canvas')
        self.btn_predict = self.builder.get_object('btn_predict')
        self.btn_clear = self.builder.get_object('btn_clear')
        self.lbl_status = self.builder.get_object('lbl_status')

        # Backing PIL image to record strokes
        self.image = Image.new("L", (280, 280), color=255)
        self.draw  = ImageDraw.Draw(self.image)

        # Load trained model
        self.model = DigitCNN()
        self.model.load_state_dict(
            torch.load("digit_model.pth", map_location='cpu')
        )
        self.model.eval()

        # Wire signals to methods
        self.builder.connect_callbacks(self)

    # Called on canvas <B1-Motion>
    def on_paint(self, event):
        x, y = event.x, event.y
        r = 8
        # draw on screen
        self.canvas.create_oval(x-r, y-r, x+r, y+r,
                                fill='black', outline='black')
        # draw in PIL
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    # Called when “Clear” is clicked
    def on_clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill=255)
        self.lbl_status.config(text="Draw a digit!")

    # Called when “Predict” is clicked
    def on_predict_digit(self):
        # match MNIST preprocessing
        img28 = self.image.resize((28, 28), Image.LANCZOS)
        img28 = ImageOps.invert(img28)
        arr = np.array(img28) / 255.0
        tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)
            digit = out.argmax(dim=1).item()

        self.lbl_status.config(text=f"Prediction: {digit}")

if __name__ == '__main__':
    root = tk.Tk()
    app  = DrawingApp(root)
    root.mainloop()
