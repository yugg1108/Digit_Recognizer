# Note : Any new thing you save will overwrite the previous one.

import tkinter as tk
from PIL import Image, ImageDraw

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # We make the canvas 20x larger than 28x28 so humans can actually draw on it
        self.canvas_width = 560
        self.canvas_height = 560
        self.brush_size = 12
        
        # 1. Setup the visible GUI Canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack(pady=10)
        
        # Bind the mouse drag event to the paint function
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # 2. Setup a hidden PIL image in memory that perfectly matches the visible canvas
        # "L" means 8-bit pixels, black and white (grayscale)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # 3. Setup UI Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        btn_save = tk.Button(btn_frame, text="Save (28x28)", command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=10)
        
        btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        btn_clear.pack(side=tk.LEFT, padx=10)

    def paint(self, event):
        # Calculate the brush stroke coordinates based on mouse position
        x1 = event.x - self.brush_size
        y1 = event.y - self.brush_size
        x2 = event.x + self.brush_size
        y2 = event.y + self.brush_size
        
        # Draw on the visible tkinter canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        
        # Draw simultaneously on the hidden PIL image
        self.draw.ellipse([x1, y1, x2, y2], fill="white")

    def clear_canvas(self):
        # Wipe the visible canvas
        self.canvas.delete("all")
        # Wipe the hidden PIL image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)

    def save_image(self):
        # Resize the 560x560 memory image down to 28x28 using high-quality anti-aliasing
        small_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Save it to your computer
        filename = "digit_28x28.png"
        small_image.save(filename)
        print(f"Success! Image saved as {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()