print("\n" *100)
print("Welcome to Pasific Labs | Model ver. 1.4 | Build ver. 0.5 Stable | Copyrights Pasific Labs All Rights Reserved")
print("="*20)

print("Loading libraries... \n")


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
from tensorflow import keras

print("Libraries loaded successfully\nStarting...")
print("Loading Model...")

try:
    model = keras.models.load_model("model_digit.keras")
    print("Model loaded successfully! Have Fun!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pasific Labs | Digit Recognizer v0.5")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Style config
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', background="#f0f0f0", font=('Helvetica', 12))
        self.style.configure('Header.TLabel', font=('Helvetica', 18, 'bold'))

        # Layout Utama di sini coeg
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_panel = ttk.Frame(self.main_frame)
        self.top_panel.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(self.top_panel, text="Draw a digit using the mouse! Then click Predict.", style='Header.TLabel').pack(pady=(0, 10))
        ttk.Label(self.top_panel, text="(C) Copyrights 2024 Pasific Labs All Rights Reserved", font=("Helvetica", 10, "italic")).pack(pady=(0, 1))
        ttk.Label(self.top_panel, text="Programmed by Yudha Prasetiya / Codesmith @ Pasific Labs", font=("Helvetica", 10, "italic")).pack(pady=(0, 10))

        # Panel kiri - Drawing Area
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

        ttk.Label(self.left_panel, text="Draw a Digit (0-9)", style='Header.TLabel').pack(pady=(0, 10))
        
        self.canvas_width = 300
        self.canvas_height = 300
        self.canvas = tk.Canvas(self.left_panel, width=self.canvas_width, height=self.canvas_height, bg='white', 
                                highlightthickness=1, highlightbackground="#ccc")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw_digit)
        
        # Image object
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        self.btn_frame = ttk.Frame(self.left_panel)
        self.btn_frame.pack(pady=20)
        
        self.predict_btn = ttk.Button(self.btn_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(self.btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Panel kanan - Results & Debug
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(self.right_panel, text="Prediction Result", style='Header.TLabel').pack(pady=(0, 10))
        
        self.result_label = ttk.Label(self.right_panel, text="-", font=("Helvetica", 72, "bold"))
        self.result_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(self.right_panel, text="Confidence: -", font=("Helvetica", 14))
        self.confidence_label.pack()

        ttk.Separator(self.right_panel, orient='horizontal').pack(fill='x', pady=20)

        # Debug View
        ttk.Label(self.right_panel, text="Model Input (Processed)", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))
        self.debug_canvas = tk.Canvas(self.right_panel, width=140, height=140, bg='black')
        self.debug_canvas.pack()
        
        self.debug_info = ttk.Label(self.right_panel, text="Inverted -> Centered -> Resized (28x28)", font=("Helvetica", 10, "italic"))
        self.debug_info.pack(pady=5)
        

    def draw_digit(self, event):
        x, y = event.x, event.y
        r = 12 # Brush radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse((x-r, y-r, x+r, y+r), fill='black', outline='black')
        print("Drawig at", x, y)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, self.canvas_width, self.canvas_height), fill="white")
        self.result_label.config(text="-")
        self.confidence_label.config(text="Confidence: -")
        self.debug_canvas.delete("all")

    def preprocess_image(self):
        # 1. Invert warna (Bg putih -> bg HIDEUNG)
        # Angka di MNIST itu hitam di bg putih
        img_inverted = ImageOps.invert(self.image)
        
        # 2. Cari bounding box
        bbox = img_inverted.getbbox()
        
        if bbox is None:
            return None # Canvas is empty

        # 3. Cropping ke bounding box
        img_cropped = img_inverted.crop(bbox)
        
        # 4. Resize ke 20x20
        # Kita mau fit digit ke dalam 20x20 box, dan center di 28x28 image
        target_size = 20
        width, height = img_cropped.size
        
        scale = target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img_resized = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 5. Buat canvas 28x28 hitam dan paste digit yang sudah di resize di tengah
        final_img = Image.new("L", (28, 28), 0) # 0 is black
        
        # Kalkulasi posisi tengah
        paste_x = (28 - new_width) // 2
        paste_y = (28 - new_height) // 2
        
        final_img.paste(img_resized, (paste_x, paste_y))
        
        return final_img

    def predict_digit(self):
        processed_img = self.preprocess_image()
        print("Image processed:", processed_img)
        
        if processed_img is None:
            self.result_label.config(text="?")
            self.confidence_label.config(text="You need to draw a digit first!")
            return

        # Update Debug View
        # Resize ke 140x140 biar enak di liat
        debug_view = processed_img.resize((140, 140), Image.Resampling.NEAREST)
        self.debug_photo = ImageTk.PhotoImage(debug_view)
        self.debug_canvas.create_image(70, 70, image=self.debug_photo)

        # Prepare buat si model
        img_array = np.array(processed_img)
        img_array = img_array / 255.0 # Normalisasi
        img_array = img_array.reshape(1, 28, 28, 1)
        
        # Predict
        print("Predicting...")
        pred = model.predict(img_array)
        digit = np.argmax(pred)
        confidence = pred[0][digit]
        
        self.result_label.config(text=str(digit))
        self.confidence_label.config(text=f"Confidence: {confidence:.2%}")
        print(f"Predicted: {digit} ({confidence:.4f})")
        print(pred)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
