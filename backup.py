# THIS IS FOR BACKUP PURPOSES ONLY
# DO NOT CHANGE ANYTHING HERE
# THE ACTUAL APP IS IN app.py

print("\n" *50)
print("Welcome to Pasific Labs | Build ver. 0.3 non-stable | Copyrights Pasific Labs All Rights Reserved")
print("="*20)

print("Loading libraries... \n")
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras

print("Libraries loaded successfuly\nStarting...")

model = keras.models.load_model("model_digit.keras")
window = tk.Tk()

window.title("Digit Recognizer")
print("App started! Have fun!")

canvas = tk.Canvas(window, width=280, height=280, bg='white')
canvas.grid(row=0, column=0, pady=20)

image = Image.new("L", (280, 280), 'white')
draw = ImageDraw.Draw(image)

def draw_digit(event):
    x = event.x
    y = event.y
    r = 10 # radius si brush
    canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
    draw.ellipse((x-r, y-r, x+r, y+r), fill='black', outline='black')
    print(f"Drew at ({x}, {y})")

canvas.bind("<B1-Motion>", draw_digit)

def predict_digit():
    print("Converting image...\n")
    # Convert canvas sekarang ke array numpy
    img = np.array(image)
    print("Image converted to array:\n", img)

    # Threshold
    img = np.where(img > 200, 255, 0).astype(np.uint8)
    print("Applying threshold...\n", img)

    print("Searching for contrast...\n")
    # Cari pixel gelap
    coords = np.column_stack(np.where(img == 0))

    if coords.size == 0:
        result_label.config(text="Tidak ada angka")
        return

    print("Creating Bounding box...\n")
    # Bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    print(f"Bounding box - x: {x_min} to {x_max}, y: {y_min} to {y_max}")

    # Jika bounding box terlalu kecil (misal 1x1), kita perbesar sedikit
    if y_max - y_min < 5 or x_max - x_min < 5:
        result_label.config(text="Angka terlalu kecil")
        return

    print("Cropping image...\n")
    # Crop
    img_cropped = img[y_min:y_max+1, x_min:x_max+1]
    print("Cropped image:\n", img_cropped)

    print("Resizing image...\n")
    # Resize ke 28x28
    img_pil = Image.fromarray(img_cropped.astype(np.uint8))
    img_resized = img_pil.resize((20, 20))

    # Buat canvas putih 28x28
    canvas28 = Image.new("L", (28, 28), color=255)
    
    # hitung posisi tengah
    left = (28 - 20) // 2
    top = (28 - 20) // 2

    canvas28.paste(img_resized, (left, top))
    print("Image pasted on 28x28 canvas:\n", np.array(canvas28))

    print("Normalizing input...\n")

    # Normalisasi dan reshape
    img_final = np.array(canvas28) / 255.0
    img_final = img_final.reshape(1, 28, 28, 1)

    # Prediksi
    pred = model.predict(img_final)
    digit = np.argmax(pred)

    result_label.config(text=f"Prediksi: {digit}")
    print(pred)


def clear_canvas():
    print("Clearing canvas...")
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill="white")
    print("Canvas cleared!")

btn = tk.Button(window, text="Predict", command=predict_digit)
btn.grid(row=1, column=0)

clear_btn = tk.Button(window, text="Clear", command=clear_canvas)
clear_btn.grid(row=2, column=0)


result_label = tk.Label(window, text="Prediksi: -", font=("Helvetica", 18))
result_label.grid(row=3, column=0)
prediction_label = tk.Label(window, text="", font=("Helvetica", 14))
prediction_label.grid(row=4, column=0)

window.mainloop()

