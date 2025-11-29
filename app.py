print("\n" *50)
print("Welcome to Pasific Labs | Build ver. 0.2 Stable | Copyrights Pasific Labs All Rights Reserved")
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
    img = image.resize((28, 28))

    # Convert to numpy
    img = np.array(img)

    # THRESHOLD: tebalkan garis → semua pixel > 200 jadi putih, sisanya hitam
    img = np.where(img > 200, 255, 0)

    # Normalisasi
    img = img / 255.0

    img = img.reshape(1, 28, 28, 1)
    print("Predicting digit...\n")
    pred = model.predict(img)
    digit = np.argmax(pred)
    print(f"Predicted digit: {digit} with confidence {pred[0][digit]:.4f}")
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

