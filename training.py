# Welcome to Pasific Labs | 28 Nov 2025 | 17.32
# Project: Bias
from datetime import datetime
current_datetime = datetime.now()
current_time = current_datetime.time()

print("\n" * 50)
print("Welcome to Pasific Labs |", current_datetime.date(), "|", current_time)
print("="*20)
print("Loading libraries...\n")

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
print("Project: Bias")
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
print("="*20)

print("\n" * 2)

print("Loading MNIST dataset...\n")

# x_train = 60.000 gambar training dalam bentuk array angka | y_train = 60.000 label dari mnist
# x_test = 10.000 gambar testing | y_test = 10.000 label dari mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

print("Normalizing data...\n")
# Normalisasi
x_train = x_train / 255.0   # 0-255 -> 0-1
x_test = x_test / 255.0     # 0-255 -> 0-1

print(x_train, x_test, sep="\n")
print(y_train, y_test, sep="\n")

print("Reshaping data... \n")
print(y_train[0:10])  # Tampilkan 10 label pertama
print(y_test[0:10])   # Tampilkan 10 label pertama
print(x_train[0:10])
print(x_test[0:10])

# Reshape                (-1 = banyaknya gambar (dihitung oleh python). 28, 28 = ukuran gambar. 1 = channel grayscale.) / (batch_size, width, height, channels)
x_train = x_train.reshape(-1, 28, 28, 1) # 28x28 -> 28x28x1
x_test = x_test.reshape(-1, 28, 28, 1) # 28x28 -> 28x28x1


print("x_train reshaped:", x_train.shape)
print("y_train reshaped:", y_train.shape)
print("x_test reshaped:", x_test.shape)
print("y_test reshaped:", y_test.shape, "\n")

print("Creating model... \n")
# Membuat model
# Sequential = berurutan, susun layer CNN dari awal sampe akhir
model = models.Sequential()

# Convolutional layer
print("Adding layers to the model...\n")
# .add() = menambahkan layer ke model
# Conv2D = Convolutional layer. Conv2D (filters, kernel_size, activation='relu'), activation = fungsi aktivasi untuk menyalakan sinyal penting
# input_shape = ukuran gambar di awal
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))

print("Adding pooling layer...\n")
# Pooling layer
# MaxPooling2D = Pooling layer. MaxPooling2D(pool_size=(2,2))
model.add(layers.MaxPooling2D(2,2))

print("Adding second convolutional layer...\n")
# Convolutional layer ke2 Tidak perlu input shape karena layer sebelumnya sudah ada
# Butuh lebih bbanyak filter buat nangkep pola yang lebih kompleks
model.add(layers.Conv2D(64, (3,3), activation='relu'))

print("Adding second pooling layer...\n")
# Pooling layer ke 2
model.add(layers.MaxPooling2D(2,2))

print("Adding flatten layer...\n")
# Flatten
# Flatten = mengubah gambar menjadi array
model.add(layers.Flatten())

print("Adding fully connected layer...\n")
# Fully Connected Layer
# Dense = Fully Connected Layer . Dense(Jumlah Neuron, activation='relu')
model.add(layers.Dense(64, activation='relu'))

print("Adding output layer...\n")
# Output Layer
model.add(layers.Dense(10, activation='softmax')) # Softmax adalah fungsi aktivasi untuk membuat output menjadi probabilitas

print("Compiling model...\n")
# Compile modelnya
model.compile(
    optimizer='adam',                           # Optimizer Adam (algoritma untuk mengupdate bobot)
    loss='sparse_categorical_crossentropy',     # Loss function untuk multi-class classification (0-9)
    metrics=['accuracy']                        # Mengukur akurasi selama training dan testing
)

model.summary()  # Tampilkan ringkasan model

print("Training model...\n")
model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test)) # Latih model dengan data training selama 12 epoch
print("Training completed.\n", "saving model...\n")
model.save("model_digit.keras")  # Simpan model ke file
print("Model saved as model_digit.keras\n")
model.summary()




# # TESTING AREA

# index = 5
# image = x_test[index]  # Ambil gambar ke 0 dari data testing

# plt.imshow(image.reshape(28,28), cmap='gray')  # Tampilkan gambar
# plt.title(f"Label Asli: {y_test[index]}")  # Tampilkan label asli
# plt.show()

# prediction = model.predict(image.reshape(1, 28, 28, 1))  # Prediksi gambar (reshaped ke bentuk yang sesuai)
# predicted_label = np.argmax(prediction)  # Ambil label dengan probabilitas tertinggi

# print("Prediksi model: ", predicted_label)  # Tampilkan prediksi model
# print("Probabilitas: ", prediction)  # Tampilkan probabilitas


# Uncomment this section leter
# img = Image.open("digit.png").convert("L")  # Buka gambar dan konversi ke grayscale
# img = img.resize((28, 28))  # Ubah ukuran gambar ke 28

# img = np.array(img)  # Konversi gambar menjadi array
# img = img / 255.0 # Normalisasi

# img = img.reshape(1, 28, 28, 1)  # Reshape ke bentuk yang sesuai untuk model

# prediction = model.predict(img) # Prediksi gambar
# prediction_label = np.argmax(prediction)  # Ambil label dengan probabilitas tertinggi

# print("Prediksi model: ", prediction_label)  # Tampilkan prediksi model
# print("probabilitas: ", prediction)