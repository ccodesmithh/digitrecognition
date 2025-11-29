# Proyek: Bias

Aplikasi pengenalan angka tulisan tangan yang dibuat dengan bahasa uler 🐍, TensorFlow/Keras, lengkap dengan dokumentasi line-by-line.

## Deskripsi

Aplikasi ini menggunakan model Convolutional Neural Network (CNN) untuk mengenali angka yang digambar oleh user di atas kanvas.
Disclaimer: Proyek ini hanya bertujuan untuk sarana belajar saya secara pribadi.

## File

- **`app.py`**: Aplikasi utama dengan Graphical User Interface (GUI) berbasis Tkinter. User dapat menggambar angka dan mendapatkan prediksi.
- **`training.py`**: Script untuk melatih model CNN memakai dataset MNIST. Model yang telah dilatih disimpan sebagai `model_digit.keras`.
- **`model_digit.keras`**: File model Keras yang telah dilatih.
- **`model.png`**: Diagram arsitektur model CNN.

## Library yang Dibutuhkan

Untuk menjalankan proyek ini, Anda perlu menginstal library berikut:

- tensorflow
- Pillow
- numpy
- matplotlib

Cara install:

```bash
pip install tensorflow pillow numpy matplotlib
```

## Cara Menjalankan

### 1. Menjalankan Aplikasi

Untuk menjalankan aplikasi, run script `app.py`:

```bash
python app.py
```

### 2. Melatih Model

Untuk melatih ulang model, run script `training.py`. file `model_digit.keras` yang baru akan otomati terbuat.

```bash
python training.py
```