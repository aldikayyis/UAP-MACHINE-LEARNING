import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Muat model yang telah dilatih
try:
    model = load_model(r'G:\ALDI\codingan\Machine Learning\Aksara Jawa\aksara_jawa_model.h5')
except OSError as e:
    st.error(f"Error saat memuat model: {e}")
    st.stop()

# Judul aplikasi
st.title("Aplikasi Pengenalan Aksara Jawa")
st.write("Unggah gambar aksara Jawa untuk memprediksi kelas aksara.")

# Daftar kelas aksara Jawa (sesuaikan dengan model)
class_names = ["ha", "na", "ca", "ra", "ka", "da", "ta", "sa", "wa", "la", 
               "ma", "ga", "ba", "tha", "nga", "pa", "ja", "ya", "nya", "pha"]  # Sesuaikan jumlah kelas

# Fungsi untuk memproses gambar
def predict_image(img):
    img = img.resize((224, 224))  # Ukuran gambar yang digunakan model
    img = np.array(img)  # Konversi ke numpy array
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
    return img

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar aksara Jawa", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    img = img.convert('RGB')  # Konversi ke RGB
    st.image(img, caption="Gambar yang diunggah.", use_column_width=True)

    # Prediksi gambar
    img_array = predict_image(img)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Ambil indeks prediksi dengan probabilitas tertinggi
    
    # Tampilkan hasil prediksi
    st.write(f"Prediksi mentah: {predictions}")
    st.write(f"Indeks prediksi: {predicted_class}")
    if predicted_class < len(class_names):
        st.write(f"Prediksi Kelas: {class_names[predicted_class]}")
    else:
        st.error("Indeks prediksi berada di luar daftar kelas. Periksa model atau daftar kelas.")
