import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model .keras yang sudah dilatih
model_path = "model2/model_serangga_hama_padi.h5"
'  # Ganti dengan path model .keras kamu
model = tf.keras.models.load_model(model_path)

# Kelas/label untuk prediksi
class_names = ['Kutu beras', 'Lalat cecid', 'Penggerek batang padi', 'penggulung daun padi', 'semut', 'ulat berduri', 'wereng hijau']

# Streamlit Title
st.title("Klasifikasi Gambar Serangga Hama Padi")

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Menggunakan PIL untuk membuka gambar yang diunggah
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)

    # Mengubah gambar sesuai ukuran input model MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Preprocessing untuk MobileNetV2
    img_array = preprocess_input(img_array)

    # Menambahkan dimensi batch untuk memproses gambar
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi dengan model
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # Menampilkan hasil prediksi dan confidence
    st.markdown(f"### Prediksi: **{class_names[class_idx]}**")
    st.markdown(f"### Keyakinan: **{confidence:.2%}**")
