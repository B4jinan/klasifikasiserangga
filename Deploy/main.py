import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =======================
# Load Model
# =======================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_serangga_hama_padi.h5"

# Cek keberadaan file model
if not MODEL_PATH.exists():
    st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()

# Load model
model = tf.keras.models.load_model(str(MODEL_PATH))

# =======================
# Label Kelas
# =======================
class_names = [
    'Kutu beras', 'Lalat cecid', 'Penggerek batang padi',
    'penggulung daun padi', 'semut', 'ulat berduri', 'wereng hijau'
]

# =======================
# Judul Aplikasi
# =======================
st.title("Klasifikasi Gambar Serangga Hama Padi")

# =======================
# Upload Gambar
# =======================
uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Ubah ukuran gambar sesuai input model MobileNetV2
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Preprocessing
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Tampilkan hasil
    st.markdown(f"### 🐞 Prediksi: **{class_names[class_idx]}**")
    st.markdown(f"### 📊 Keyakinan: **{confidence:.2f}%**")
