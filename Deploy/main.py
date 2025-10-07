import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# Konfigurasi
# =========================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_serangga_hama_padi.h5"

CLASS_NAMES = [
    'Kutu beras', 'Lalat cecid', 'penggulung daun padi',
    'semut', 'snail', 'ulat berduri', 'wereng hijau'
]

# Kalibrasi & ambang open-set
TEMP = 2.0          # temperature scaling (1.5–3.0; sesuaikan di validasi)
TOPK = 3
TH_TOP1 = 0.50      # jika top-1 < 0.55 → tidak teridentifikasi
TH_MARGIN = 0.10    # jika (top1-top2) < 0.15 → tidak teridentifikasi
TH_ENTROPY = 1.45   # jika entropi > 1.30 (ln(7)≈1.95) → tidak teridentifikasi

# =========================
# Utilitas
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))

def softmax_temperature(logits, T=1.0):
    z = logits / T
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def predictive_entropy(p):
    # p shape (1, C)
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1)[0])

def decide_unknown(probs, th_top1=TH_TOP1, th_margin=TH_MARGIN, th_entropy=TH_ENTROPY):
    # ambil top-2
    idx_sorted = np.argsort(probs[0])[::-1]
    top1, top2 = probs[0, idx_sorted[0]], probs[0, idx_sorted[1]]
    margin = top1 - top2
    H = predictive_entropy(probs)
    is_unknown = (top1 < th_top1) or (margin < th_margin) or (H > th_entropy)
    return is_unknown, idx_sorted[0], top1, margin, H

# =========================
# UI
# =========================
st.set_page_config(page_title="Klasifikasi Gambar Serangga Hama Padi", page_icon="🐛")
st.title("Klasifikasi Gambar Serangga Hama Padi")

if not MODEL_PATH.exists():
    st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])

model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocessing (ikuti MobileNetV2)
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    x = np.expand_dims(preprocess_input(np.array(img)), axis=0)

    # Prediksi → kalibrasi suhu
    logits = model(x, training=False).numpy()
    probs = softmax_temperature(logits, T=TEMP)

    # Keputusan open-set
    unknown, top_idx, top1, margin, H = decide_unknown(probs)

    if unknown:
        st.markdown("### 🐞 Prediksi: **Tidak teridentifikasi**")
        st.markdown(f"Probabilitas tertinggi hanya **{top1*100:.2f}%**; margin **{margin*100:.2f}%**; entropi **{H:.2f}**.")
        st.info("Gambar kemungkinan **bukan 7 hama pada model** atau kualitas gambar kurang baik. Coba unggah ulang dengan sudut/ketajaman berbeda.")
    else:
        st.markdown(f"### 🐞 Prediksi: **{CLASS_NAMES[top_idx]}**")
        st.markdown(f"### 📊 Keyakinan (kalibrasi): **{top1*100:.2f}%**")

    # Tampilkan Top-3 untuk transparansi
    idx = np.argsort(probs[0])[::-1][:TOPK]
    st.write("**Top 3 Prediksi:**")
    st.table({
        "Kelas": [CLASS_NAMES[i] for i in idx],
        "Probabilitas": [f"{probs[0,i]*100:.2f}%" for i in idx]
    })

    # Catatan ambang (supaya mudah tuning bila perlu)
    st.caption(
        f"Ambang open-set → Top1<{TH_TOP1}, Margin<{TH_MARGIN}, Entropy>{TH_ENTROPY}. "
        f"(T= {TEMP}) — sesuaikan berdasarkan validasi."
    )
