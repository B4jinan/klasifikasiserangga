import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# Konfigurasi dasar
# =========================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_serangga_hama_padi.h5"

CLASS_NAMES = [
    'Kutu beras', 'Lalat cecid', 'penggulung daun padi',
    'semut', 'snail', 'ulat berduri', 'wereng hijau'
]

# =========================
# Parameter kalibrasi & open-set (tanpa sidebar)
# Silakan ubah nilainya di sini jika perlu penyesuaian.
# =========================
TEMP = 1.5        # temperature scaling untuk kalibrasi probabilitas
TH_TOP1 = 0.35    # ambang minimal probabilitas kelas tertinggi
TH_MARGIN = 0.10  # ambang minimal selisih (Top1 - Top2)
TH_ENTROPY = 1.60 # ambang maksimal entropi (semakin besar = model makin ragu)
RULE_TOP1_HARD = 0.25   # jika Top1 < nilai ini -> paksa "Tidak teridentifikasi"
RULE_STD_FLAT = 0.05    # jika sebaran probabilitas terlalu datar (std < ini) -> "Tidak teridentifikasi"

# =========================
# Utilitas
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))

def softmax_on_logits(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def apply_temperature(outputs, T=1.0):
    """
    Menerapkan temperature scaling di ruang logit.
    - Jika 'outputs' sudah probabilitas (softmax), pakai log(p).
    - Jika 'outputs' masih logits, langsung dipakai.
    """
    looks_like_probs = (
        np.all(outputs >= -1e-6) and np.all(outputs <= 1 + 1e-6)
        and np.allclose(outputs.sum(axis=1), 1.0, atol=1e-3)
    )
    logits = np.log(np.clip(outputs, 1e-12, 1.0)) if looks_like_probs else outputs
    return softmax_on_logits(logits / T)

def predictive_entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1)[0])

def decide_unknown(probs, th_top1, th_margin, th_entropy):
    idx_sorted = np.argsort(probs[0])[::-1]
    top1, top2 = probs[0, idx_sorted[0]], probs[0, idx_sorted[1]]
    margin = top1 - top2
    H = predictive_entropy(probs)
    is_unknown = (top1 < th_top1) or (margin < th_margin) or (H > th_entropy)
    return is_unknown, idx_sorted, top1, margin, H

# =========================
# UI
# =========================
st.set_page_config(page_title="Klasifikasi Gambar Serangga Hama Padi", page_icon="🐛")
st.title("Klasifikasi Gambar Serangga Hama Padi")

if not MODEL_PATH.exists():
    st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader("Unggah gambar…", type=["jpg", "jpeg", "png"])
model = load_model()

# =========================
# Prediksi + Open-Set (tanpa sidebar)
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocess (ikuti MobileNetV2)
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    x = np.expand_dims(preprocess_input(np.array(img)), axis=0)

    # Prediksi mentah -> kalibrasi temperature
    raw = model(x, training=False).numpy()  # bisa logits atau probs
    probs = apply_temperature(raw, T=TEMP)

    # Keputusan open-set dasar (berdasar top1, margin, entropi)
    unknown, idx_sorted, top1, margin, H = decide_unknown(
        probs, th_top1=TH_TOP1, th_margin=TH_MARGIN, th_entropy=TH_ENTROPY
    )

    # ======= Aturan tambahan yang lebih ketat untuk gambar "random" =======
    avg_prob = float(probs.mean())
    std_prob = float(probs.std())
    force_unknown = (top1 < RULE_TOP1_HARD) or (std_prob < RULE_STD_FLAT) or unknown
    # ======================================================================

    if force_unknown:
        st.markdown("### 🐞 Prediksi: **Tidak teridentifikasi**")
        st.markdown(
            f"Probabilitas tertinggi **{top1*100:.2f}%**; margin **{margin*100:.2f}%**; "
            f"entropi **{H:.2f}**; sebaran std **{std_prob:.3f}**."
        )
        st.info(
            "Gambar tidak cocok dengan 7 kelas hama dalam model. "
            "Kemungkinan gambar bukan serangga pada dataset atau kualitas gambar kurang baik."
        )
    else:
        top_idx = idx_sorted[0]
        st.markdown(f"### 🐞 Prediksi: **{CLASS_NAMES[top_idx]}**")
        st.markdown(f"### 📊 Keyakinan (kalibrasi): **{top1*100:.2f}%**")

    # Top-3 transparansi
    topk = idx_sorted[:3]
    st.write("**Top 3 Prediksi:**")
    st.table({
        "Kelas": [CLASS_NAMES[i] for i in topk],
        "Probabilitas": [f"{probs[0, i]*100:.2f}%" for i in topk]
    })

    st.caption(
        f"Kalibrasi T={TEMP} | Ambang: Top1>{TH_TOP1}, Margin>{TH_MARGIN}, Entropy<{TH_ENTROPY} | "
        f"Rule tambahan: Top1<{RULE_TOP1_HARD} atau Std<{RULE_STD_FLAT} ⇒ Tidak teridentifikasi | "
        f"Ringkasan: Top1={top1*100:.2f}%, Margin={margin*100:.2f}%, Entropy={H:.2f}, "
        f"MeanProb={avg_prob:.3f}, StdProb={std_prob:.3f}"
    )
