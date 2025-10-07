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

# ---- Grad-CAM (opsional) ----
def _find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            return layer.name
    return None

def grad_cam(model, x, class_index):
    last_name = _find_last_conv(model)
    if last_name is None: 
        return None
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    cam = tf.reduce_sum(tf.multiply(pooled, conv_out), axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

def overlay_cam(pil_img, cam, alpha=0.35):
    import cv2
    img = np.array(pil_img.convert("RGB"))
    cam_r = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap((cam_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend = (heat * alpha + img * (1 - alpha)).astype(np.uint8)
    return Image.fromarray(blend)

# =========================
# UI
# =========================
st.set_page_config(page_title="Klasifikasi Gambar Serangga Hama Padi", page_icon="🐛")
st.title("Klasifikasi Gambar Serangga Hama Padi")

if not MODEL_PATH.exists():
    st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()

# Sidebar: kontrol threshold & temperature
st.sidebar.header("Pengaturan Deteksi Open-Set")
TEMP = st.sidebar.slider("Temperature (kalibrasi)", 0.8, 3.0, 1.5, 0.1)
TH_TOP1 = st.sidebar.slider("Ambang Top-1", 0.10, 0.80, 0.35, 0.01)
TH_MARGIN = st.sidebar.slider("Ambang Margin (Top1−Top2)", 0.00, 0.40, 0.10, 0.01)
TH_ENTROPY = st.sidebar.slider("Ambang Entropi", 0.80, 1.95, 1.60, 0.01)
SHOW_CAM = st.sidebar.checkbox("Tampilkan Grad-CAM", value=False)

uploaded_file = st.file_uploader("Unggah gambar…", type=["jpg", "jpeg", "png"])
model = load_model()

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocess (ikuti MobileNetV2)
    img = ImageOps.exif_transpose(image).convert("RGB").resize((224, 224))
    x = np.expand_dims(preprocess_input(np.array(img)), axis=0)

    # Prediksi mentah -> kalibrasi temperature
    raw = model(x, training=False).numpy()  # bisa logits atau probs
    probs = apply_temperature(raw, T=TEMP)

    # Keputusan open-set
    unknown, idx_sorted, top1, margin, H = decide_unknown(
        probs, th_top1=TH_TOP1, th_margin=TH_MARGIN, th_entropy=TH_ENTROPY
    )

    if unknown:
        st.markdown("### 🐞 Prediksi: **Tidak teridentifikasi**")
        st.markdown(f"Probabilitas tertinggi **{top1*100:.2f}%**; margin **{margin*100:.2f}%**; entropi **{H:.2f}**.")
        st.info("Kemungkinan bukan 7 hama pada model atau kualitas gambar kurang baik. Coba unggah ulang dengan sudut/ketajaman berbeda.")
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

    # Grad-CAM opsional
    if SHOW_CAM and not unknown:
        cam = grad_cam(model, x, class_index=idx_sorted[0])
        if cam is not None:
            st.image(overlay_cam(image, cam), caption="Grad-CAM", use_column_width=True)
        else:
            st.info("Layer konvolusi tidak terdeteksi untuk Grad-CAM.")

    st.caption(
        f"Top1={top1*100:.2f}%, Margin={margin*100:.2f}%, Entropy={H:.2f}, T={TEMP} "
        f"→ Thresholds: Top1>{TH_TOP1}, Margin>{TH_MARGIN}, Entropy<{TH_ENTROPY}"
    )
