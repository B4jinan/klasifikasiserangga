import os
import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageOps
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
# Utilitas numerik
# =========================
def softmax_on_logits(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def apply_temperature(outputs, T=1.0):
    """
    Temperature scaling di ruang logit.
    - Jika 'outputs' terlihat seperti probabilitas (softmax), gunakan log(p) sebagai logits.
    - Jika 'outputs' adalah logits mentah, gunakan langsung.
    Return: (probs, logits_T)
    """
    looks_like_probs = (
        np.all(outputs >= -1e-6) and np.all(outputs <= 1 + 1e-6)
        and np.allclose(outputs.sum(axis=1), 1.0, atol=1e-3)
    )
    logits = np.log(np.clip(outputs, 1e-12, 1.0)) if looks_like_probs else outputs
    logits_T = logits / T
    probs = softmax_on_logits(logits_T)
    return probs, logits_T

def predictive_entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(axis=1)[0])

def energy_score(logits_T, T=1.0):
    # E(x) = -T * logsumexp(logits/T) ; di sini logits_T sudah dibagi T
    m = np.max(logits_T, axis=1, keepdims=True)
    lse = m + np.log(np.exp(logits_T - m).sum(axis=1, keepdims=True))
    return float(-T * lse[0, 0])

# =========================
# Grad-CAM opsional
# =========================
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
# Ekstraksi fitur & bobot classifier (untuk cosine)
# =========================
def penultimate_and_classifier(model):
    gap_name, dense_layer = None, None
    # cari GlobalAveragePooling2D terakhir
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_name = layer.name
    # cari Dense (classifier) paling akhir
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer
            break
    if gap_name is None or dense_layer is None:
        return None, None, None
    feat_extractor = tf.keras.Model(inputs=model.input,
                                    outputs=model.get_layer(gap_name).output)
    W, b = dense_layer.get_weights()
    return feat_extractor, W, b

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Klasifikasi Gambar Serangga Hama Padi", page_icon="🐛", layout="centered")
st.title("Klasifikasi Gambar Serangga Hama Padi")

if not MODEL_PATH.exists():
    st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
    st.stop()

# Sidebar controls
st.sidebar.header("Pengaturan Deteksi Open-Set")
TEMP = st.sidebar.slider("Temperature (kalibrasi)", 0.8, 3.0, 1.5, 0.1)
TH_TOP1 = st.sidebar.slider("Ambang Top-1", 0.10, 0.99, 0.85, 0.01)
TH_MARGIN = st.sidebar.slider("Ambang Margin (Top1−Top2)", 0.00, 0.50, 0.25, 0.01)
TH_ENTROPY = st.sidebar.slider("Ambang Entropi", 0.80, 1.95, 1.30, 0.01)
TH_ENERGY = st.sidebar.slider("Ambang Energy (lebih besar → OOD)", -10.0, 5.0, -1.0, 0.1)
USE_COS = st.sidebar.checkbox("Aktifkan Cosine Similarity ke bobot kelas", value=True)
TH_COS = st.sidebar.slider("Ambang Cosine (lebih kecil → OOD)", 0.0, 1.0, 0.60, 0.01)
SHOW_CAM = st.sidebar.checkbox("Tampilkan Grad-CAM (jika teridentifikasi)", value=False)

uploaded = st.file_uploader("Unggah gambar…", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)

model = load_model()
feat_model, W_cls, _ = penultimate_and_classifier(model)

if uploaded:
    img_raw = Image.open(uploaded)
    st.image(img_raw, caption="Gambar yang diunggah", use_container_width=True)

    # Preprocess (ikuti MobileNetV2)
    img = ImageOps.exif_transpose(img_raw).convert("RGB").resize((224, 224))
    x = np.expand_dims(preprocess_input(np.array(img)), axis=0)

    # Prediksi → probs terkalibrasi & logits_T
    raw_out = model(x, training=False).numpy()
    probs, logits_T = apply_temperature(raw_out, T=TEMP)

    # TopK & metrik
    idx_sorted = np.argsort(probs[0])[::-1]
    top1, top2 = probs[0, idx_sorted[0]], probs[0, idx_sorted[1]]
    margin = top1 - top2
    H = predictive_entropy(probs)
    E = energy_score(logits_T, T=TEMP)

    # Cosine similarity ke bobot kelas (jika tersedia)
    cos = None
    if USE_COS and (feat_model is not None) and (W_cls is not None):
        feat = feat_model(x, training=False).numpy()              # (1, D)
        feat = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
        Wn = W_cls / (np.linalg.norm(W_cls, axis=0, keepdims=True) + 1e-12)
        cos = float(np.max(feat @ Wn))                             # max cosine ke kelas manapun

    # Keputusan OOD (salah satu terpenuhi → tidak teridentifikasi)
    unknown = (
        (top1 < TH_TOP1) or
        (margin < TH_MARGIN) or
        (H > TH_ENTROPY) or
        (E > TH_ENERGY) or
        ((cos is not None) and (cos < TH_COS))
    )

    if unknown:
        st.markdown("### 🐞 Prediksi: **Tidak teridentifikasi**")
        st.info("Kemungkinan gambar tidak masuk 7 kelas hama atau kualitas/objek tidak sesuai domain data latih.")
    else:
        top_idx = idx_sorted[0]
        st.markdown(f"### 🐞 Prediksi: **{CLASS_NAMES[top_idx]}**")
        st.markdown(f"### 📊 Keyakinan (kalibrasi): **{top1*100:.2f}%**")

    # Top-3 transparansi
    st.write("**Top 3 Prediksi:**")
    topk = idx_sorted[:3]
    st.table({
        "Kelas": [CLASS_NAMES[i] for i in topk],
        "Probabilitas": [f"{probs[0, i]*100:.2f}%" for i in topk]
    })

    # Grad-CAM
    if SHOW_CAM and not unknown:
        cam = grad_cam(model, x, class_index=idx_sorted[0])
        if cam is not None:
            st.image(overlay_cam(img_raw, cam), caption="Grad-CAM", use_container_width=True)
        else:
            st.info("Layer konvolusi tidak terdeteksi untuk Grad-CAM.")

    # Ringkasan metrik & ambang
    cos_txt = f", Cosine={cos:.2f}" if cos is not None else ""
    st.caption(
        f"Top1={top1*100:.2f}%, Margin={margin*100:.2f}%, Entropy={H:.2f}, Energy={E:.2f}{cos_txt} | "
        f"Thresholds → Top1>{TH_TOP1}, Margin>{TH_MARGIN}, Entropy<{TH_ENTROPY}, Energy<{TH_ENERGY}"
        + (f", Cos>{TH_COS}" if cos is not None else "")
    )
