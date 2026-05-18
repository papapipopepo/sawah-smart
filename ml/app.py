import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io, base64, os, time, json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── API Clients ───────────────────────────────────────────────────────────────
from openai import OpenAI
from langfuse import observe

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Langfuse v4 — inisialisasi otomatis via env vars, gunakan @observe decorator
os.environ.setdefault("LANGFUSE_PUBLIC_KEY",  os.getenv("LANGFUSE_PUBLIC_KEY", ""))
os.environ.setdefault("LANGFUSE_SECRET_KEY",  os.getenv("LANGFUSE_SECRET_KEY", ""))
os.environ.setdefault("LANGFUSE_HOST",        os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"))

# ── Vegetation Index Functions ────────────────────────────────────────────────

def green_mask(rgb, threshold=5, min_green=40):
    R = rgb[:, :, 0].astype(np.int32)
    G = rgb[:, :, 1].astype(np.int32)
    B = rgb[:, :, 2].astype(np.int32)
    return (G > R + threshold) & (G > B + threshold) & (G > min_green)

def compute_VARI(rgb, eps=1e-6):
    r = rgb.astype(np.float32)
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return np.clip((G - R) / (G + R - B + eps), -1.0, 1.0)

def compute_ExG(rgb):
    r = rgb.astype(np.float32)
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (2 * G - R - B) / 255.0

def compute_ExGR(rgb):
    r = rgb.astype(np.float32) / 255.0
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (2 * G - R - B) - (1.4 * R - G)

def compute_GLI(rgb, eps=1e-6):
    r = rgb.astype(np.float32)
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (2 * G - R - B) / (2 * G + R + B + eps)

def compute_MGRVI(rgb, eps=1e-6):
    r = rgb.astype(np.float32)
    R, G = r[:, :, 0], r[:, :, 1]
    return (G**2 - R**2) / (G**2 + R**2 + eps)

def compute_RGBVI(rgb, eps=1e-6):
    r = rgb.astype(np.float32)
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (G**2 - B * R) / (G**2 + B * R + eps)

def compute_NGRDI(rgb, eps=1e-6):
    r = rgb.astype(np.float32)
    R, G = r[:, :, 0], r[:, :, 1]
    return (G - R) / (G + R + eps)

VI_REGISTRY = {
    "VARI": compute_VARI,
    "ExG": compute_ExG,
    "ExGR": compute_ExGR,
    "GLI": compute_GLI,
    "MGRVI": compute_MGRVI,
    "RGBVI": compute_RGBVI,
    "NGRDI": compute_NGRDI,
}

VI_CMAPS = {
    "VARI": "RdYlGn", "ExG": "Greens", "ExGR": "PiYG",
    "GLI": "YlGn", "MGRVI": "RdYlGn", "RGBVI": "Greens", "NGRDI": "RdYlGn",
}

def extract_vi_stats(v):
    v = v.flatten()
    return {
        "mean":      float(np.mean(v)),
        "std":       float(np.std(v)),
        "median":    float(np.median(v)),
        "p10":       float(np.percentile(v, 10)),
        "p25":       float(np.percentile(v, 25)),
        "p75":       float(np.percentile(v, 75)),
        "p90":       float(np.percentile(v, 90)),
        "frac_gt0":  float(np.mean(v > 0)),
        "frac_gt01": float(np.mean(v > 0.1)),
        "iqr":       float(np.percentile(v, 75) - np.percentile(v, 25)),
    }

def compute_all_vi(rgb):
    mask = green_mask(rgb)
    stats, maps = {}, {}
    for name, fn in VI_REGISTRY.items():
        vi_map = fn(rgb)
        vi_use = vi_map.flatten()  # seluruh gambar agar piksel kuning/mature ikut terhitung
        stats[name] = extract_vi_stats(vi_use)
        maps[name]  = vi_map
    return stats, maps, mask, float(mask.mean())

# ── ML Model ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml_model():
    import joblib
    from skimage.feature import graycomatrix, graycoprops  # noqa: ensure importable

    # Coba patch model dulu (lebih akurat untuk per-patch inference)
    patch_files = [f for f in os.listdir(".") if f.startswith("best_patch_") and f.endswith(".joblib")]
    if patch_files and os.path.exists("label_encoder_patch.joblib"):
        clf = joblib.load(patch_files[0])
        le  = joblib.load("label_encoder_patch.joblib")
        vi_name = next((vi for vi in VI_REGISTRY if f"best_patch_{vi}_" in patch_files[0]), "ExGR")
        return clf, le, vi_name, "patch"

    # Fallback: model whole-image
    model_files = [f for f in os.listdir(".") if f.startswith("best_3class_") and f.endswith(".joblib")]
    enc_path = "label_encoder_3class.joblib"
    if model_files and os.path.exists(enc_path):
        clf = joblib.load(model_files[0])
        le  = joblib.load(enc_path)
        vi_name = next(
            (vi for vi in VI_REGISTRY if f"best_3class_{vi}_" in model_files[0]),
            None
        )
        return clf, le, vi_name, "whole-image"
    return None, None, None, None


# ── Debug: cek file model yang ada di folder ─────────────────────────────────

def get_model_info():
    """Scan folder dan kembalikan info file model yang ditemukan + dipakai."""
    info = {
        "svm_patch_files":  sorted(f for f in os.listdir(".")
                                   if f.startswith("best_patch_") and f.endswith(".joblib")),
        "svm_whole_files":  sorted(f for f in os.listdir(".")
                                   if f.startswith("best_3class_") and f.endswith(".joblib")),
        "cnn_files":        sorted(f for f in os.listdir(".")
                                   if f.endswith(".keras") and "cnn" in f.lower() and "hybrid" not in f.lower()),
        "hybrid_files":     sorted(f for f in os.listdir(".")
                                   if f.startswith("best_hybrid_cnn_") and f.endswith(".keras")),
    }
    # Tentukan yang AKAN dipakai (sesuai logic loader)
    info["svm_used"] = (
        info["svm_patch_files"][0] if info["svm_patch_files"] and os.path.exists("label_encoder_patch.joblib")
        else (info["svm_whole_files"][0] if info["svm_whole_files"] and os.path.exists("label_encoder_3class.joblib") else None)
    )
    info["cnn_used"]    = info["cnn_files"][0]    if info["cnn_files"]    else None
    info["hybrid_used"] = info["hybrid_files"][0] if info["hybrid_files"] else None
    return info


# ── CNN Model (Deep Learning) ─────────────────────────────────────────────────

CNN_IMG_SIZE   = 224
CNN_CLASS_ORDER = ["Vegetative", "Generative", "Mature"]

@st.cache_resource
def load_cnn_model():
    """Load CNN .keras dari folder kerja. Return (model, classes) atau (None, None)."""
    try:
        import tensorflow as tf  # lazy import — TF berat
    except ImportError:
        return None, None

    # Cari file CNN murni — EXCLUDE hybrid (yang punya "hybrid" di nama)
    cnn_files = [f for f in os.listdir(".")
                 if f.endswith(".keras") and "cnn" in f.lower() and "hybrid" not in f.lower()]
    if not cnn_files:
        return None, None

    try:
        model = tf.keras.models.load_model(cnn_files[0])
    except Exception as e:
        st.warning(f"Gagal load CNN model: {e}")
        return None, None

    classes_path = "cnn_classes.txt"
    classes = CNN_CLASS_ORDER
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f if line.strip()]
    return model, classes


@st.cache_resource
def load_hybrid_model():
    """Load Hybrid CNN+VI model. Return (model, vi_name, classes) atau (None, None, None)."""
    try:
        import tensorflow as tf
    except ImportError:
        return None, None, None

    hybrid_files = [f for f in os.listdir(".")
                    if f.startswith("best_hybrid_cnn_") and f.endswith(".keras")]
    if not hybrid_files:
        return None, None, None

    fname = hybrid_files[0]
    # Extract VI name: best_hybrid_cnn_ExG.keras -> "ExG"
    vi_name = fname.replace("best_hybrid_cnn_", "").replace(".keras", "")

    try:
        model = tf.keras.models.load_model(fname)
    except Exception as e:
        st.warning(f"Gagal load Hybrid model: {e}")
        return None, None, None

    if vi_name not in VI_REGISTRY:
        st.warning(f"VI '{vi_name}' tidak ada di registry. Default ExGR.")
        vi_name = "ExGR"

    return model, vi_name, CNN_CLASS_ORDER


def predict_hybrid(rgb, model, vi_name, classes):
    """Prediksi pakai Hybrid CNN+VI. Return (label, confidence, distribusi_dict).

    Hybrid model butuh 2 input: image (224x224x3) + 18 VI features.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from skimage.feature import graycomatrix, graycoprops

    # Branch 1: image input
    img = cv2.resize(rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = preprocess_input(img.astype(np.float32))
    arr = np.expand_dims(arr, axis=0)

    # Branch 2: VI features (18 = 10 stats + 8 GLCM)
    vi_map = VI_REGISTRY[vi_name](rgb)
    stats  = extract_vi_stats(vi_map.flatten())
    stat_names = list(stats.keys())

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = graycoprops(glcm, prop).flatten()
        glcm_feats += [float(np.mean(vals)), float(np.std(vals))]

    vi_features = np.array(
        [stats[s] for s in stat_names] + glcm_feats, dtype=np.float32
    ).reshape(1, -1)  # shape (1, 18)

    proba   = model.predict([arr, vi_features], verbose=0)[0]
    pred_id = int(np.argmax(proba))
    return classes[pred_id], float(proba[pred_id]), dict(zip(classes, proba.tolist()))


def predict_cnn(rgb, model, classes):
    """Prediksi pakai CNN. Return (label, confidence, distribusi_dict)."""
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = cv2.resize(rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = preprocess_input(img.astype(np.float32))
    arr = np.expand_dims(arr, axis=0)

    proba   = model.predict(arr, verbose=0)[0]
    pred_id = int(np.argmax(proba))
    return classes[pred_id], float(proba[pred_id]), dict(zip(classes, proba.tolist()))


def predict_ml_patches(rgb, clf, le, vi_name, grid=(3, 3)):
    """Bagi gambar jadi grid NxN, prediksi tiap patch, agregasi soft voting."""
    from collections import Counter
    h, w       = rgb.shape[:2]
    rows, cols = grid
    ph, pw     = h // rows, w // cols
    color_map  = {"Vegetative": "#22c55e", "Generative": "#eab308", "Mature": "#ef4444"}
    use_proba  = hasattr(clf, "predict_proba")

    results    = []
    all_probas = []  # kumpulkan probabilitas tiap patch untuk soft voting

    for r in range(rows):
        row_res = []
        for c in range(cols):
            y1, y2 = r * ph, (r + 1) * ph
            x1, x2 = c * pw, (c + 1) * pw
            patch   = rgb[y1:y2, x1:x2]
            label   = "N/A"
            try:
                feats = _extract_feats_for_patch(patch, clf, vi_name)
                if feats is not None:
                    arr = np.array([feats], dtype=np.float32)
                    if use_proba:
                        proba = clf.predict_proba(arr)[0]  # [P(kelas0), P(kelas1), ...]
                        all_probas.append(proba)
                        pred_id = np.argmax(proba)
                    else:
                        pred_id = clf.predict(arr)[0]
                    label = le.inverse_transform([pred_id])[0]
            except Exception:
                pass
            row_res.append({"label": label, "color": color_map.get(label, "#6b7280"),
                            "y1": y1, "y2": y2, "x1": x1, "x2": x2})
        results.append(row_res)

    all_labels = [c["label"] for row in results for c in row if c["label"] != "N/A"]
    counts     = Counter(all_labels)

    if not all_labels:
        return results, "N/A", counts

    # Highest-stage wins dari hard label patch — agronomis: 1 patch Mature = tanaman Mature
    STAGE_RANK = {"Mature": 2, "Generative": 1, "Vegetative": 0}
    dominant = max(set(all_labels), key=lambda l: STAGE_RANK.get(l, -1))

    return results, dominant, counts


def _extract_feats_for_patch(patch, clf, vi_name):
    """Ekstraksi fitur dari satu patch — dipakai oleh predict_ml_patches."""
    from skimage.feature import graycomatrix, graycoprops

    mask   = green_mask(patch)
    vi_map = VI_REGISTRY[vi_name](patch)

    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = {}
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = graycoprops(glcm, prop).flatten()
        glcm_feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        glcm_feats[f"glcm_{prop}_std"]  = float(np.std(vals))

    glcm_names = [f"glcm_{p}_{s}" for p in ["contrast", "correlation", "energy", "homogeneity"]
                  for s in ["mean", "std"]]

    try:
        first_step = clf[0] if hasattr(clf, "__getitem__") else clf
        n_feats = getattr(first_step, "n_features_in_", 18)
    except Exception:
        n_feats = 18

    stats     = extract_vi_stats(vi_map.flatten())
    stat_vals = list(stats.values())
    glcm_vals = [glcm_feats[g] for g in glcm_names]

    if n_feats >= 19:
        return [float(mask.mean())] + stat_vals + glcm_vals
    return stat_vals + glcm_vals


def render_patch_heatmap(rgb, patch_results, alpha=0.40):
    """Overlay warna patch di atas gambar asli."""
    import matplotlib.patches as mpatches
    color_rgb = {
        "Vegetative": (34/255, 197/255, 94/255),
        "Generative": (234/255, 179/255,  8/255),
        "Mature":     (239/255,  68/255, 68/255),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(rgb); axes[0].set_title("Gambar Asli", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(rgb)
    for row in patch_results:
        for cell in row:
            c = color_rgb.get(cell["label"], (0.4, 0.4, 0.4))
            rect = plt.Rectangle((cell["x1"], cell["y1"]),
                                 cell["x2"] - cell["x1"], cell["y2"] - cell["y1"],
                                 linewidth=2, edgecolor="white", facecolor=c, alpha=alpha)
            axes[1].add_patch(rect)
            cx = (cell["x1"] + cell["x2"]) / 2
            cy = (cell["y1"] + cell["y2"]) / 2
            axes[1].text(cx, cy, cell["label"][:3], color="white", ha="center", va="center",
                         fontsize=9, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor=c, alpha=0.75, edgecolor="none"))
    legend = [mpatches.Patch(color=v, label=k) for k, v in
              {"Vegetative":"#22c55e","Generative":"#eab308","Mature":"#ef4444"}.items()]
    axes[1].legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.85)
    axes[1].set_title("Patch Heatmap", fontweight="bold"); axes[1].axis("off")
    plt.tight_layout()
    return fig


def predict_ml(rgb, clf, le, vi_name):
    """Ekstraksi fitur identik dengan training notebook.
    18 fitur: VI stats full-image (10) + GLCM (8)
    19 fitur: green_fraction (1) + VI stats full-image (10) + GLCM (8)
    """
    from skimage.feature import graycomatrix, graycoprops

    mask   = green_mask(rgb)
    vi_map = VI_REGISTRY[vi_name](rgb)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    glcm_feats = {}
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = graycoprops(glcm, prop).flatten()
        glcm_feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        glcm_feats[f"glcm_{prop}_std"]  = float(np.std(vals))

    glcm_names = [f"glcm_{p}_{s}" for p in ["contrast","correlation","energy","homogeneity"]
                  for s in ["mean","std"]]

    # Deteksi jumlah fitur — aman untuk Pipeline maupun model tunggal
    try:
        first_step = clf[0] if hasattr(clf, "__getitem__") else clf
        n_feats = getattr(first_step, "n_features_in_", 18)
    except Exception:
        n_feats = 18

    # Selalu pakai full-image (flatten), sama dengan training di notebook
    stats      = extract_vi_stats(vi_map.flatten())
    stat_names = list(stats.keys())
    stat_vals  = [stats[s] for s in stat_names]
    glcm_vals  = [glcm_feats[g] for g in glcm_names]

    if n_feats >= 19:
        feats = [float(mask.mean())] + stat_vals + glcm_vals
    else:
        feats = stat_vals + glcm_vals

    arr = np.array([feats], dtype=np.float32)
    pred_id = clf.predict(arr)[0]
    label = le.inverse_transform([pred_id])[0]

    # Confidence — pakai predict_proba kalau tersedia
    confidence = 1.0
    if hasattr(clf, "predict_proba"):
        try:
            confidence = float(clf.predict_proba(arr)[0].max())
        except Exception:
            pass
    return label, confidence

# ── LLM Vision ────────────────────────────────────────────────────────────────

def encode_image(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@observe(name="llm-vision-classify")
def classify_with_llm(pil_image, vi_stats, green_frac=0.0, model="gpt-4o"):
    vi_text = "\n".join(
        f"- {vi}: mean={s['mean']:.3f}, std={s['std']:.3f}, "
        f"median={s['median']:.3f}, frac_gt0={s['frac_gt0']:.2f}"
        for vi, s in vi_stats.items()
    )

    system_prompt = (
        "Kamu adalah ahli agronomi spesialis tanaman padi. "
        "Klasifikasikan fase pertumbuhan padi dari gambar menjadi salah satu:\n"
        "- Vegetative: fase vegetatif, daun hijau segar, belum ada malai\n"
        "- Generative: fase generatif, malai sudah muncul, bulir berkembang, warna mulai kekuningan\n"
        "- Mature: fase matang, bulir kuning keemasan penuh, malai menunduk, SIAP PANEN\n"
        "- BukanPadi: gambar BUKAN tanaman padi (mis. orang, hewan, objek, tanaman lain) "
        "atau tidak bisa dianalisis (terlalu gelap/blur/abstrak)\n\n"
        "Panduan green_fraction: >50% = Vegetative, 25-50% = Generative, <25% = Mature\n"
        "PENTING: kalau gambar jelas bukan tanaman padi, WAJIB jawab 'BukanPadi' "
        "— jangan paksa klasifikasi.\n\n"
        "Balas HANYA dalam format JSON:\n"
        '{"kelas":"Vegetative|Generative|Mature|BukanPadi","keyakinan":"Tinggi|Sedang|Rendah",'
        '"alasan":"penjelasan visual singkat","rekomendasi":"saran tindakan untuk petani"}'
    )
    user_prompt = (
        f"Analisis gambar padi ini dan klasifikasikan fase pertumbuhannya.\n\n"
        f"Green Fraction (proporsi piksel hijau): {green_frac:.2%}\n"
        f"Data Indeks Vegetasi RGB dari gambar:\n{vi_text}\n\n"
        "Berikan jawaban JSON."
    )

    t0 = time.time()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(pil_image)}"
                }},
            ]},
        ],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.1,
    )
    latency = time.time() - t0
    result  = json.loads(resp.choices[0].message.content)
    return result, latency, resp.usage

# ── Visualization ─────────────────────────────────────────────────────────────

def render_vi_maps(rgb, vi_maps, mask):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    axes[0].imshow(rgb)
    axes[0].set_title("RGB Asli", fontweight="bold", fontsize=12)
    axes[0].axis("off")

    for i, (name, vi_map) in enumerate(vi_maps.items()):
        masked = vi_map.astype(float).copy()
        masked[~mask] = np.nan
        im = axes[i + 1].imshow(masked, cmap=VI_CMAPS[name])
        axes[i + 1].set_title(name, fontweight="bold", fontsize=12)
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deteksi Panen Padi",
    page_icon="🌾",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🌾 Sistem Deteksi Kesiapan Panen Padi")
st.caption("Indeks Vegetasi RGB + LLM Vision (GPT-4o) | IoT ESP32-CAM | Thesis")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")
    llm_model      = st.selectbox("Model LLM", ["gpt-4o", "gpt-4o-mini"])
    show_vi_maps   = st.toggle("Tampilkan Peta VI",       value=True)
    show_vi_stats  = st.toggle("Tampilkan Statistik VI",  value=True)
    use_ml_compare = st.toggle("Bandingkan dengan ML",    value=True)
    use_cnn        = st.toggle("Pakai CNN (Deep Learning)", value=False)
    use_hybrid     = st.toggle("Pakai Hybrid CNN+VI",         value=False)
    use_patch      = st.toggle("Patch-based Analysis",   value=True)
    grid_size      = st.select_slider("Grid Patch", options=[2, 3, 4, 5], value=3) if use_patch else 3
    st.divider()
    st.markdown("**Keterangan Kelas**")
    st.markdown("🟢 **Vegetative** — masih tumbuh")
    st.markdown("🟡 **Generative** — bulir berkembang")
    st.markdown("🔴 **Mature** — SIAP PANEN")

    st.divider()
    with st.expander("🔍 Debug: Model Info", expanded=False):
        _info = get_model_info()

        st.markdown("**SVM** _(untuk perbandingan ML + patch analysis)_")
        if _info["svm_used"]:
            badge = "🟢 PATCH" if _info["svm_used"].startswith("best_patch_") else "🟡 WHOLE-IMAGE"
            st.markdown(f"{badge}  `{_info['svm_used']}`")
        else:
            st.markdown("🔴 Tidak ada SVM model")

        if len(_info["svm_patch_files"]) > 1 or len(_info["svm_whole_files"]) > 1:
            with st.container():
                st.caption("File yang TERSEDIA (urut alfabetis, yang pertama dipakai):")
                for f in _info["svm_patch_files"] + _info["svm_whole_files"]:
                    marker = "👉 " if f == _info["svm_used"] else "    "
                    st.caption(f"{marker}`{f}`")

        st.markdown("---")
        st.markdown("**CNN** _(toggle 'Pakai CNN')_")
        if _info["cnn_used"]:
            st.markdown(f"🟢 `{_info['cnn_used']}`")
        else:
            st.markdown("🔴 Tidak ada CNN model")

        st.markdown("---")
        st.markdown("**Hybrid CNN+VI** _(toggle 'Pakai Hybrid')_")
        if _info["hybrid_used"]:
            _vi = _info["hybrid_used"].replace("best_hybrid_cnn_", "").replace(".keras", "")
            st.markdown(f"🟢 `{_info['hybrid_used']}` — VI: **{_vi}**")
        else:
            st.markdown("🔴 Tidak ada Hybrid model")

        if len(_info["hybrid_files"]) > 1:
            st.caption("Hybrid file yang tersedia:")
            for f in _info["hybrid_files"]:
                marker = "👉 " if f == _info["hybrid_used"] else "    "
                st.caption(f"{marker}`{f}`")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_main, tab_history = st.tabs(["📷 Klasifikasi", "📋 Riwayat Prediksi"])

# ════════════════════════════════════════════════════════════════════════════
with tab_main:
    uploaded = st.file_uploader(
        "Upload foto padi (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        help="Foto dari ESP32-CAM atau kamera lainnya",
    )

    if uploaded is None:
        st.info("Upload foto padi untuk memulai analisis.")
        st.stop()

    pil_img = Image.open(uploaded).convert("RGB")
    rgb     = np.array(pil_img)

    with st.spinner("Menghitung indeks vegetasi..."):
        vi_stats, vi_maps, mask, green_frac = compute_all_vi(rgb)

    clf, le, vi_name_ml, model_type = load_ml_model()

    # ── Baris 1: Gambar + Hasil LLM ──────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("📷 Gambar Input")
        st.image(pil_img, use_container_width=True)
        st.caption(f"**{uploaded.name}** — {rgb.shape[1]}×{rgb.shape[0]} px")

    with col_result:
        st.subheader(f"🤖 Klasifikasi LLM ({llm_model})")

        with st.spinner(f"Menganalisis dengan {llm_model}..."):
            llm_result, latency, usage = classify_with_llm(
                pil_img, vi_stats, green_frac=green_frac, model=llm_model
            )

        kelas       = llm_result.get("kelas", "Unknown")
        keyakinan   = llm_result.get("keyakinan", "-")
        alasan      = llm_result.get("alasan", "")
        rekomendasi = llm_result.get("rekomendasi", "")
        badge = {"Vegetative": "🟢", "Generative": "🟡", "Mature": "🔴",
                 "BukanPadi": "🚫"}.get(kelas, "⚪")

        if kelas == "BukanPadi":
            st.error(f"## {badge} BUKAN GAMBAR PADI")
            st.warning(
                "⚠️ LLM mendeteksi gambar ini bukan tanaman padi. "
                "Hasil model ML (SVM/CNN/Hybrid) di bawah ini **kemungkinan besar tidak valid** "
                "— model dipaksa pilih salah satu fase karena dilatih hanya untuk gambar padi."
            )
        elif kelas == "Mature":
            st.success(f"## {badge} {kelas.upper()} — SIAP PANEN!")
            st.balloons()
        elif kelas == "Generative":
            st.warning(f"## {badge} {kelas} — Belum siap panen")
        elif kelas == "Vegetative":
            st.info(f"## {badge} {kelas} — Masih tumbuh")
        else:
            st.info(f"## {badge} {kelas}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Keyakinan",  keyakinan)
        c2.metric("Latency",    f"{latency:.2f}s")
        c3.metric("Token",      usage.total_tokens)

        with st.expander("📝 Penjelasan LLM", expanded=True):
            st.markdown(f"**Analisis:** {alasan}")
            st.markdown(f"**Rekomendasi:** {rekomendasi}")

    # ── Perbandingan ML ───────────────────────────────────────────────────────
    if use_ml_compare:
        st.divider()
        st.subheader("⚖️ Perbandingan LLM vs Model ML")

        # Confidence threshold untuk deteksi gambar non-padi
        CONFIDENCE_THRESHOLD = 0.55  # di bawah ini → "Unknown"

        ml_pred, ml_conf = "-", 0.0
        if clf is not None:
            with st.spinner("Prediksi dengan model ML..."):
                ml_pred, ml_conf = predict_ml(rgb, clf, le, vi_name_ml)
            if ml_conf < CONFIDENCE_THRESHOLD:
                ml_pred = f"Unknown ({ml_pred}?)"
        else:
            st.warning(
                "Model SVM tidak ditemukan. "
                "Copy file `best_3class_*.joblib` dan `label_encoder_3class.joblib` "
                "ke folder yang sama dengan `app.py`, lalu refresh."
            )

        # CNN prediction (opsional)
        cnn_pred, cnn_conf, cnn_dist = "-", 0.0, {}
        if use_cnn:
            cnn_model, cnn_classes = load_cnn_model()
            if cnn_model is not None:
                with st.spinner("Prediksi dengan CNN..."):
                    cnn_pred, cnn_conf, cnn_dist = predict_cnn(rgb, cnn_model, cnn_classes)
                if cnn_conf < CONFIDENCE_THRESHOLD:
                    cnn_pred = f"Unknown ({cnn_pred}?)"
            else:
                st.info("Model CNN tidak ditemukan. Letakkan `best_cnn_*.keras` di folder app.py.")

        # Hybrid prediction (opsional)
        hyb_pred, hyb_conf, hyb_dist, hyb_vi = "-", 0.0, {}, None
        if use_hybrid:
            hyb_model, hyb_vi, hyb_classes = load_hybrid_model()
            if hyb_model is not None:
                with st.spinner(f"Prediksi dengan Hybrid CNN+{hyb_vi}..."):
                    hyb_pred, hyb_conf, hyb_dist = predict_hybrid(rgb, hyb_model, hyb_vi, hyb_classes)
                if hyb_conf < CONFIDENCE_THRESHOLD:
                    hyb_pred = f"Unknown ({hyb_pred}?)"
            else:
                st.info("Model Hybrid tidak ditemukan. Letakkan `best_hybrid_cnn_*.keras` di folder app.py.")

        # Banner global: deteksi kemungkinan bukan padi (consensus)
        ml_confidences = [c for c in [ml_conf, cnn_conf if use_cnn else None,
                                       hyb_conf if use_hybrid else None]
                          if c is not None and c > 0]
        avg_ml_conf = np.mean(ml_confidences) if ml_confidences else 1.0
        n_unknown   = sum(1 for p in [ml_pred, cnn_pred, hyb_pred]
                          if isinstance(p, str) and p.startswith("Unknown"))

        if kelas == "BukanPadi" and avg_ml_conf < CONFIDENCE_THRESHOLD:
            st.error(
                "🚫 **Konsensus: ini BUKAN gambar padi.** "
                "LLM dan model ML sama-sama tidak yakin — kemungkinan besar gambar di luar domain."
            )
        elif kelas == "BukanPadi":
            st.warning(
                "⚠️ LLM bilang **bukan padi**, tapi model ML memberikan prediksi dengan confidence cukup tinggi. "
                "Bisa jadi gambar ambigu — periksa manual."
            )
        elif n_unknown >= 2:
            st.warning(
                f"⚠️ **{n_unknown} dari {len(ml_confidences)} model ML tidak yakin** (confidence < {CONFIDENCE_THRESHOLD:.0%}). "
                "Mungkin gambar bukan padi atau di luar distribusi training."
            )
        elif avg_ml_conf < CONFIDENCE_THRESHOLD:
            st.info(
                f"ℹ️ Rata-rata confidence ML rendah ({avg_ml_conf:.0%}). Gambar mungkin sulit diklasifikasi."
            )

        # Display perbandingan — tergantung kombinasi toggle
        cols_data = [("LLM Vision", kelas, f"Keyakinan: {keyakinan}")]
        if ml_pred != "-":
            cols_data.append(("SVM", ml_pred, f"VI: {vi_name_ml} · {ml_conf*100:.0f}%"))
        if use_cnn and cnn_pred != "-":
            cols_data.append(("CNN", cnn_pred, f"Conf: {cnn_conf*100:.1f}%"))
        if use_hybrid and hyb_pred != "-":
            cols_data.append((f"Hybrid CNN+{hyb_vi}", hyb_pred, f"Conf: {hyb_conf*100:.1f}%"))

        if len(cols_data) > 1:
            cols = st.columns(len(cols_data))
            for col, (label, val, delta) in zip(cols, cols_data):
                col.metric(label, val, delta=delta)

            # Caption file model yang dipakai
            _info_pred = get_model_info()
            cap_parts = ["**Model dipakai:**"]
            cap_parts.append(f"LLM=`{llm_model}`")
            if ml_pred != "-" and _info_pred["svm_used"]:
                cap_parts.append(f"SVM=`{_info_pred['svm_used']}`")
            if use_cnn and cnn_pred != "-" and _info_pred["cnn_used"]:
                cap_parts.append(f"CNN=`{_info_pred['cnn_used']}`")
            if use_hybrid and hyb_pred != "-" and _info_pred["hybrid_used"]:
                cap_parts.append(f"Hybrid=`{_info_pred['hybrid_used']}`")
            st.caption(" · ".join(cap_parts))

            # Distribusi probabilitas (CNN + Hybrid)
            if (use_cnn and cnn_dist) or (use_hybrid and hyb_dist):
                with st.expander("📊 Distribusi probabilitas", expanded=False):
                    if use_cnn and cnn_dist:
                        st.markdown("**CNN (RGB only)**")
                        for cls, p in cnn_dist.items():
                            st.progress(p, text=f"{cls}: {p*100:.2f}%")
                    if use_hybrid and hyb_dist:
                        st.markdown(f"**Hybrid CNN + {hyb_vi}**")
                        for cls, p in hyb_dist.items():
                            st.progress(p, text=f"{cls}: {p*100:.2f}%")

    # ── Patch-based Analysis ──────────────────────────────────────────────────
    if use_patch and clf is not None:
        st.divider()
        st.subheader("🔲 Patch-based Analysis")
        st.caption(f"Gambar dibagi {grid_size}×{grid_size} = {grid_size**2} patch — tiap patch diklasifikasi secara independen")

        with st.spinner("Menganalisis patch..."):
            patch_results, patch_majority, patch_counts = predict_ml_patches(
                rgb, clf, le, vi_name_ml, grid=(grid_size, grid_size)
            )

        # Metrics
        total_patch = sum(patch_counts.values())
        c1, c2, c3, c4 = st.columns(4)
        badge_map = {"Vegetative": "🟢", "Generative": "🟡", "Mature": "🔴"}
        c1.metric("Hasil Dominan", f"{badge_map.get(patch_majority,'')} {patch_majority}")
        c2.metric("Mature",      f"{patch_counts.get('Mature', 0)}/{total_patch} patch")
        c3.metric("Generative",  f"{patch_counts.get('Generative', 0)}/{total_patch} patch")
        c4.metric("Vegetative",  f"{patch_counts.get('Vegetative', 0)}/{total_patch} patch")

        # Konsistensi dengan LLM
        if patch_majority == kelas:
            st.success(f"✅ Patch majority **{patch_majority}** konsisten dengan hasil LLM **{kelas}**")
        else:
            st.warning(f"⚠️ Patch majority **{patch_majority}** berbeda dengan LLM **{kelas}** — area lahan mungkin tidak homogen")

        # Heatmap
        fig_patch = render_patch_heatmap(rgb, patch_results)
        st.pyplot(fig_patch)
        plt.close(fig_patch)

        # Tabel detail per patch
        with st.expander("📋 Detail Per Patch"):
            patch_table = []
            for r, row in enumerate(patch_results):
                for c, cell in enumerate(row):
                    patch_table.append({
                        "Patch": f"R{r+1}C{c+1}",
                        "Hasil": cell["label"],
                        "Posisi": f"({cell['x1']},{cell['y1']})→({cell['x2']},{cell['y2']})"
                    })
            st.dataframe(pd.DataFrame(patch_table), use_container_width=True)

    elif use_patch and clf is None:
        st.divider()
        st.warning("⚠️ Patch analysis membutuhkan model ML. Pastikan file `.joblib` ada di folder yang sama.")

    # ── Peta VI ───────────────────────────────────────────────────────────────
    if show_vi_maps:
        st.divider()
        st.subheader("🗺️ Peta Indeks Vegetasi (7 VI)")
        fig = render_vi_maps(rgb, vi_maps, mask)
        st.pyplot(fig)
        plt.close(fig)

    # ── Tabel Statistik ───────────────────────────────────────────────────────
    if show_vi_stats:
        st.divider()
        st.subheader("📊 Statistik Indeks Vegetasi")

        # Tampilkan green_fraction sebagai metrik utama kematangan
        gf_pct = green_frac * 100
        gf_label = "Vegetative (hijau dominan)" if gf_pct > 50 else \
                   "Generative (mulai menguning)" if gf_pct > 25 else \
                   "Mature (kuning/emas dominan)"
        st.metric("🌿 Green Fraction (proporsi piksel hijau)",
                  f"{gf_pct:.1f}%", delta=gf_label, delta_color="off")

        df_stats = pd.DataFrame(vi_stats).T.round(4)

        def color_text(val):
            """Teks putih di background gelap, hitam di background terang."""
            try:
                norm = (float(val) - df_stats.values.min()) / (df_stats.values.max() - df_stats.values.min() + 1e-9)
                return "color: white" if norm > 0.6 else "color: black"
            except Exception:
                return ""

        st.dataframe(
            df_stats.style
                .background_gradient(cmap="YlGn", axis=None, vmin=df_stats.values.min(), vmax=df_stats.values.max())
                .map(color_text),
            use_container_width=True,
        )

    # ── Simpan ke riwayat ─────────────────────────────────────────────────────
    st.session_state.history.append({
        "waktu":      datetime.now().strftime("%H:%M:%S"),
        "file":       uploaded.name,
        "kelas_llm":  kelas,
        "keyakinan":  keyakinan,
        "kelas_ml":   ml_pred if use_ml_compare else "-",
        "cocok":      (ml_pred == kelas) if (use_ml_compare and clf is not None) else "-",
        "latency_s":  round(latency, 2),
        "token":      usage.total_tokens,
        "alasan":     alasan,
    })

# ════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📋 Riwayat Prediksi — Sesi Ini")

    if not st.session_state.history:
        st.info("Belum ada prediksi. Upload gambar di tab Klasifikasi.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, "riwayat_prediksi.csv", "text/csv")
        with col_clr:
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state.history = []
                st.rerun()

        # Ringkasan statistik sesi
        st.divider()
        st.subheader("📈 Ringkasan Sesi")
        total = len(df_hist)
        dist  = df_hist["kelas_llm"].value_counts()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Prediksi", total)
        c2.metric("Mature (Siap Panen)", dist.get("Mature", 0))
        c3.metric("Generative",          dist.get("Generative", 0))
        c4.metric("Vegetative",          dist.get("Vegetative", 0))
