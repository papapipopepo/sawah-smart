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

    model_files = [f for f in os.listdir(".") if f.startswith("best_3class_") and f.endswith(".joblib")]
    enc_path = "label_encoder_3class.joblib"
    if model_files and os.path.exists(enc_path):
        clf = joblib.load(model_files[0])
        le  = joblib.load(enc_path)
        # Cari VI name dengan mencocokkan ke VI_REGISTRY (robust terhadap model name dengan underscore)
        vi_name = next(
            (vi for vi in VI_REGISTRY if f"best_3class_{vi}_" in model_files[0]),
            None
        )
        return clf, le, vi_name
    return None, None, None

def predict_ml(rgb, clf, le, vi_name):
    """Ekstraksi fitur harus identik dengan training notebook.
    Model lama (18 fitur): VI stats dari green-masked pixels + 8 GLCM.
    Model baru (19 fitur): green_fraction + VI stats full-image + 8 GLCM.
    n_features_in_ pada pipeline dideteksi otomatis untuk memilih mode.
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

    # Deteksi jumlah fitur dari model yang disimpan
    n_feats = clf[0].n_features_in_ if hasattr(clf[0], "n_features_in_") else 18

    if n_feats >= 19:
        # Model baru: green_fraction + VI full-image + GLCM
        stats     = extract_vi_stats(vi_map.flatten())
        stat_names = list(stats.keys())
        feats = [float(mask.mean())] + [stats[s] for s in stat_names] + [glcm_feats[g] for g in glcm_names]
    else:
        # Model lama: VI green-masked + GLCM (tanpa green_fraction)
        vi_use    = vi_map[mask] if mask.mean() > 0.01 else vi_map.flatten()
        stats     = extract_vi_stats(vi_use)
        stat_names = list(stats.keys())
        feats = [stats[s] for s in stat_names] + [glcm_feats[g] for g in glcm_names]

    pred_id = clf.predict(np.array([feats], dtype=np.float32))[0]
    return le.inverse_transform([pred_id])[0]

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
        "- Mature: fase matang, bulir kuning keemasan penuh, malai menunduk, SIAP PANEN\n\n"
        "Panduan green_fraction: >50% = Vegetative, 25-50% = Generative, <25% = Mature\n\n"
        "Balas HANYA dalam format JSON:\n"
        '{"kelas":"Vegetative|Generative|Mature","keyakinan":"Tinggi|Sedang|Rendah",'
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
    st.divider()
    st.markdown("**Keterangan Kelas**")
    st.markdown("🟢 **Vegetative** — masih tumbuh")
    st.markdown("🟡 **Generative** — bulir berkembang")
    st.markdown("🔴 **Mature** — SIAP PANEN")

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

    clf, le, vi_name_ml = load_ml_model()

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
        badge = {"Vegetative": "🟢", "Generative": "🟡", "Mature": "🔴"}.get(kelas, "⚪")

        if kelas == "Mature":
            st.success(f"## {badge} {kelas.upper()} — SIAP PANEN!")
            st.balloons()
        elif kelas == "Generative":
            st.warning(f"## {badge} {kelas} — Belum siap panen")
        else:
            st.info(f"## {badge} {kelas} — Masih tumbuh")

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

        if clf is not None:
            with st.spinner("Prediksi dengan model ML..."):
                ml_pred = predict_ml(rgb, clf, le, vi_name_ml)

            match   = ml_pred == kelas
            c_llm, c_ml, c_match = st.columns(3)
            c_llm.metric("LLM Vision",  kelas,   delta=f"Keyakinan: {keyakinan}")
            c_ml.metric("Model ML",     ml_pred, delta=f"VI: {vi_name_ml}")
            c_match.metric("Hasil",     "✅ Sama" if match else "⚠️ Berbeda",
                           delta="Konsisten" if match else "Perlu dicermati")
        else:
            st.warning(
                "Model ML tidak ditemukan. "
                "Copy file `best_3class_*.joblib` dan `label_encoder_3class.joblib` "
                "ke folder yang sama dengan `app.py`, lalu refresh."
            )
            ml_pred = "-"

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
