"""
Cloud Run service: upload_padi + multi-model detect

Arsitektur 2-tier sesuai tesis SawahSmart:
  Tier 1 (gatekeeper) : VLM open-world reject (is_rice only)
  Tier 2 (classifier) : Hybrid EffNetB0+ExGR (default), SVM-RBF+ExGR, atau VLM zero-shot

Endpoints:
- POST   /upload                  : Terima JPEG dari ESP32-CAM, simpan ke GCS
- POST   /predict                 : Deteksi + upload (model via header X-Model)
- POST   /predict-existing/<name> : Deteksi foto yang sudah ada di GCS (X-Model)
- GET    /models                  : Daftar model tersedia (Hybrid/SVM/VLM)
- GET    /images                  : List gambar terbaru dari GCS
- GET    /image/<name>            : Serve gambar (GET) / hapus (DELETE)
- GET    /health                  : Health check

Alur deteksi (run_detection):
  1. Gatekeeper VLM: cek is_rice. Reject kalau bukan padi.
  2. Klasifikasi via model terpilih:
     - VLM (GPT-4o/4o-mini, Gemini 2.5 Flash/Flash-Lite): gatekeeper + fase 1 call
     - Hybrid EffNetB0+ExGR (lokal): VLM gatekeeper dulu, lalu Hybrid klasifikasi
     - SVM-RBF+ExGR (lokal): VLM gatekeeper dulu, lalu SVM klasifikasi
  3. ExGR heatmap + green fraction selalu dihitung (numpy/matplotlib)

Env vars: GCS_BUCKET_NAME, HYBRID_MODEL_PATH, SVM_MODEL_PATH, LABEL_ENCODER_PATH,
  OPENAI_API_KEY, GEMINI_API_KEY, DEFAULT_MODEL, GATEKEEPER_MODEL, LANGFUSE_*
"""

import os
import io
import base64
import datetime
import time

import numpy as np
from flask import Flask, request, jsonify, Response
from google.cloud import storage

app = Flask(__name__)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "padi-images-thesis-496412")
HYBRID_MODEL_PATH = os.environ.get("HYBRID_MODEL_PATH",
                                   "/app/models/best_dl_Hybrid_ExGR.keras")
SVM_MODEL_PATH = os.environ.get("SVM_MODEL_PATH",
                                "/app/models/best_3class_ExGR_SVM_RBF.joblib")
LABEL_ENCODER_PATH = os.environ.get("LABEL_ENCODER_PATH",
                                    "/app/models/label_encoder_3class.joblib")
CNN_IMG_SIZE = 224
CLASS_ORDER = ["Vegetative", "Generative", "Mature"]

# ============================================================
# Registry model — selector di web kirim model_id via header X-Model.
# Cohort sesuai tesis SawahSmart (Bab 4 snapshot 2026-06-21):
#   2 supervised lokal (Hybrid + SVM) + 4 VLM endpoint (Pro dihapus).
# Provider VLM pakai endpoint OpenAI-compatible (1 code path).
# ============================================================
PROVIDERS = {
    "openai": {"base_url": None,
               "key_env": "OPENAI_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY"},
}

MODEL_REGISTRY = {
    # model_id (web/header)            provider    api_model                  label tampil
    "hybrid-effnetb0-exgr":  {"provider": "local",  "model": None,                    "label": "Hybrid EffNetB0 + ExGR"},
    "svm-rbf-exgr":          {"provider": "local",  "model": None,                    "label": "ML SVM-RBF + ExGR"},
    "gemini-2.5-flash-lite": {"provider": "gemini", "model": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash-Lite"},
    "gemini-2.5-flash":      {"provider": "gemini", "model": "gemini-2.5-flash",      "label": "Gemini 2.5 Flash"},
    "gpt-4o":                {"provider": "openai", "model": "gpt-4o",                "label": "GPT-4o"},
    "gpt-4o-mini":           {"provider": "openai", "model": "gpt-4o-mini",           "label": "GPT-4o-mini"},
}

# Path lokal per model_id (untuk available_models check & loading).
LOCAL_MODEL_PATH = {
    "hybrid-effnetb0-exgr": HYBRID_MODEL_PATH,
    "svm-rbf-exgr":         SVM_MODEL_PATH,
}

GATEKEEPER_MODEL = os.environ.get("GATEKEEPER_MODEL", "gemini-2.5-flash-lite")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "hybrid-effnetb0-exgr")

storage_client = storage.Client()


# ============================================================
# Latency instrumentation — print prefix [LATENCY] biar mudah di-grep dari
# Cloud Run log (gcloud logging read). Stage = nama checkpoint, t0 = mulai
# (untuk hitung elapsed). Return waktu sekarang (epoch ms) supaya bisa dipakai
# sebagai t0 untuk stage berikutnya.
# ============================================================
def log_latency(stage, t0=None, **extra):
    now_ms = int(time.time() * 1000)
    parts = [f"stage={stage}", f"t_ms={now_ms}"]
    if t0 is not None:
        parts.append(f"elapsed_ms={now_ms - t0}")
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    print("[LATENCY] " + " ".join(parts), flush=True)
    return now_ms

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Filename, X-Device-ID, X-Model",
}


def cors(data, status=200, mimetype=None):
    if isinstance(data, dict):
        r = jsonify(data)
    else:
        r = Response(data, mimetype=mimetype or "application/octet-stream")
    for k, v in CORS_HEADERS.items():
        r.headers[k] = v
    r.status_code = status
    return r


# ============================================================
# Model loading (lazy, sekali di proses)
# ============================================================
_HYBRID_MODEL = None
_SVM_MODEL = None
_LABEL_ENCODER = None


def get_hybrid_model():
    global _HYBRID_MODEL
    if _HYBRID_MODEL is None:
        import tensorflow as tf
        print(f"[MODEL] Loading Hybrid {HYBRID_MODEL_PATH}...")
        _HYBRID_MODEL = tf.keras.models.load_model(HYBRID_MODEL_PATH)
        print(f"[MODEL] Hybrid loaded. Inputs: {len(_HYBRID_MODEL.inputs)}")
    return _HYBRID_MODEL


def get_svm_model():
    global _SVM_MODEL, _LABEL_ENCODER
    if _SVM_MODEL is None:
        import joblib
        print(f"[MODEL] Loading SVM {SVM_MODEL_PATH}...")
        _SVM_MODEL = joblib.load(SVM_MODEL_PATH)
        _LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
        print(f"[MODEL] SVM + label encoder loaded.")
    return _SVM_MODEL, _LABEL_ENCODER


# ============================================================
# VI / preprocessing helpers (port dari ml/eval_bench.py).
# ExGR = (2G - R - B) - (1.4R - G), sesuai feature engineering thesis.
# ============================================================
def compute_ExGR(rgb):
    r = rgb.astype(np.float32) / 255.0
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (2 * G - R - B) - (1.4 * R - G)


def green_mask(rgb, threshold=5, min_green=40):
    R = rgb[:, :, 0].astype(np.int32)
    G = rgb[:, :, 1].astype(np.int32)
    B = rgb[:, :, 2].astype(np.int32)
    return (G > R + threshold) & (G > B + threshold) & (G > min_green)


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


def extract_glcm_feats(rgb):
    import cv2
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = graycoprops(glcm, prop).flatten()
        feats += [float(np.mean(vals)), float(np.std(vals))]
    return feats


def render_exgr_heatmap_b64(vi_map, mask):
    """Render ExGR heatmap pakai matplotlib 'Greens' cmap → PNG base64."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked = vi_map.astype(float).copy()
    masked[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=90)
    im = ax.imshow(masked, cmap="Greens")
    ax.set_title("Indeks Vegetasi ExGR", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def decode_jpeg_to_rgb(image_data):
    """Decode JPEG bytes → numpy RGB array (H, W, 3) uint8."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    return np.array(img)


def _extract_exgr_features(rgb):
    """Hitung VI 18-dim (10 ExGR stats + 8 GLCM) untuk Hybrid & SVM."""
    vi_map = compute_ExGR(rgb)
    mask = green_mask(rgb)
    stats = extract_vi_stats(vi_map.flatten())
    glcm_feats = extract_glcm_feats(rgb)
    vi_vec = np.array([list(stats.values()) + glcm_feats], dtype=np.float32)
    return vi_vec, vi_map, mask, stats


def predict_hybrid_cnn(rgb):
    """Inferensi Hybrid EffNetB0 + ExGR (18 VI features). Return dict prediction lengkap."""
    import cv2
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # Branch 1: image (224x224x3)
    img = cv2.resize(rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_in = preprocess_input(img.astype(np.float32))
    img_in = np.expand_dims(img_in, axis=0)

    # Branch 2: VI features (1, 18) = 10 ExGR stats + 8 GLCM
    vi_vec, vi_map, mask, stats = _extract_exgr_features(rgb)

    # Inference
    model = get_hybrid_model()
    proba = model.predict([img_in, vi_vec], verbose=0)[0]
    pred_id = int(np.argmax(proba))
    label = CLASS_ORDER[pred_id]

    return {
        "label": label,
        "confidence": float(proba[pred_id]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_ORDER, proba.tolist())},
        "vi_name": "ExGR",
        "vi_stats": stats,
        "green_fraction": float(mask.mean()),
        "exg_heatmap_b64": render_exgr_heatmap_b64(vi_map, mask),
    }


def predict_svm_rbf(rgb):
    """Inferensi ML SVM-RBF + ExGR (18 VI features). Return dict prediction lengkap."""
    vi_vec, vi_map, mask, stats = _extract_exgr_features(rgb)

    clf, label_encoder = get_svm_model()
    y_pred_idx = clf.predict(vi_vec)[0]
    # Decode via label encoder (joblib disimpan dengan urutan training).
    if hasattr(label_encoder, "inverse_transform"):
        label = str(label_encoder.inverse_transform([y_pred_idx])[0])
    else:
        label = CLASS_ORDER[int(y_pred_idx)]

    # SVM bisa expose probabilities kalau dilatih dengan probability=True.
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(vi_vec)[0]
        # Map index → class name via label encoder kalau ada
        if hasattr(label_encoder, "inverse_transform"):
            cls_names = [str(label_encoder.inverse_transform([i])[0])
                         for i in range(len(proba))]
        else:
            cls_names = CLASS_ORDER[:len(proba)]
        prob_dict = {c: float(p) for c, p in zip(cls_names, proba.tolist())}
        confidence = float(proba.max())
    else:
        prob_dict = {label: 1.0}
        confidence = 1.0

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "vi_name": "ExGR",
        "vi_stats": stats,
        "green_fraction": float(mask.mean()),
        "exg_heatmap_b64": render_exgr_heatmap_b64(vi_map, mask),
    }


# ============================================================
# Gatekeeper VLM: open-world reject (is_rice only). Tier 1 dari arsitektur 2-tier.
# ============================================================

# Langfuse opsional (graceful) — dukung v3 (from langfuse) & v2 (langfuse.decorators)
try:
    from langfuse import observe as _observe          # langfuse v3+
except Exception:
    try:
        from langfuse.decorators import observe as _observe  # langfuse v2
    except Exception:
        _observe = None


def lf_observe(name):
    """Decorator: pakai langfuse @observe kalau tersedia, else no-op."""
    def deco(fn):
        if _observe is None:
            return fn
        try:
            return _observe(name=name)(fn)
        except Exception:
            return fn
    return deco


def detect_mime(image_data):
    """Deteksi mime dari magic bytes (penting agar OpenAI vision tidak error)."""
    if image_data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if image_data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # fallback


def downscale_for_llm(image_data, max_dim=1024, quality=85):
    """Perkecil gambar sebelum kirim ke LLM — hemat token & lebih cepat.
    Selalu re-encode JPEG (mime jadi pasti image/jpeg)."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _openai_class():
    """Pakai langfuse.openai (auto-track model/token/cost) kalau ada, else openai biasa."""
    try:
        from langfuse.openai import OpenAI as _LFOpenAI  # auto-instrumented
        return _LFOpenAI
    except Exception:
        from openai import OpenAI as _OpenAI
        return _OpenAI


def get_llm_client(model_id):
    """Return (client, api_model_name) untuk model_id, atau (None, None) kalau tak tersedia."""
    reg = MODEL_REGISTRY.get(model_id)
    if not reg or reg["provider"] == "local":
        return None, None
    prov = PROVIDERS[reg["provider"]]
    api_key = os.environ.get(prov["key_env"])
    if not api_key:
        return None, None
    OpenAI = _openai_class()
    if prov["base_url"]:
        client = OpenAI(api_key=api_key, base_url=prov["base_url"])
    else:
        client = OpenAI(api_key=api_key)
    return client, reg["model"]


@lf_observe("llm-classify")
def llm_classify(image_data, model_id):
    """VLM call: gatekeeper (is_rice) + klasifikasi fase + probabilitas + alasan.

    Return {enabled, is_rice, label, probabilities, reason, recommendation, tokens, model}.
    Kalau key tidak ada / error → enabled=False (fail-open).
    """
    result = {"enabled": False, "is_rice": True,
              "label": None, "probabilities": {}, "reason": "", "recommendation": "",
              "tokens": 0, "model": model_id}

    client, api_model = get_llm_client(model_id)
    if client is None:
        result["reason"] = f"model {model_id} tidak tersedia (API key belum di-set)"
        return result

    try:
        import json
        # Perkecil gambar → hemat token (hindari limit) + lebih cepat
        small = downscale_for_llm(image_data)
        mime = "image/jpeg"
        b64 = base64.b64encode(small).decode("utf-8")

        system = (
            "Kamu ahli agronomi padi. Analisis gambar lalu kembalikan:\n"
            "1. is_rice: TANAMAN PADI (sawah/daun padi/malai/bulir)? false kalau "
            "orang/hewan/kendaraan/objek/tanaman lain/ruangan.\n"
            "2. kelas: fase padi — 'Vegetative' (daun hijau, belum ada malai), "
            "'Generative' (malai muncul, bulir berkembang, mulai menguning), "
            "'Mature' (kuning keemasan, malai menunduk, SIAP PANEN). null kalau is_rice=false.\n"
            "3. probabilities: estimasi peluang tiap fase (jumlah ~1.0).\n"
            "4. alasan: SANGAT SINGKAT maks 1 kalimat. 5. rekomendasi: SANGAT SINGKAT maks 1 kalimat.\n"
            "Balas HANYA JSON valid & ringkas: "
            '{"is_rice":true,"kelas":"Vegetative|Generative|Mature|null",'
            '"probabilities":{"Vegetative":0.0,"Generative":0.0,"Mature":0.0},'
            '"alasan":"...","rekomendasi":"..."}'
        )
        resp = client.chat.completions.create(
            model=api_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": "Klasifikasikan fase padi pada gambar ini."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]},
            ],
            response_format={"type": "json_object"},
            max_tokens=800,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Salvage JSON yang terpotong: ambil sampai '}' terakhir
            cut = raw.rfind("}")
            parsed = json.loads(raw[:cut + 1]) if cut > 0 else {}
        result["enabled"] = True
        result["is_rice"] = bool(parsed.get("is_rice", True))
        result["label"] = parsed.get("kelas") or None
        probs = parsed.get("probabilities", {}) or {}
        result["probabilities"] = {c: float(probs.get(c, 0) or 0) for c in CLASS_ORDER}
        result["reason"] = str(parsed.get("alasan", ""))
        result["recommendation"] = str(parsed.get("rekomendasi", ""))
        result["tokens"] = resp.usage.total_tokens if resp.usage else 0
        print(f"[LLM:{model_id}] mime={mime} is_rice={result['is_rice']} "
              f"label={result['label']}")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            result["reason"] = (f"Kuota model {model_id} habis (rate limit). "
                                "Coba model lain (GPT-4o / Gemini lain) atau tunggu sebentar.")
        else:
            result["reason"] = f"LLM error: {msg[:200]}"
        print(f"[LLM:{model_id}] error: {e}")

    return result


def exgr_visualization(rgb):
    """Hitung ExGR: green_fraction, stats, heatmap base64 (dipakai VLM classifier track)."""
    vi_map = compute_ExGR(rgb)
    mask = green_mask(rgb)
    return {
        "vi_name": "ExGR",
        "vi_stats": extract_vi_stats(vi_map.flatten()),
        "green_fraction": float(mask.mean()),
        "exg_heatmap_b64": render_exgr_heatmap_b64(vi_map, mask),
    }


# Local supervised dispatcher (model_id → predict fn).
LOCAL_PREDICTORS = {
    "hybrid-effnetb0-exgr": predict_hybrid_cnn,
    "svm-rbf-exgr":         predict_svm_rbf,
}


def run_detection(rgb, image_data, model_id):
    """Pipeline 2-tier: VLM gatekeeper (is_rice) → klasifikasi (Hybrid/SVM/VLM).

    Return dict: {status:'ok'|'rejected', reject_reason?, validation?, prediction?}
    """
    if model_id not in MODEL_REGISTRY:
        model_id = DEFAULT_MODEL
    info = {"model": model_id}

    is_local = MODEL_REGISTRY[model_id]["provider"] == "local"

    if is_local:
        # Tier 1: VLM gatekeeper (is_rice only). Tier 2: supervised lokal (Hybrid/SVM).
        gate = llm_classify(image_data, GATEKEEPER_MODEL)
        info["llm_check"] = gate
        if gate.get("enabled") and not gate.get("is_rice", True):
            return {"status": "rejected", "reject_reason": "not_rice", "validation": info}
        predictor = LOCAL_PREDICTORS[model_id]
        pred = predictor(rgb)
        pred["engine"] = MODEL_REGISTRY[model_id]["label"]
        pred["validation"] = info
        return {"status": "ok", "prediction": pred}

    # VLM track: satu panggilan gatekeeper + klasifikasi (zero-shot).
    res = llm_classify(image_data, model_id)
    info["llm_check"] = {k: res[k] for k in ("enabled", "is_rice", "model", "tokens")}
    if not res.get("enabled"):
        return {"status": "error", "error": res.get("reason", "LLM tidak tersedia"),
                "validation": info}
    if not res.get("is_rice", True):
        info["llm_check"]["reason"] = res.get("reason", "")
        return {"status": "rejected", "reject_reason": "not_rice", "validation": info}

    probs = res.get("probabilities", {}) or {}
    label = res.get("label")
    # Fallback: kalau label kosong tapi ada probabilitas, ambil yang tertinggi
    if (not label or label == "Unknown") and probs:
        label = max(probs, key=probs.get)
    label = label or "Unknown"
    conf = probs.get(label, 0.0)
    viz = exgr_visualization(rgb)
    pred = {
        "label": label,
        "confidence": float(conf),
        "probabilities": probs,
        "engine": MODEL_REGISTRY[model_id]["label"],
        "notes": res.get("reason", ""),
        "recommendation": res.get("recommendation", ""),
        "validation": info,
        **viz,
    }
    return {"status": "ok", "prediction": pred}


# ============================================================
# Endpoints
# ============================================================
@app.route("/upload", methods=["POST", "OPTIONS"])
def upload_padi():
    if request.method == "OPTIONS":
        return cors({}, 204)

    t_receive = log_latency("upload_receive", endpoint="/upload")

    image_data = request.get_data()
    if not image_data:
        return cors({"error": "No image data received"}, 400)

    filename = request.headers.get("X-Filename")
    device_id = request.headers.get("X-Device-ID", "unknown")

    if not filename:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"padi_{device_id}_{ts}.jpg"

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_string(image_data, content_type="image/jpeg")
        blob.metadata = {"device_id": device_id}
        blob.patch()
        log_latency("upload_gcs_done", t0=t_receive,
                    file=filename, bytes=len(image_data))

        gcs_path = f"gs://{GCS_BUCKET_NAME}/uploads/{filename}"
        print(f"[OK] Saved: {gcs_path} ({len(image_data)} bytes) from {device_id}")

        return cors({
            "status": "ok",
            "filename": filename,
            "gcs_path": gcs_path,
            "size": len(image_data)
        }, 201)

    except Exception as e:
        print(f"[ERROR] GCS upload failed: {e}")
        return cors({"error": str(e)}, 500)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Upload JPEG ke GCS + jalankan inferensi (Hybrid/SVM/VLM) + ExGR heatmap."""
    if request.method == "OPTIONS":
        return cors({}, 204)

    t_receive = log_latency("predict_receive", endpoint="/predict")

    image_data = request.get_data()
    if not image_data:
        return cors({"error": "No image data received"}, 400)

    filename = request.headers.get("X-Filename")
    device_id = request.headers.get("X-Device-ID", "manual")
    model_id = request.headers.get("X-Model", DEFAULT_MODEL)

    if not filename:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"padi_{device_id}_{ts}.jpg"

    try:
        # 1. Decode + deteksi (validasi + klasifikasi) SEBELUM upload
        rgb = decode_jpeg_to_rgb(image_data)
        log_latency("predict_decode_done", t0=t_receive,
                    file=filename, model=model_id)
        result = run_detection(rgb, image_data, model_id)
        log_latency("predict_infer_done", t0=t_receive,
                    file=filename, model=model_id, status=result["status"])
        if result["status"] != "ok":
            result["filename"] = filename
            return cors(result, 200)

        # 2. Lolos → upload ke GCS
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_string(image_data, content_type="image/jpeg")
        blob.metadata = {"device_id": device_id}
        blob.patch()
        log_latency("predict_resp_ready", t0=t_receive,
                    file=filename, model=model_id)

        gcs_path = f"gs://{GCS_BUCKET_NAME}/uploads/{filename}"
        prediction = result["prediction"]
        print(f"[PREDICT] {filename}: {prediction['label']} "
              f"({prediction['confidence']*100:.1f}%) [{model_id}] from {device_id}")

        return cors({
            "status": "ok",
            "filename": filename,
            "gcs_path": gcs_path,
            "size": len(image_data),
            "prediction": prediction,
        }, 201)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] predict failed: {e}")
        return cors({"error": str(e)}, 500)


@app.route("/images", methods=["GET", "OPTIONS"])
def list_images():
    if request.method == "OPTIONS":
        return cors({}, 204)

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blobs = list(bucket.list_blobs(prefix="uploads/"))

        images = []
        for blob in blobs:
            if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            meta = blob.metadata or {}
            images.append({
                "filename": blob.name.split("/")[-1],
                "size": blob.size,
                "created": blob.time_created.isoformat() if blob.time_created else "",
                "device_id": meta.get("device_id", "unknown"),
            })

        images.sort(key=lambda x: x["created"], reverse=True)
        images = images[:50]

        return cors({"images": images, "count": len(images)})

    except Exception as e:
        print(f"[ERROR] list_images: {e}")
        return cors({"error": str(e)}, 500)


@app.route("/image/<filename>", methods=["GET", "DELETE", "OPTIONS"])
def image_resource(filename):
    """GET = serve image, DELETE = hapus dari GCS."""
    if request.method == "OPTIONS":
        return cors({}, 204)

    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(f"uploads/{filename}")

    if request.method == "DELETE":
        try:
            if not blob.exists():
                return cors({"error": "Image not found"}, 404)
            blob.delete()
            print(f"[DEL] Deleted: uploads/{filename}")
            return cors({"status": "ok", "deleted": filename})
        except Exception as e:
            print(f"[ERROR] delete_image {filename}: {e}")
            return cors({"error": str(e)}, 500)

    # GET
    try:
        image_data = blob.download_as_bytes()
        r = Response(image_data, mimetype="image/jpeg")
        for k, v in CORS_HEADERS.items():
            r.headers[k] = v
        r.headers["Cache-Control"] = "public, max-age=3600"
        return r
    except Exception as e:
        print(f"[ERROR] serve_image {filename}: {e}")
        return cors({"error": "Image not found"}, 404)


@app.route("/predict-existing/<filename>", methods=["POST", "OPTIONS"])
def predict_existing(filename):
    """Jalankan inferensi (Hybrid/SVM/VLM) pada foto yang sudah ada di GCS (tanpa re-upload)."""
    if request.method == "OPTIONS":
        return cors({}, 204)

    t_receive = log_latency("predict_existing_receive",
                            endpoint="/predict-existing", file=filename)

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"uploads/{filename}")

        if not blob.exists():
            return cors({"error": "File not found in GCS"}, 404)

        # Download photo dari GCS
        image_data = blob.download_as_bytes()
        blob.reload()
        device_id = (blob.metadata or {}).get("device_id", "unknown")
        model_id = request.headers.get("X-Model", DEFAULT_MODEL)
        log_latency("predict_existing_gcs_fetch", t0=t_receive,
                    file=filename, model=model_id, bytes=len(image_data))

        # Decode + deteksi (validasi + klasifikasi sesuai model terpilih)
        rgb = decode_jpeg_to_rgb(image_data)
        log_latency("predict_existing_decode_done", t0=t_receive,
                    file=filename, model=model_id)
        result = run_detection(rgb, image_data, model_id)
        log_latency("predict_existing_infer_done", t0=t_receive,
                    file=filename, model=model_id, status=result["status"])
        result["filename"] = filename
        result["device_id"] = device_id

        if result["status"] == "ok":
            print(f"[PREDICT-EXISTING] {filename}: {result['prediction']['label']} "
                  f"[{model_id}] device={device_id}")
        return cors(result, 200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] predict_existing failed: {e}")
        return cors({"error": str(e)}, 500)


def available_models():
    """Daftar model yang bisa dipakai (file lokal ada atau API key set)."""
    out = []
    for mid, reg in MODEL_REGISTRY.items():
        if reg["provider"] == "local":
            ok = os.path.exists(LOCAL_MODEL_PATH.get(mid, ""))
            if mid == "svm-rbf-exgr":
                ok = ok and os.path.exists(LABEL_ENCODER_PATH)
        else:
            ok = bool(os.environ.get(PROVIDERS[reg["provider"]]["key_env"]))
        out.append({"id": mid, "label": reg["label"],
                    "provider": reg["provider"], "available": ok})
    return out


@app.route("/models", methods=["GET", "OPTIONS"])
def models():
    if request.method == "OPTIONS":
        return cors({}, 204)
    return cors({"models": available_models(), "default": DEFAULT_MODEL})


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return cors({}, 204)
    return cors({
        "status": "ok",
        "bucket": GCS_BUCKET_NAME,
        "hybrid_loaded": _HYBRID_MODEL is not None,
        "svm_loaded": _SVM_MODEL is not None,
        "default_model": DEFAULT_MODEL,
        "gatekeeper_model": GATEKEEPER_MODEL,
        "models": available_models(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
