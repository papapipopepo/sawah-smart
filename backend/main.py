"""
Cloud Run service: upload_padi + multi-model detect
- POST   /upload                  : Terima JPEG dari ESP32-CAM, simpan ke GCS
- POST   /predict                 : Deteksi + upload (model via header X-Model)
- POST   /predict-existing/<name> : Deteksi foto yang sudah ada di GCS (X-Model)
- GET    /models                  : Daftar model tersedia (CNN/OpenAI/Gemini)
- GET    /images                  : List gambar terbaru dari GCS
- GET    /image/<name>            : Serve gambar (GET) / hapus (DELETE)
- GET    /health                  : Health check

Alur deteksi (run_detection):
  1. Blur pre-filter Laplacian (opsional, BLUR_THRESHOLD; default 0 = mati)
  2. Klasifikasi via model terpilih:
     - LLM (GPT-4o/Gemini): 1 panggilan = gatekeeper (is_rice/is_clear) + fase
     - CNN Hybrid (lokal): gatekeeper LLM gratis dulu, lalu CNN klasifikasi fase
  3. ExG heatmap + green fraction selalu dihitung (numpy/matplotlib)

Env vars: GCS_BUCKET_NAME, MODEL_PATH, OPENAI_API_KEY (GPT), GEMINI_API_KEY (Gemini),
  DEFAULT_MODEL (default gemini-2.0-flash), GATEKEEPER_MODEL, BLUR_THRESHOLD,
  LANGFUSE_* (opsional, tracking)
"""

import os
import io
import base64
import datetime

import numpy as np
from flask import Flask, request, jsonify, Response
from google.cloud import storage

app = Flask(__name__)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "padi-images-thesis-496412")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/best_hybrid_cnn_ExG.keras")
CNN_IMG_SIZE = 224
CLASS_ORDER = ["Vegetative", "Generative", "Mature"]

# Validasi gambar: LLM gatekeeper (is_rice + is_clear). Laplacian = pre-filter opsional.
# Default 0 = Laplacian MATI (andalkan LLM untuk deteksi blur). Set >0 utk aktifkan.
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "0"))

# ============================================================
# Registry model — selector di web kirim model_id via header X-Model
# Provider LLM pakai endpoint OpenAI-compatible (1 code path).
# ============================================================
PROVIDERS = {
    "openai": {"base_url": None,
               "key_env": "OPENAI_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": "GEMINI_API_KEY"},
}

MODEL_REGISTRY = {
    # model_id (dipakai web/header)   provider    api_model               label tampil
    "cnn-hybrid":       {"provider": "local",  "model": None,                 "label": "CNN Hybrid (lokal)"},
    "gpt-4o":           {"provider": "openai", "model": "gpt-4o",             "label": "GPT-4o"},
    "gpt-4o-mini":      {"provider": "openai", "model": "gpt-4o-mini",        "label": "GPT-4o-mini"},
    "gemini-2.0-flash": {"provider": "gemini", "model": "gemini-2.0-flash",   "label": "Gemini 2.0 Flash (gratis)"},
    "gemini-2.5-flash": {"provider": "gemini", "model": "gemini-2.5-flash",   "label": "Gemini 2.5 Flash (gratis)"},
    "gemini-2.5-pro":   {"provider": "gemini", "model": "gemini-2.5-pro",     "label": "Gemini 2.5 Pro (gratis)"},
}

# Model default untuk gatekeeper saat klasifikasi pakai CNN (LLM gratis)
GATEKEEPER_MODEL = os.environ.get("GATEKEEPER_MODEL", "gemini-2.0-flash")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemini-2.0-flash")

storage_client = storage.Client()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Filename, X-Device-ID",
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
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        import tensorflow as tf
        print(f"[MODEL] Loading {MODEL_PATH}...")
        _MODEL = tf.keras.models.load_model(MODEL_PATH)
        print(f"[MODEL] Loaded. Inputs: {len(_MODEL.inputs)}")
    return _MODEL


# ============================================================
# VI / preprocessing helpers (port dari ml/app.py)
# ============================================================
def compute_ExG(rgb):
    r = rgb.astype(np.float32)
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (2 * G - R - B) / 255.0


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


def render_exg_heatmap_b64(exg_map, mask):
    """Render ExG heatmap pakai matplotlib 'Greens' cmap → PNG base64."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masked = exg_map.astype(float).copy()
    masked[~mask] = np.nan

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=90)
    im = ax.imshow(masked, cmap="Greens")
    ax.set_title("Indeks Vegetasi ExG", fontsize=11, fontweight="bold")
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


def predict_hybrid_cnn(rgb):
    """Inferensi Hybrid CNN+ExG. Return dict prediction lengkap."""
    import cv2
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # Branch 1: image (224x224x3)
    img = cv2.resize(rgb, (CNN_IMG_SIZE, CNN_IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_in = preprocess_input(img.astype(np.float32))
    img_in = np.expand_dims(img_in, axis=0)

    # Branch 2: VI features (1, 18) = 10 ExG stats + 8 GLCM
    exg_map = compute_ExG(rgb)
    mask = green_mask(rgb)
    stats = extract_vi_stats(exg_map.flatten())
    glcm_feats = extract_glcm_feats(rgb)
    vi_vec = np.array([list(stats.values()) + glcm_feats], dtype=np.float32)

    # Inference
    model = get_model()
    proba = model.predict([img_in, vi_vec], verbose=0)[0]
    pred_id = int(np.argmax(proba))
    label = CLASS_ORDER[pred_id]

    return {
        "label": label,
        "confidence": float(proba[pred_id]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_ORDER, proba.tolist())},
        "vi_name": "ExG",
        "vi_stats": stats,
        "green_fraction": float(mask.mean()),
        "exg_heatmap_b64": render_exg_heatmap_b64(exg_map, mask),
    }


# ============================================================
# Validasi gambar 3-lapis: blur → LLM gatekeeper
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


def blur_score(rgb):
    """Variance of Laplacian — ukuran ketajaman. Rendah = buram."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_mime(image_data):
    """Deteksi mime dari magic bytes (penting agar OpenAI vision tidak error)."""
    if image_data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if image_data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # fallback


def get_llm_client(model_id):
    """Return (client, api_model_name) untuk model_id, atau (None, None) kalau tak tersedia."""
    reg = MODEL_REGISTRY.get(model_id)
    if not reg or reg["provider"] == "local":
        return None, None
    prov = PROVIDERS[reg["provider"]]
    api_key = os.environ.get(prov["key_env"])
    if not api_key:
        return None, None
    from openai import OpenAI
    if prov["base_url"]:
        client = OpenAI(api_key=api_key, base_url=prov["base_url"])
    else:
        client = OpenAI(api_key=api_key)
    return client, reg["model"]


@lf_observe("llm-classify")
def llm_classify(image_data, model_id):
    """LLM: gatekeeper (is_rice, is_clear) + klasifikasi fase + probabilitas + alasan.

    Return {enabled, is_rice, is_clear, label, probabilities, reason, recommendation, tokens, model}.
    Kalau key tidak ada / error → enabled=False (fail-open).
    """
    result = {"enabled": False, "is_rice": True, "is_clear": True,
              "label": None, "probabilities": {}, "reason": "", "recommendation": "",
              "tokens": 0, "model": model_id}

    client, api_model = get_llm_client(model_id)
    if client is None:
        result["reason"] = f"model {model_id} tidak tersedia (API key belum di-set)"
        return result

    try:
        import json
        mime = detect_mime(image_data)
        b64 = base64.b64encode(image_data).decode("utf-8")

        system = (
            "Kamu ahli agronomi padi. Analisis gambar lalu kembalikan:\n"
            "1. is_rice: TANAMAN PADI (sawah/daun padi/malai/bulir)? false kalau "
            "orang/hewan/kendaraan/objek/tanaman lain/ruangan.\n"
            "2. is_clear: kualitas cukup jelas? false kalau buram/gelap berat.\n"
            "3. kelas: fase padi — 'Vegetative' (daun hijau, belum ada malai), "
            "'Generative' (malai muncul, bulir berkembang, mulai menguning), "
            "'Mature' (kuning keemasan, malai menunduk, SIAP PANEN). null kalau is_rice=false.\n"
            "4. probabilities: estimasi peluang tiap fase (jumlah ~1.0).\n"
            "5. alasan: penjelasan visual singkat. 6. rekomendasi: saran untuk petani.\n"
            "Balas HANYA JSON: "
            '{"is_rice":true,"is_clear":true,"kelas":"Vegetative|Generative|Mature|null",'
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
            max_tokens=400,
            temperature=0.1,
        )
        parsed = json.loads(resp.choices[0].message.content)
        result["enabled"] = True
        result["is_rice"] = bool(parsed.get("is_rice", True))
        result["is_clear"] = bool(parsed.get("is_clear", True))
        result["label"] = parsed.get("kelas") or None
        probs = parsed.get("probabilities", {}) or {}
        result["probabilities"] = {c: float(probs.get(c, 0) or 0) for c in CLASS_ORDER}
        result["reason"] = str(parsed.get("alasan", ""))
        result["recommendation"] = str(parsed.get("rekomendasi", ""))
        result["tokens"] = resp.usage.total_tokens if resp.usage else 0
        print(f"[LLM:{model_id}] mime={mime} is_rice={result['is_rice']} "
              f"is_clear={result['is_clear']} label={result['label']}")
    except Exception as e:
        result["reason"] = f"LLM error: {e}"
        print(f"[LLM:{model_id}] error: {e}")

    return result


def exg_visualization(rgb):
    """Hitung ExG: green_fraction, stats, heatmap base64 (dipakai CNN & LLM)."""
    exg_map = compute_ExG(rgb)
    mask = green_mask(rgb)
    return {
        "vi_name": "ExG",
        "vi_stats": extract_vi_stats(exg_map.flatten()),
        "green_fraction": float(mask.mean()),
        "exg_heatmap_b64": render_exg_heatmap_b64(exg_map, mask),
    }


def run_detection(rgb, image_data, model_id):
    """Pipeline: blur pre-filter → gatekeeper LLM → klasifikasi (CNN/LLM).

    Return dict: {status:'ok'|'rejected', reject_reason?, validation?, prediction?}
    """
    if model_id not in MODEL_REGISTRY:
        model_id = DEFAULT_MODEL
    info = {"model": model_id}

    # Lapis 1: blur Laplacian (opsional, default mati)
    bscore = blur_score(rgb)
    info["blur_score"] = round(bscore, 2)
    if BLUR_THRESHOLD > 0 and bscore < BLUR_THRESHOLD:
        return {"status": "rejected", "reject_reason": "blur", "validation": info}

    is_local = MODEL_REGISTRY[model_id]["provider"] == "local"

    if is_local:
        # CNN: gatekeeper pakai LLM gratis dulu, lalu CNN klasifikasi fase
        gate = llm_classify(image_data, GATEKEEPER_MODEL)
        info["llm_check"] = gate
        if gate.get("enabled"):
            if not gate.get("is_rice", True):
                return {"status": "rejected", "reject_reason": "not_rice", "validation": info}
            if not gate.get("is_clear", True):
                return {"status": "rejected", "reject_reason": "blur", "validation": info}
        pred = predict_hybrid_cnn(rgb)
        pred["engine"] = MODEL_REGISTRY[model_id]["label"]
        pred["validation"] = info
        return {"status": "ok", "prediction": pred}

    # LLM: satu panggilan gatekeeper + klasifikasi
    res = llm_classify(image_data, model_id)
    info["llm_check"] = {k: res[k] for k in ("enabled", "is_rice", "is_clear", "model", "tokens")}
    if not res.get("enabled"):
        return {"status": "error", "error": res.get("reason", "LLM tidak tersedia"),
                "validation": info}
    if not res.get("is_rice", True):
        info["llm_check"]["reason"] = res.get("reason", "")
        return {"status": "rejected", "reject_reason": "not_rice", "validation": info}
    if not res.get("is_clear", True):
        info["llm_check"]["reason"] = res.get("reason", "")
        return {"status": "rejected", "reject_reason": "blur", "validation": info}

    label = res.get("label") or "Unknown"
    probs = res.get("probabilities", {})
    conf = probs.get(label, max(probs.values()) if probs else 0.0)
    viz = exg_visualization(rgb)
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
    """Upload JPEG ke GCS + jalankan inferensi CNN + ExG heatmap."""
    if request.method == "OPTIONS":
        return cors({}, 204)

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
        result = run_detection(rgb, image_data, model_id)
        if result["status"] != "ok":
            result["filename"] = filename
            return cors(result, 200)

        # 2. Lolos → upload ke GCS
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_string(image_data, content_type="image/jpeg")
        blob.metadata = {"device_id": device_id}
        blob.patch()

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
    """Jalankan inferensi CNN pada foto yang sudah ada di GCS (tanpa re-upload)."""
    if request.method == "OPTIONS":
        return cors({}, 204)

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

        # Decode + deteksi (validasi + klasifikasi sesuai model terpilih)
        rgb = decode_jpeg_to_rgb(image_data)
        result = run_detection(rgb, image_data, model_id)
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


# ============================================================
# Capture-on-demand flag (in-memory, ephemeral)
# Web POST untuk request capture, ESP32 GET untuk consume (auto-clear)
# ============================================================
_capture_requested = False
_capture_request_ts = 0


@app.route("/capture-request", methods=["GET", "POST", "OPTIONS"])
def capture_request():
    global _capture_requested, _capture_request_ts

    if request.method == "OPTIONS":
        return cors({}, 204)

    if request.method == "POST":
        # Web request: set flag
        _capture_requested = True
        _capture_request_ts = int(datetime.datetime.utcnow().timestamp())
        print(f"[CAPTURE] Request received at ts={_capture_request_ts}")
        return cors({"status": "ok", "requested": True, "ts": _capture_request_ts}, 201)

    # GET (ESP32 polling): return flag dan auto-clear kalau true
    was_requested = _capture_requested
    if was_requested:
        _capture_requested = False
        print(f"[CAPTURE] ESP32 picked up request (was set at ts={_capture_request_ts})")
    return cors({"capture": was_requested, "ts": _capture_request_ts})


def available_models():
    """Daftar model yang bisa dipakai (key provider tersedia)."""
    out = []
    for mid, reg in MODEL_REGISTRY.items():
        if reg["provider"] == "local":
            ok = os.path.exists(MODEL_PATH)
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
        "model_loaded": _MODEL is not None,
        "capture_pending": _capture_requested,
        "blur_threshold": BLUR_THRESHOLD,
        "default_model": DEFAULT_MODEL,
        "gatekeeper_model": GATEKEEPER_MODEL,
        "models": available_models(),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
