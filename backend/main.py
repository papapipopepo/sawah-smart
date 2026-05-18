"""
Cloud Run service: upload_padi + predict
- POST   /upload          : Terima JPEG dari ESP32-CAM, simpan ke GCS
- POST   /predict         : Upload JPEG + jalankan inferensi CNN Hybrid + ExG heatmap
- GET    /images          : List gambar terbaru dari GCS (JSON)
- GET    /image/<name>    : Serve gambar dari GCS (proxy)
- DELETE /image/<name>    : Hapus gambar dari GCS
- GET    /health          : Health check
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

    if not filename:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"padi_{device_id}_{ts}.jpg"

    try:
        # 1. Upload ke GCS
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"uploads/{filename}")
        blob.upload_from_string(image_data, content_type="image/jpeg")
        blob.metadata = {"device_id": device_id}
        blob.patch()

        gcs_path = f"gs://{GCS_BUCKET_NAME}/uploads/{filename}"

        # 2. Decode & inferensi
        rgb = decode_jpeg_to_rgb(image_data)
        prediction = predict_hybrid_cnn(rgb)

        print(f"[PREDICT] {filename}: {prediction['label']} "
              f"({prediction['confidence']*100:.1f}%) from {device_id}")

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
            if not blob.name.endswith(".jpg"):
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


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return cors({}, 204)
    model_loaded = _MODEL is not None
    return cors({
        "status": "ok",
        "bucket": GCS_BUCKET_NAME,
        "model_path": MODEL_PATH,
        "model_loaded": model_loaded,
        "capture_pending": _capture_requested,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
