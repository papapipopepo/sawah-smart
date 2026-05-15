"""
Cloud Run service: upload_padi
- POST /upload        : Terima JPEG dari ESP32-CAM, simpan ke GCS
- GET  /images        : List gambar terbaru dari GCS (JSON)
- GET  /image/<name>  : Serve gambar dari GCS (proxy)
- GET  /health        : Health check
"""

import os
import datetime
from flask import Flask, request, jsonify, Response
from google.cloud import storage

app = Flask(__name__)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "padi-images-thesis-496412")
storage_client = storage.Client()

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Filename, X-Device-ID",
}


def cors(data, status=200):
    if isinstance(data, dict):
        r = jsonify(data)
    else:
        r = Response(data, mimetype="image/jpeg")
    for k, v in CORS_HEADERS.items():
        r.headers[k] = v
    r.status_code = status
    return r


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


@app.route("/images", methods=["GET", "OPTIONS"])
def list_images():
    """Kembalikan daftar 50 gambar terbaru sebagai JSON."""
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

        # Urutkan terbaru di atas, ambil 50 terakhir
        images.sort(key=lambda x: x["created"], reverse=True)
        images = images[:50]

        return cors({"images": images, "count": len(images)})

    except Exception as e:
        print(f"[ERROR] list_images: {e}")
        return cors({"error": str(e)}, 500)


@app.route("/image/<filename>", methods=["GET", "OPTIONS"])
def serve_image(filename):
    """Proxy: serve gambar dari GCS langsung ke browser."""
    if request.method == "OPTIONS":
        return cors({}, 204)

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"uploads/{filename}")
        image_data = blob.download_as_bytes()
        r = Response(image_data, mimetype="image/jpeg")
        for k, v in CORS_HEADERS.items():
            r.headers[k] = v
        r.headers["Cache-Control"] = "public, max-age=3600"
        return r

    except Exception as e:
        print(f"[ERROR] serve_image {filename}: {e}")
        return cors({"error": "Image not found"}, 404)


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return cors({}, 204)
    return cors({"status": "ok", "bucket": GCS_BUCKET_NAME})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
