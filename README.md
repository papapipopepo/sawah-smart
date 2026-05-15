# SawahSmart — Sistem Monitoring Padi Cerdas

Sistem monitoring tanaman padi berbasis IoT dan AI menggunakan ESP32-CAM, Firebase, Google Cloud Run, dan Machine Learning untuk deteksi kesiapan panen.

## Struktur Proyek

```
sawahsmart/
├── firmware/       # ESP32-CAM (PlatformIO / Arduino)
├── backend/        # Cloud Run Flask API (GCS image upload)
├── web/            # Web dashboard PWA (Netlify)
├── ml/             # Machine learning (SVM, Streamlit)
└── docs/           # Diagram sistem (.drawio)
```

## Komponen Sistem

| Komponen | Teknologi | Fungsi |
|---|---|---|
| ESP32-CAM | Arduino / PlatformIO | Capture & upload foto ke Cloud Run |
| Cloud Run | Python Flask + GCS | Terima & simpan foto dari ESP32-CAM |
| Firebase RTDB | Firebase | Data sensor realtime (suhu, kelembapan, dll) |
| Web Dashboard | HTML/CSS/JS PWA | Monitoring & galeri foto |
| ML Model | SVM + ExGR | Klasifikasi kesiapan panen padi |

## Setup

### Firmware (ESP32-CAM)
1. Install [PlatformIO](https://platformio.org/)
2. Edit `firmware/src/main.cpp` — isi `WIFI_SSID`, `WIFI_PASSWORD`, `GCP_ENDPOINT`
3. `cd firmware && pio run --target upload`

### Backend (Cloud Run)
```bash
gcloud run deploy upload-padi \
  --source backend/ \
  --region asia-southeast2 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET_NAME=padi-images-thesis-496412
```

### Web Dashboard
Deploy folder `web/` ke [Netlify](https://netlify.com) (drag & drop).

### ML (Streamlit)
```bash
cd ml
pip install -r requirements.txt
streamlit run app.py
```
