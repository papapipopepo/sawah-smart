#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <time.h>
#include "esp_camera.h"
#include "config.h"

// ============================================================
// KONFIGURASI — ganti sesuai setup Anda (lihat config.h)
// ============================================================

// URL Cloud Run Anda setelah deploy
#define GCP_BASE_URL    "https://upload-padi-qg55cyk7ea-et.a.run.app"
#define GCP_UPLOAD      GCP_BASE_URL "/upload"
#define GCP_CAPTURE_REQ GCP_BASE_URL "/capture-request"

// Interval polling /capture-request (ms). 5000 = cek setiap 5 detik
#define POLL_INTERVAL_MS  5000UL

// ============================================================
// Pin ESP32-CAM AI-Thinker
// ============================================================
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22
#define FLASH_LED_PIN    4

// ============================================================
// Prototypes
// ============================================================
bool initCamera();
void connectWiFi();
bool captureAndUpload();
bool checkCaptureRequest();

unsigned long lastPollTime = 0;
int uploadCount = 0;
int failCount   = 0;
int pollCount   = 0;

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n========================================");
  Serial.println("  ESP32-CAM Padi Harvest Detection");
  Serial.println("  Board: AI-Thinker ESP32-S | Sensor: OV3660 3MP");
  Serial.println("  Resolusi: QXGA 2048x1536");
  Serial.println("========================================");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!initCamera()) {
    Serial.println("[ERROR] Kamera gagal init — restart dalam 5 detik");
    delay(5000);
    ESP.restart();
  }
  Serial.println("[OK] Kamera siap");

  connectWiFi();

  Serial.println("\n[READY] Sistem siap. Menunggu perintah capture dari dashboard web...");
  Serial.printf("[POLL] Polling %s setiap %lu detik\n", GCP_CAPTURE_REQ, POLL_INTERVAL_MS / 1000);
  lastPollTime = 0; // langsung poll di iterasi pertama
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WARN] WiFi terputus — reconnect...");
    connectWiFi();
  }

  if (millis() - lastPollTime >= POLL_INTERVAL_MS) {
    lastPollTime = millis();
    if (checkCaptureRequest()) {
      Serial.println("[CAPTURE] Permintaan capture diterima — mengambil foto...");
      captureAndUpload();
    }
  }
}

// ============================================================
bool initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0       = Y2_GPIO_NUM;
  cfg.pin_d1       = Y3_GPIO_NUM;
  cfg.pin_d2       = Y4_GPIO_NUM;
  cfg.pin_d3       = Y5_GPIO_NUM;
  cfg.pin_d4       = Y6_GPIO_NUM;
  cfg.pin_d5       = Y7_GPIO_NUM;
  cfg.pin_d6       = Y8_GPIO_NUM;
  cfg.pin_d7       = Y9_GPIO_NUM;
  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sscb_sda = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;

  // OV3660 3MP: QXGA (2048×1536) butuh PSRAM ~6MB
  // Fallback ke resolusi lebih rendah kalau PSRAM tidak ada/cukup
  if (psramFound()) {
    cfg.frame_size   = FRAMESIZE_QXGA;  // 2048×1536 (max OV3660)
    cfg.jpeg_quality = 10;              // 0-63, lower = better quality
    cfg.fb_count     = 1;               // 1 buffer untuk hemat PSRAM di QXGA
    Serial.println("[CAM] PSRAM ditemukan — pakai QXGA 2048x1536 (OV3660 3MP)");
  } else {
    // Tanpa PSRAM, QXGA tidak mungkin — fallback ke VGA
    cfg.frame_size   = FRAMESIZE_VGA;   // 640×480
    cfg.jpeg_quality = 15;
    cfg.fb_count     = 1;
    Serial.println("[WARN] Tanpa PSRAM — fallback ke VGA (resolusi rendah)");
  }

  if (esp_camera_init(&cfg) != ESP_OK) return false;

  // Buang beberapa frame agar sensor OV2640 stabil (exposure & AWB)
  for (int i = 0; i < 3; i++) {
    camera_fb_t *dummy = esp_camera_fb_get();
    if (dummy) esp_camera_fb_return(dummy);
    delay(100);
  }

  // Deteksi sensor type (OV2640 = 0x26, OV3660 = 0x3660, OV5640 = 0x5640)
  sensor_t *s = esp_camera_sensor_get();
  Serial.printf("[CAM] Sensor PID: 0x%04X", s->id.PID);
  switch (s->id.PID) {
    case OV2640_PID: Serial.println(" (OV2640 — 2MP)"); break;
    case OV3660_PID: Serial.println(" (OV3660 — 3MP) ✓"); break;
    case OV5640_PID: Serial.println(" (OV5640 — 5MP)"); break;
    default:         Serial.println(" (sensor tidak dikenali)"); break;
  }

  // Tuning sensor — OV3660 punya color & exposure handling lebih baik dari OV2640
  s->set_brightness(s, 0);    // -2 ke 2 (0 = netral, OV3660 cukup terang default)
  s->set_contrast(s, 0);      // -2 ke 2
  s->set_saturation(s, 0);    // -2 ke 2 (OV3660 saturasi natural cukup baik)
  s->set_whitebal(s, 1);      // enable white balance
  s->set_awb_gain(s, 1);      // enable AWB gain
  s->set_wb_mode(s, 0);       // 0 = auto white balance
  s->set_exposure_ctrl(s, 1); // enable AEC
  s->set_aec2(s, 1);          // DSP AEC (lebih akurat di OV3660)
  s->set_gain_ctrl(s, 1);     // enable AGC
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)2);  // GAIN 4x ceiling (anti noise outdoor)
  s->set_bpc(s, 1);           // bad pixel correction ON (perbaiki dead pixels)
  s->set_wpc(s, 1);           // white pixel correction ON
  s->set_raw_gma(s, 1);       // gamma correction
  s->set_lenc(s, 1);          // lens correction (kurangi vignetting)
  s->set_hmirror(s, 0);
  s->set_vflip(s, 0);
  s->set_dcw(s, 1);           // downsize EN
  s->set_colorbar(s, 0);

  return true;
}

// ============================================================
void connectWiFi() {
  Serial.printf("[WiFi] Menghubungkan ke: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] Terhubung — IP: %s\n", WiFi.localIP().toString().c_str());
    // Sync waktu via NTP (UTC+7 WIB)
    configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov");
    Serial.print("[NTP] Sinkronisasi waktu");
    struct tm t;
    for (int i = 0; i < 20 && !getLocalTime(&t); i++) {
      delay(500);
      Serial.print(".");
    }
    Serial.println(getLocalTime(&t) ? " OK" : " Gagal (pakai millis)");
  } else {
    Serial.println("\n[WiFi] Gagal terhubung — restart dalam 5 detik");
    delay(5000);
    ESP.restart();
  }
}

// ============================================================
bool captureAndUpload() {
  Serial.println("\n[CAM] Mengambil gambar...");

  // Nyalakan flash sesaat untuk pencahayaan
  digitalWrite(FLASH_LED_PIN, HIGH);
  delay(150);
  camera_fb_t *fb = esp_camera_fb_get();
  digitalWrite(FLASH_LED_PIN, LOW);

  if (!fb) {
    Serial.println("[ERROR] Gagal ambil frame");
    failCount++;
    return false;
  }
  Serial.printf("[CAM] Frame: %zu bytes (%dx%d)\n", fb->len, fb->width, fb->height);

  // Buat nama file dengan timestamp NTP (fallback ke millis jika NTP belum sync)
  char filename[48];
  struct tm t;
  if (getLocalTime(&t)) {
    char ts[20];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &t);
    snprintf(filename, sizeof(filename), "padi_%s_%s.jpg",
             WiFi.macAddress().c_str(), ts);
  } else {
    snprintf(filename, sizeof(filename), "padi_%s_%lu.jpg",
             WiFi.macAddress().c_str(), millis());
  }

  // Kirim ke Cloud Function via HTTPS
  WiFiClientSecure client;
  client.setInsecure(); // OK untuk development — ganti dengan CA cert untuk produksi

  HTTPClient http;
  if (!http.begin(client, GCP_UPLOAD)) {
    Serial.println("[HTTP] Gagal begin");
    esp_camera_fb_return(fb);
    failCount++;
    return false;
  }

  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Filename",   filename);
  http.addHeader("X-Device-ID",  WiFi.macAddress());
  http.setTimeout(30000); // 30 detik timeout (file QXGA ~400-800KB butuh lebih lama)

  Serial.printf("[HTTP] Mengirim ke GCP (%s)...\n", GCP_UPLOAD);
  int code = http.POST(fb->buf, fb->len);
  esp_camera_fb_return(fb); // Bebaskan memori frame sesegera mungkin

  bool success = false;
  if (code > 0) {
    Serial.printf("[HTTP] Response: %d\n", code);
    String body = http.getString();
    Serial.println("[HTTP] Body: " + body);
    success = (code == 200 || code == 201);
  } else {
    Serial.printf("[HTTP] Error: %s\n", http.errorToString(code).c_str());
  }

  http.end();

  if (success) {
    uploadCount++;
    Serial.printf("[OK] Upload berhasil #%d (file: %s)\n", uploadCount, filename);
  } else {
    failCount++;
    Serial.printf("[FAIL] Upload gagal (total gagal: %d)\n", failCount);
  }

  Serial.printf("[STAT] Berhasil: %d | Gagal: %d\n", uploadCount, failCount);
  return success;
}

// ============================================================
// Polling /capture-request: return true kalau ada permintaan capture
// ============================================================
bool checkCaptureRequest() {
  pollCount++;
  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;
  if (!http.begin(client, GCP_CAPTURE_REQ)) {
    return false;
  }
  http.setTimeout(5000);

  int code = http.GET();
  bool shouldCapture = false;

  if (code == 200) {
    String body = http.getString();
    // Parsing JSON sederhana: cari "capture":true
    // (Hindari ArduinoJson untuk hemat memory)
    shouldCapture = (body.indexOf("\"capture\":true") >= 0)
                 || (body.indexOf("\"capture\": true") >= 0);
    if (pollCount % 12 == 0) {  // log setiap 1 menit (12 poll x 5 detik)
      Serial.printf("[POLL] #%d — idle (waiting for trigger)\n", pollCount);
    }
  } else if (code > 0) {
    Serial.printf("[POLL] HTTP %d (poll #%d)\n", code, pollCount);
  } else {
    if (pollCount % 6 == 0) {
      Serial.printf("[POLL] Error %s (poll #%d)\n", http.errorToString(code).c_str(), pollCount);
    }
  }

  http.end();
  return shouldCapture;
}
