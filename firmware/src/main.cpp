#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include "esp_camera.h"

// ============================================================
// KONFIGURASI — ganti sesuai setup Anda
// ============================================================
#define WIFI_SSID       "ALZRA"
#define WIFI_PASSWORD   "Alzra1111"

// URL Cloud Function GCP Anda setelah deploy
// Contoh: https://asia-southeast2-project-id.cloudfunctions.net/upload_padi
#define GCP_ENDPOINT    "https://upload-padi-qg55cyk7ea-et.a.run.app/upload"

// Interval ambil gambar (ms). 30000 = 30 detik
#define CAPTURE_INTERVAL_MS  30000UL

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

unsigned long lastCaptureTime = 0;
int uploadCount = 0;
int failCount   = 0;

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n========================================");
  Serial.println("  ESP32-CAM Padi Harvest Detection");
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

  // Langsung ambil gambar pertama saat boot
  captureAndUpload();
  lastCaptureTime = millis();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WARN] WiFi terputus — reconnect...");
    connectWiFi();
  }

  if (millis() - lastCaptureTime >= CAPTURE_INTERVAL_MS) {
    lastCaptureTime = millis();
    captureAndUpload();
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

  // SVGA butuh PSRAM, VGA fallback jika tidak ada
  if (psramFound()) {
    cfg.frame_size   = FRAMESIZE_SVGA;  // 800×600
    cfg.jpeg_quality = 12;
    cfg.fb_count     = 2;
    Serial.println("[CAM] PSRAM ditemukan — pakai SVGA");
  } else {
    cfg.frame_size   = FRAMESIZE_VGA;   // 640×480
    cfg.jpeg_quality = 15;
    cfg.fb_count     = 1;
    Serial.println("[CAM] Tanpa PSRAM — pakai VGA");
  }

  if (esp_camera_init(&cfg) != ESP_OK) return false;

  // Kalibrasi sensor OV2640
  sensor_t *s = esp_camera_sensor_get();
  s->set_brightness(s, 1);
  s->set_contrast(s, 1);
  s->set_saturation(s, 0);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);       // 0 = auto white balance
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 0);
  s->set_gain_ctrl(s, 1);
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)0);
  s->set_bpc(s, 0);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_hmirror(s, 0);
  s->set_vflip(s, 0);
  s->set_dcw(s, 1);
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

  // Buat nama file dengan timestamp (uptime ms sebagai pengganti RTC)
  char filename[40];
  snprintf(filename, sizeof(filename), "padi_%s_%lu.jpg",
           WiFi.macAddress().c_str(), millis());

  // Kirim ke Cloud Function via HTTPS
  WiFiClientSecure client;
  client.setInsecure(); // OK untuk development — ganti dengan CA cert untuk produksi

  HTTPClient http;
  if (!http.begin(client, GCP_ENDPOINT)) {
    Serial.println("[HTTP] Gagal begin");
    esp_camera_fb_return(fb);
    failCount++;
    return false;
  }

  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Filename",   filename);
  http.addHeader("X-Device-ID",  WiFi.macAddress());
  http.setTimeout(15000); // 15 detik timeout

  Serial.printf("[HTTP] Mengirim ke GCP (%s)...\n", GCP_ENDPOINT);
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
