#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <time.h>
#include "esp_camera.h"
#include "esp_sleep.h"

// ============================================================
// KONFIGURASI
// ============================================================
#define WIFI_SSID       "ALZRA 2.4GHz"
#define WIFI_PASSWORD   "Alzra1111"

// URL Cloud Run setelah deploy
#define GCP_BASE_URL    "https://upload-padi-je6fikdiha-et.a.run.app"
#define GCP_UPLOAD      GCP_BASE_URL "/upload"

// Deep sleep & capture
#define SLEEP_SECONDS        600     // tidur 10 menit setelah siklus sukses
#define SLEEP_RETRY_SECONDS  180     // tidur 3 menit kalau kamera/WiFi gagal, lalu coba lagi
#define PHOTOS_PER_CYCLE     3       // jumlah foto per siklus
#define CAPTURE_GAP_MS       15000UL // jeda antar foto (15 detik — variasi exposure)
#define WARMUP_DELAY_MS      15000UL // jeda 15 detik setelah bangun sebelum foto pertama

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
bool connectWiFi();
bool captureAndUpload(int photoIndex, int totalPhotos);
void goToDeepSleep(int seconds);
void keepAliveSleep(int seconds);

// Bertahan melintasi deep sleep (disimpan di RTC memory)
RTC_DATA_ATTR int bootCount = 0;

int uploadCount = 0;
int failCount   = 0;

// ============================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  bootCount++;
  Serial.println("\n========================================");
  Serial.println("  ESP32-CAM Padi Harvest Detection");
  Serial.println("  Mode: Deep Sleep + Burst Capture");
  Serial.printf("  Siklus ke-%d | %d foto/siklus | tidur %d detik\n",
                bootCount, PHOTOS_PER_CYCLE, SLEEP_SECONDS);
  Serial.println("========================================");

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Init kamera — kalau gagal, tidur singkat lalu coba lagi (jangan tunggu 30 menit)
  if (!initCamera()) {
    Serial.printf("[ERROR] Kamera gagal init — tidur %d detik lalu coba lagi\n", SLEEP_RETRY_SECONDS);
    goToDeepSleep(SLEEP_RETRY_SECONDS);
  }
  Serial.println("[OK] Kamera siap");

  // Konek WiFi — kalau gagal, tidur singkat lalu coba lagi (foto tidak hilang lama)
  if (!connectWiFi()) {
    Serial.printf("[ERROR] WiFi gagal — tidur %d detik lalu coba lagi\n", SLEEP_RETRY_SECONDS);
    goToDeepSleep(SLEEP_RETRY_SECONDS);
  }

  // Warm-up: tunggu 15 detik biar sensor & WiFi stabil sebelum foto pertama
  Serial.printf("\n[WARMUP] Tunggu %lu detik sebelum foto pertama...\n", WARMUP_DELAY_MS / 1000UL);
  delay(WARMUP_DELAY_MS);

  // Burst capture: ambil & upload PHOTOS_PER_CYCLE foto
  Serial.printf("\n[CYCLE] Mulai burst capture %d foto...\n", PHOTOS_PER_CYCLE);
  for (int i = 1; i <= PHOTOS_PER_CYCLE; i++) {
    bool ok = captureAndUpload(i, PHOTOS_PER_CYCLE);

    if (i < PHOTOS_PER_CYCLE) {
      // Kalau foto gagal total, refresh koneksi WiFi sebelum foto berikutnya
      if (!ok) {
        Serial.println("[WiFi] Foto gagal — reconnect WiFi sebelum lanjut...");
        WiFi.disconnect();
        delay(500);
        connectWiFi();
      }
      delay(CAPTURE_GAP_MS);
    }
  }

  Serial.printf("\n[CYCLE] Selesai — Berhasil: %d | Gagal: %d\n", uploadCount, failCount);

  // Powerbank auto-cutoff kalau beban <60-100 mA → pakai light sleep + pulse flash
  // sebagai dummy load, biar powerbank tetap nyala tanpa bikin ESP overheat
  keepAliveSleep(SLEEP_SECONDS);
}

void loop() {
  // Tidak terpakai — semua kerja di setup(), lalu deep sleep me-reset chip
}

// ============================================================
// Idle window dengan WiFi STA tetap nyala → narik ~70-80 mA konstan,
// powerbank lihat beban terus-menerus jadi nggak auto-cutoff.
// Mitigasi panas: kamera di-deinit, CPU diturunkan ke 80 MHz, modem sleep dimatikan
// (biar radio benar-benar aktif, bukan duty cycle rendah yang dianggap idle).
// Setelah window habis, ESP.restart() biar state bersih (mirip wake dari deep sleep).
void keepAliveSleep(int seconds) {
  Serial.printf("[KEEPALIVE] Idle %d detik (WiFi on @ 80 MHz, kamera off)...\n", seconds);
  Serial.flush();

  esp_camera_deinit();
  setCpuFrequencyMhz(80);

  // Pastikan WiFi connected & radio aktif (bukan modem sleep)
  if (WiFi.status() != WL_CONNECTED) {
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int wait = 0;
    while (WiFi.status() != WL_CONNECTED && wait < 20) { delay(500); wait++; }
  }
  WiFi.setSleep(false);

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  // Tunggu N detik dengan log periodik (chunk 30 dtk atau seluruhnya kalau <30 dtk)
  uint32_t total_s = (uint32_t)seconds;
  uint32_t chunk_log = 30;
  for (uint32_t s = 0; s < total_s; s += chunk_log) {
    uint32_t chunk = (total_s - s >= chunk_log) ? chunk_log : (total_s - s);
    delay(chunk * 1000UL);
    Serial.printf("[KEEPALIVE] %u/%u dtk | heap: %u | RSSI: %d dBm\n",
                  s + chunk, total_s, ESP.getFreeHeap(), WiFi.RSSI());
  }

  setCpuFrequencyMhz(240);
  Serial.println("[KEEPALIVE] Selesai — restart untuk siklus berikutnya");
  Serial.flush();
  ESP.restart();
}

// ============================================================
void goToDeepSleep(int seconds) {
  Serial.printf("[SLEEP] Deep sleep %d detik...\n", seconds);
  Serial.flush();

  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);

  esp_sleep_enable_timer_wakeup((uint64_t)seconds * 1000000ULL);
  esp_deep_sleep_start();
  // Tidak ada kode setelah ini — chip reset saat bangun, kembali ke setup()
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

  // QXGA 2048x1536 — butuh PSRAM, file ~100-200KB
  if (psramFound()) {
    cfg.frame_size   = FRAMESIZE_QXGA;  // 2048x1536
    cfg.jpeg_quality = 10;              // 0-63, lower = better quality
    cfg.fb_count     = 1;
    Serial.println("[CAM] PSRAM ditemukan — pakai QXGA 2048x1536");
  } else {
    cfg.frame_size   = FRAMESIZE_VGA;   // 640x480 fallback
    cfg.jpeg_quality = 15;
    cfg.fb_count     = 1;
    Serial.println("[WARN] Tanpa PSRAM — fallback ke VGA (resolusi rendah)");
  }

  if (esp_camera_init(&cfg) != ESP_OK) return false;

  // Buang beberapa frame agar sensor stabil (exposure & AWB)
  for (int i = 0; i < 3; i++) {
    camera_fb_t *dummy = esp_camera_fb_get();
    if (dummy) esp_camera_fb_return(dummy);
    delay(100);
  }

  sensor_t *s = esp_camera_sensor_get();
  Serial.printf("[CAM] Sensor PID: 0x%04X", s->id.PID);
  switch (s->id.PID) {
    case OV2640_PID: Serial.println(" (OV2640 — 2MP)"); break;
    case OV3660_PID: Serial.println(" (OV3660 — 3MP) OK"); break;
    case OV5640_PID: Serial.println(" (OV5640 — 5MP)"); break;
    default:         Serial.println(" (sensor tidak dikenali)"); break;
  }

  s->set_brightness(s, 0);
  s->set_contrast(s, 0);
  s->set_saturation(s, 0);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_agc_gain(s, 0);
  s->set_gainceiling(s, (gainceiling_t)2);
  s->set_bpc(s, 1);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
  s->set_hmirror(s, 1);
  s->set_vflip(s, 1);
  s->set_dcw(s, 1);
  s->set_colorbar(s, 0);

  return true;
}

// ============================================================
bool connectWiFi() {
  Serial.printf("[WiFi] Menghubungkan ke: %s\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  // INADDR_NONE = tetap pakai DHCP untuk IP, tapi paksa DNS ke Google
  WiFi.config(INADDR_NONE, INADDR_NONE, INADDR_NONE,
              IPAddress(8, 8, 8, 8), IPAddress(8, 8, 4, 4));
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WiFi] Gagal terhubung");
    return false;
  }

  Serial.printf("\n[WiFi] Terhubung — IP: %s\n", WiFi.localIP().toString().c_str());
  configTime(7 * 3600, 0, "pool.ntp.org", "time.nist.gov");
  Serial.print("[NTP] Sinkronisasi waktu");
  struct tm t;
  for (int i = 0; i < 20 && !getLocalTime(&t); i++) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(getLocalTime(&t) ? " OK" : " Gagal (pakai millis)");
  return true;
}

// ============================================================
bool captureAndUpload(int photoIndex, int totalPhotos) {
  Serial.printf("\n[CAM] Foto %d/%d — free heap: %u bytes\n",
                photoIndex, totalPhotos, ESP.getFreeHeap());

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

  // Nama file: padi_<MAC>_<timestamp>_<index>.jpg (index agar unik walau detik sama)
  char filename[64];
  struct tm t;
  if (getLocalTime(&t)) {
    char ts[20];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &t);
    snprintf(filename, sizeof(filename), "padi_%s_%s_%d.jpg",
             WiFi.macAddress().c_str(), ts, photoIndex);
  } else {
    snprintf(filename, sizeof(filename), "padi_%s_%lu_%d.jpg",
             WiFi.macAddress().c_str(), millis(), photoIndex);
  }

  // Retry upload hingga 3x
  bool success = false;
  const int MAX_RETRY = 3;

  for (int attempt = 1; attempt <= MAX_RETRY && !success; attempt++) {
    if (attempt > 1) {
      Serial.printf("[HTTP] Retry %d/%d — tunggu 3 detik...\n", attempt, MAX_RETRY);
      delay(3000);
    }

    WiFiClientSecure client;
    client.setInsecure(); // skip SSL cert — OK untuk development

    HTTPClient http;
    if (!http.begin(client, GCP_UPLOAD)) {
      Serial.println("[HTTP] Gagal begin — skip attempt");
      client.stop();
      continue;
    }

    http.addHeader("Content-Type", "image/jpeg");
    http.addHeader("X-Filename",   filename);
    http.addHeader("X-Device-ID",  WiFi.macAddress());
    http.setTimeout(60000); // 60 detik — QXGA butuh lebih lama dari default

    Serial.printf("[HTTP] Upload %s (attempt %d)...\n", filename, attempt);
    uint32_t t1 = millis();
    Serial.printf("[LATENCY] T1_trigger_ms=%lu file=%s attempt=%d\n", t1, filename, attempt);
    int code = http.POST(fb->buf, fb->len);
    uint32_t t_resp = millis();
    Serial.printf("[LATENCY] T_resp_ms=%lu elapsed_ms=%lu code=%d file=%s\n",
                  t_resp, t_resp - t1, code, filename);

    if (code > 0) {
      Serial.printf("[HTTP] Response: %d\n", code);
      String body = http.getString();
      Serial.println("[HTTP] Body: " + body);
      success = (code == 200 || code == 201);
    } else {
      Serial.printf("[HTTP] Error: %s\n", http.errorToString(code).c_str());
    }

    http.end();
    client.stop();
  }

  esp_camera_fb_return(fb);

  if (success) {
    uploadCount++;
    Serial.printf("[OK] Foto %d/%d berhasil (%s)\n", photoIndex, totalPhotos, filename);
  } else {
    failCount++;
    Serial.printf("[FAIL] Foto %d/%d gagal setelah %d percobaan\n",
                  photoIndex, totalPhotos, MAX_RETRY);
  }
  return success;
}
