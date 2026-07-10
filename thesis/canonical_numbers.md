# Canonical Numbers — Source of Truth

Catatan: file ini = **single source of truth** untuk semua angka di tesis.
Diturunkan langsung dari OUTPUT CODE, bukan dari narasi tesis.

## Sumber data primer

| Sumber                                                         | Isi                                                                            |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `thesis/Thesis_Template__1_/img/bootstrap_ci_test.csv`         | bootstrap mean + 95% CI 1000-resample untuk 46 supervised + 5 VLM              |
| `ml/vlm_predictions.json`                                      | per-image VLM prediction (104 test + 51 bench + 12 non-rice probe × 5 model)   |
| `thesis/parse_latency.py` (input `latency_logs.json`, `esp_serial.log`) | latency breakdown E2E + server stage                                   |
| `ml/eval_bench.py` (output di-hardcode di `ml/make_bench_figures.py`) | bench-scale Mature Recall supervised (Hybrid 51/51, SVM 0/51)           |
| `ml/eval_bench.py` line 47-49                                  | parameter GLCM aktual: `distances=[1, 3]`, 4 angles, `levels=256`              |
| `ml/eval_bench.py` line 60-71                                  | 10 statistik VI aktual                                                         |

## Dataset partition (lihat bab3.tex)

- Total working: **1140** (raw curated 519 → Roboflow aug 3x)
- CV pool: **1036** (347 Veg / 361 Gen / 328 Mat; rasio max 361/328 ≈ **1.10**)
- Held-out test: **104** (49 Veg / 34 Gen / 21 Mat; rasio 49/21 ≈ **2.33**)
- Bench mature: **51** (semua Mature, ESP32-CAM polybag, 4 polybag dalam 1 litter box)
- Non-rice probe: **12** (4 empty box + 4 hand + 4 non-rice plant)

## Tabel 4.1 — Top configs per track (TEST), point estimates dari CSV `f1_mean` / `mat_mean`

| Track  | Config                  | Test F1 | F1 CI [2.5, 97.5] | Mat Rec | Mat CI            |
| ------ | ----------------------- | ------- | ----------------- | ------- | ----------------- |
| Hybrid | EfficientNetB0 + ExGR   | 0.9662  | [0.9215, 1.0000]  | 0.9553  | [0.8398, 1.0000]  |
| Hybrid | EfficientNetB0 + NGRDI  | 0.9621  | [0.9133, 1.0000]  | 0.9553  | [0.8398, 1.0000]  |
| ML     | SVM (RBF) + ExGR        | 0.9577  | [0.9106, 0.9916]  | 0.9507  | [0.8500, 1.0000]  |
| DL     | EfficientNetB0          | 0.9532  | [0.8998, 0.9921]  | 0.9068  | [0.7619, 1.0000]  |
| Hybrid | EfficientNetB0 + VARI   | 0.9454  | [0.8898, 0.9912]  | 0.9553  | [0.8398, 1.0000]  |
| ML     | RandomForest + ExGR     | 0.9368  | [0.8831, 0.9811]  | 0.9030  | [0.7647, 1.0000]  |
| DL     | MobileNetV3-Small       | 0.9198  | [0.8497, 0.9732]  | 0.8591  | [0.6874, 1.0000]  |
| DL     | MobileNetV2             | 0.8821  | [0.8081, 0.9440]  | 0.7656  | [0.5625, 0.9444]  |

**Catatan revisi:** versi lama Tabel 4.1 punya error:
- SVM (RBF)+ExGR F1: ~~0.9593~~ → 0.9577 (Lampiran A + CSV setuju di 0.9577)
- SVM (RBF)+ExGR MatRec: ~~0.9533~~ → 0.9507
- Hybrid+ExGR MatRec: ~~0.9500~~ → 0.9553
- Hybrid+NGRDI F1: ~~0.9618~~ → 0.9621
- Hybrid+NGRDI MatRec: ~~0.9500~~ → 0.9553
- Hybrid+VARI F1: ~~0.9456~~ → 0.9454
- Hybrid+VARI MatRec: ~~0.9500~~ → 0.9553
- DL EffNetB0 F1: ~~0.9531~~ → 0.9532
- DL EffNetB0 MatRec: ~~0.9000~~ → 0.9068
- DL MobNetV3-S F1: ~~0.9210~~ → 0.9198
- DL MobNetV3-S MatRec: ~~0.8600~~ → 0.8591
- DL MobNetV2 F1: ~~0.8818~~ → 0.8821
- DL MobNetV2 MatRec: ~~0.7600~~ → 0.7656
- ML RF+ExGR F1: ~~0.9382~~ → 0.9368
- ML RF+ExGR MatRec: ~~0.9000~~ → 0.9030

## Tabel 4.2 — VLM zero-shot (TEST)

Dihitung langsung dari `vlm_predictions.json` over **accepted samples only**
(F1 computed atas accepted, Macro F1 dan Mature Recall sama-sama eksklusi rejected).

| Endpoint               | Accept | Macro F1 | F1 CI            | Mat Rec | Mat CI           | Acc.  |
| ---------------------- | ------ | -------- | ---------------- | ------- | ---------------- | ----- |
| Gemini 2.5 Flash Lite  | 0.952  | 0.941    | [0.9126, 0.9909] | 1.000   | [1.000, 1.000]   | 0.949 |
| GPT-4o mini            | 1.000  | 0.892    | [0.8649, 0.9637] | 0.952   | [0.9268, 1.000]  | 0.913 |
| GPT-4o                 | 0.990  | 0.831    | [0.8170, 0.9349] | 1.000   | [1.000, 1.000]   | 0.864 |
| Gemini 2.5 Flash       | 1.000  | 0.811    | [0.7950, 0.9180] | 1.000   | [1.000, 1.000]   | 0.846 |

**Generative → Mature confusion (per endpoint, dari accepted only):**
- Gemini 2.5 Flash:      14 / 34
- Gemini 2.5 Flash Lite:  3 / 31 (31 = accepted Gen frames)
- GPT-4o:                13 / 34
- GPT-4o mini:            7 / 34

## FRR & FAR Gatekeeper (canonical, dari vlm_predictions.json)

**FRR on curated rice test (n=104) per endpoint:**

| Endpoint               | Reject count | FRR     | Breakdown (Veg/Gen/Mat)  |
| ---------------------- | ------------ | ------- | ------------------------ |
| Gemini 2.5 Flash Lite  | 5/104        | 4.81%   | 1 / 3 / 1                |
| Gemini 2.5 Flash       | 0/104        | 0.00%   | 0 / 0 / 0                |
| GPT-4o                 | 1/104        | 0.96%   | 1 / 0 / 0                |
| GPT-4o mini            | 0/104        | 0.00%   | 0 / 0 / 0                |

**FAR on 12 non-rice probes per endpoint: 0.00% (semua endpoint reject 12/12).**

**Implikasi deployment:** deployment pakai Flash Lite, jadi FRR canon = **4.81%** (5/104), bukan 0.96%.

## Tabel 4.5 — Mature on ESP32-CAM bench (51 frames)

**Convention:** denominator = total true Mature on each partition (21 curated, 51 bench). Rejection by VLM gatekeeper counts as a miss.

| Classifier              | Curated (n=21) | Bench (n=51) | Δ (pp)  |
| ----------------------- | -------------- | ------------ | ------- |
| Hybrid + ExGR           | 0.9553         | 1.0000       | +4.5    |
| SVM (RBF) + ExGR        | 0.9507         | 0.0000       | -95.1   |
| GPT-4o                  | 1.0000         | 1.0000       | 0.0     |
| Gemini 2.5 Flash        | 1.0000         | 0.8824       | -11.8   |
| Gemini 2.5 Flash Lite   | 0.9524         | 0.8824       | -7.0    |
| GPT-4o mini             | 0.9524         | 0.4118       | -54.1   |

**Source:** `ml/make_bench_figures.py` after fixing `bench_mat_recall` and `test_mat_recall` to use denom=full true Mature count (matches Fig 4.5.3 Domain Shift).

**Catatan revisi vs versi lama:**
- Hybrid Curated: ~~0.9500~~ → 0.9553 → delta ~~+5.0~~ → **+4.5**
- SVM Curated: ~~0.9533~~ → 0.9507 → delta ~~-3.2~~ → **-2.9**
- Flash Lite Curated: ~~1.0000~~ (over accepted) → 0.9524 (over full 21) → delta ~~-11.8~~ → **-7.0**
- GPT-4o mini Curated: kept 0.9524 (consistent; GPT-4o mini accepted all 21 Mature curated, but 1 misclassified as Gen)

**Bug fix 2026-07-07 — SVM bench Mature Recall (~~47/51 = 0.9216~~ → 0/51 = 0.0000):**
`eval_bench.py`'s `evaluate_svm()` indexed the SVM's raw integer prediction directly into `CLASS_ORDER = ["Vegetative","Generative","Mature"]` (positional index, the convention the CNN track uses). The SVM's `LabelEncoder` sorts classes alphabetically instead (`Generative=0, Mature=1, Vegetative=2`), confirmed in the training notebook (`Thesis_ML_3Class (1).ipynb`, cell defining `le = LabelEncoder(); le.fit(CLASS_ORDER)`). The two orderings disagree at every index, so the script's Mature bucket (index 2) was actually counting raw Vegetative predictions. Decoding through the saved `label_encoder_3class.joblib` gives the real result: on the 51-frame ESP32-CAM bench set, the classical SVM predicts Vegetative for 47 frames, Generative for 4, and Mature for 0 — a complete failure to carry over from the curated test set (0.9507) to real ESP32-CAM captures, unlike Hybrid (0.9553 → 1.0000). Fix applied in `eval_bench.py` (decode via label encoder before evaluation) and `make_bench_figures.py` (`SUPERVISED_BENCH["SVM-RBF+ExGR"] = (0, 51)`); Fig 4.5.3 Domain Shift regenerated. The main CV/test benchmark tables (Tabel 4.1, canonical F1/Mature Recall) are unaffected: those numbers come from `classification_report`/`f1_score` computed on consistently-encoded `y_true`/`y_pred` pairs, not from this positional-index shortcut.

## Tabel 4.4 — Latency (parse_latency.py output verified)

| Stage                                       | n   | Median (ms) | p95 (ms) |
| ------------------------------------------- | --- | ----------- | -------- |
| End-to-end (ESP T1 → T_resp)                | 12  | 8570        | 29878    |
| Network and upload (HTTP+TLS+payload)       | 12  | 3688        | 10901    |
| Server total (T2 → T4)                      | 51  | 4006        | 7750     |
| JPEG decode                                 | 51  | 23          | 36       |
| Gatekeeper LLM + CNN compute                | 51  | 3865        | 7592     |
| Response wrap                               | 36  | 128         | 164      |
| Path: rejected (LLM only)                   | 15  | 4692        | 6488     |
| Path: accepted (LLM + Hybrid CNN)           | 36  | 3868        | 9284     |

Cloud Run cost (1 vCPU + 512 MiB, asia-southeast2): median ~$0.077 / 1k req. (Script default 1 GiB gives $0.080.)

## Lampiran A — Std maks RF, untuk pernyataan Bab 4 §1.4

CV F1 Std per RF config:
- ExGR+RF: 0.0141
- NGRDI+RF: 0.0171
- VARI+RF: 0.0167
- MGRVI+RF: 0.0109
- ExG+RF: 0.0128
- ExR+RF: 0.0181
- GLI+RF: 0.0145
- **RGBVI+RF: 0.0197 ← MAKSIMUM untuk RF**

**Revisi:** ~~"Std up to 0.0207 at Random Forest"~~ → **0.0197**.

Untuk konteks: maksimum CV Std seluruh classical pool = LogReg+RGBVI 0.0271. GradBoost+ExG = 0.0219.

## GLCM — parameter kanonis dari `ml/eval_bench.py` line 47-49

```python
glcm = graycomatrix(g, distances=[1, 3],
                    angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                    levels=256, symmetric=True, normed=True)
```

**Source of truth:**
- distances = `[1, 3]` (2 jarak: 1 px dan 3 px)
- angles = 4 (0, π/4, π/2, 3π/4)
- levels = **256** (BUKAN 32)
- symmetric = True, normed = True
- → per descriptor: 2 × 4 = **8 cells** → reduce by `mean` + `std` → 2 stats per descriptor
- 4 descriptors (contrast, correlation, energy, homogeneity) × 2 stats = **8 GLCM features per VI**

**Revisi Bab 3 §5.2:**
- ~~"linearly quantized into 32 gray levels"~~ → **"uses 256 gray levels"** (gray sudah uint8 native)
- ~~"unit pixel distance for four orientations"~~ → **"pixel distances of 1 and 3 with four orientations"**
- → Bab 2 §4.2 sudah benar, tidak perlu diubah.

## VI statistics — list kanonis dari `ml/eval_bench.py` line 60-71

```python
{
    "mean":       np.mean(v),
    "std":        np.std(v),
    "median":     np.median(v),
    "p10":        np.percentile(v, 10),
    "p25":        np.percentile(v, 25),
    "p75":        np.percentile(v, 75),
    "p90":        np.percentile(v, 90),
    "frac_gt0":   np.mean(v > 0),
    "frac_gt01":  np.mean(v > 0.1),
    "iqr":        p75 - p25,
}
```

**Source of truth (10 stats):** mean, std, median, p10, p25, p75, p90, frac_gt0, frac_gt01, iqr.

**Revisi Bab 3 §5.1:** ganti list ~~"mean, std, min, max, median, p25, p75, range, skewness, kurtosis"~~ dengan list aktual. Konsisten dengan Bab 4 feature importance (ExGR_std, ExGR_p10 muncul di Top-5).

## Hybrid hyperparameter — kanonis di Bab 3 (Tabel 3.4), Bab 2 cukup deskriptif

- Phase 1: backbone frozen, head only, LR 1e-3, 10 epochs
- Phase 2: unfreeze top 30 layers, LR 1e-5, max 25 epochs, early stop val Macro F1 patience 5
- Batch 32, optimizer Adam, loss categorical cross-entropy
- Class weights inverse class frequency

## VLM cost — verifikasi orde magnitudo

| Endpoint              | Cost / 1k call |
| --------------------- | -------------- |
| Gemini 2.5 Flash Lite | $0.045         |
| GPT-4o mini           | $0.068         |
| Gemini 2.5 Flash      | $0.20          |
| GPT-4o                | $1.13          |

Span Flash Lite → GPT-4o: 1.13 / 0.045 = **25× ≈ 1.4 orde magnitudo** (bukan "three orders").

Span classical ($0.003) → GPT-4o ($1.13) = 377× ≈ **2.6 orde magnitudo**.

Revisi bab4.tex:415: "three orders" → **"roughly 25-fold (about 1.4 orders of magnitude)"** untuk perbandingan Flash Lite vs GPT-4o; atau gunakan "more than two orders of magnitude" kalau ingin span supervised vs VLM.

## Regulasi — daftar kanonis untuk konsistensi abstrak ↔ body

Body (Bab 1, Bab 2 §11, Bab 3 §9.3, Bab 5) konsisten pakai 5 referensi:
1. UU 19/2016 (ITE)
2. PP 71/2019 (PSTE)
3. UU 27/2022 (PDP)
4. ISO/IEC 30141:2024
5. ITU-T Y.2060

Kedua abstrak (ID + EN) **harus** mencantumkan kelima referensi yang sama.

## Framing: smallholder vs bench-scale

Posisi resmi (dari memory `project_research_constraints`):
- Target sistem: smallholder farmer (petani gurem).
- Validasi: bench-scale (4 polybag dalam 1 litter box), BUKAN field smallholder.
- Abstrak + Bab 5 harus eksplisit: "designed-for smallholder, validated at bench-scale".
- Hindari frasa "in-field pada lahan petani gurem" karena tidak ada validasi in-field di lahan.

## Akhir file — semua bab harus konsisten dengan angka di sini.
