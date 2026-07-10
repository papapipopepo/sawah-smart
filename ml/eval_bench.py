"""
Evaluate trained supervised models on ESP32-CAM bench-scale captures.

Goal (Bab 4.5.3): paired comparison of Mature Recall on real bench-scale rice
images vs the public-dataset test split. Detects domain shift between training
distribution (HF + Roboflow public) and deployment-like ESP32-CAM proximal capture.

Models evaluated:
- SVM-RBF + ExGR (models/best_3class_ExGR_SVM_RBF.joblib)
- Hybrid EfficientNetB0 + ExG/ExGR VI (models/best_hybrid_cnn_ExG.keras)

Input: ml/bench_mature/Mature/*.jpg  (all true label = "Mature")
Output: per-model confusion + Mature Recall + sample mispredictions printed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import numpy as np

CLASS_ORDER = ["Vegetative", "Generative", "Mature"]
MATURE_IDX = 2
IMG_SIZE = 224  # Hybrid input

# ── Feature extraction (mirrors Thesis_ML_3Class + Thesis_DL_Hybrid_3Class) ──

def read_rgb(path: Path) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_ExGR(rgb: np.ndarray) -> np.ndarray:
    r = rgb.astype(np.float32) / 255.0
    R, G, B = r[:, :, 0], r[:, :, 1], r[:, :, 2]
    return (2 * G - R - B) - (1.4 * R - G)


def compute_glcm_features(gray: np.ndarray) -> dict:
    from skimage.feature import graycomatrix, graycoprops
    g = gray.astype(np.uint8)
    glcm = graycomatrix(g, distances=[1, 3],
                        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=256, symmetric=True, normed=True)
    feats = {}
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        vals = graycoprops(glcm, prop).flatten()
        feats[f"glcm_{prop}_mean"] = float(np.mean(vals))
        feats[f"glcm_{prop}_std"] = float(np.std(vals))
    return feats


def extract_vi_stats(vi_1d: np.ndarray) -> dict:
    v = vi_1d.flatten()
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p25": float(np.percentile(v, 25)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "frac_gt0": float(np.mean(v > 0)),
        "frac_gt01": float(np.mean(v > 0.1)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
    }


STAT_ORDER = list(extract_vi_stats(np.array([0.0])).keys())  # 10
GLCM_ORDER = [f"glcm_{p}_{s}" for p in ["contrast", "correlation", "energy", "homogeneity"]
              for s in ["mean", "std"]]                       # 8
N_VI_FEATURES = len(STAT_ORDER) + len(GLCM_ORDER)             # 18


def extract_features_one(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vi_feature_vector[18], rgb_image_resized_224x224x3)."""
    import cv2
    rgb = read_rgb(path)
    # VI features (ExGR stats + GLCM on original gray)
    exgr = compute_ExGR(rgb)
    stats = extract_vi_stats(exgr)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    glcm = compute_glcm_features(gray)
    vi_vec = np.array([stats[k] for k in STAT_ORDER]
                      + [glcm[k] for k in GLCM_ORDER], dtype=np.float32)
    # Resized image for Hybrid
    img_resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return vi_vec, img_resized.astype(np.float32)


# ── Evaluation ──

def evaluate_svm(joblib_path: Path, X_vi: np.ndarray, paths: list[Path]) -> tuple[np.ndarray, str]:
    import joblib
    clf = joblib.load(joblib_path)
    enc_path = joblib_path.parent / "label_encoder_3class.joblib"
    raw_pred = clf.predict(X_vi)
    # SVM's LabelEncoder sorts classes alphabetically (Generative=0, Mature=1,
    # Vegetative=2), which does NOT match CLASS_ORDER's positional convention
    # used by the CNN track. Decode through the encoder, then remap to
    # CLASS_ORDER indices so summarize() stays convention-agnostic.
    le = joblib.load(enc_path)
    labels = le.inverse_transform(raw_pred)
    y_pred = np.array([CLASS_ORDER.index(lbl) for lbl in labels])
    return y_pred, f"SVM-RBF + ExGR ({joblib_path.name})"


def evaluate_hybrid(keras_path: Path, X_img: np.ndarray, X_vi: np.ndarray,
                    paths: list[Path]) -> tuple[np.ndarray, str]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path, compile=False)
    probs = model.predict([X_img, X_vi], batch_size=8, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    return y_pred, f"Hybrid EffNetB0 + ExGR ({keras_path.name})"


def summarize(name: str, y_pred: np.ndarray, paths: list[Path]) -> None:
    n = len(y_pred)
    counts = np.bincount(y_pred, minlength=len(CLASS_ORDER))
    mat_recall = counts[MATURE_IDX] / n
    print(f"\n=== {name} ===")
    print(f"  N = {n}  (all true = Mature)")
    for i, c in enumerate(CLASS_ORDER):
        bar = "#" * int(counts[i] / n * 40)
        print(f"  pred {c:11s}: {counts[i]:3d}/{n}  {bar}")
    print(f"  Mature Recall = {counts[MATURE_IDX]}/{n} = {mat_recall:.3f}")
    miss = [(p, CLASS_ORDER[y]) for p, y in zip(paths, y_pred) if y != MATURE_IDX]
    if miss:
        print(f"  Misclassified ({len(miss)}):")
        for p, pred in miss[:12]:
            print(f"    {p.name[-50:]:50s} -> {pred}")
        if len(miss) > 12:
            print(f"    ... and {len(miss) - 12} more")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default="bench_mature/Mature",
                    help="folder with bench-scale Mature images (default: bench_mature/Mature)")
    ap.add_argument("--svm", default="models/best_3class_ExGR_SVM_RBF.joblib")
    ap.add_argument("--hybrid", default="models/best_hybrid_cnn_ExG.keras")
    ap.add_argument("--skip-hybrid", action="store_true",
                    help="skip Hybrid model (avoid TF load if not needed)")
    args = ap.parse_args()

    bench_dir = Path(args.bench_dir)
    paths = sorted(bench_dir.glob("*.jpg")) + sorted(bench_dir.glob("*.jpeg")) + sorted(bench_dir.glob("*.png"))
    if not paths:
        print(f"[FATAL] no images in {bench_dir.resolve()}")
        return 1
    print(f"Loading {len(paths)} images from {bench_dir} ...")

    X_vi = np.zeros((len(paths), N_VI_FEATURES), dtype=np.float32)
    X_img = np.zeros((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        try:
            vi, img = extract_features_one(p)
            X_vi[i] = vi
            X_img[i] = img
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
    print(f"Features extracted: VI={X_vi.shape}, IMG={X_img.shape}")

    svm_path = Path(args.svm)
    hyb_path = Path(args.hybrid)
    if svm_path.exists():
        y_pred, name = evaluate_svm(svm_path, X_vi, paths)
        summarize(name, y_pred, paths)
    else:
        print(f"[skip] SVM joblib not found at {svm_path}")

    if not args.skip_hybrid:
        if hyb_path.exists():
            y_pred, name = evaluate_hybrid(hyb_path, X_img, X_vi, paths)
            summarize(name, y_pred, paths)
        else:
            print(f"[skip] Hybrid keras not found at {hyb_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
