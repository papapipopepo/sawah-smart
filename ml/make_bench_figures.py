"""
Generate figures for Bab 4.5 (Field Validation).

Outputs 3 PNG:
- Fig 4.5.2 Gatekeeper.png         : VLM gatekeeper FAR (non-rice) and FRR (rice)
- Fig 4.5.3 Domain Shift.png       : paired Mature Recall test vs bench per model
- Fig 4.5.3 VLM Rejections.png     : 2x2 grid of bench Mature images rejected by VLM mini

Data source: ml/vlm_predictions.json (includes test, non_rice_probe, bench_mature).
Supervised bench numbers come from running ml/eval_bench.py (0/51 SVM, 51/51 Hybrid).
Note: eval_bench.py originally decoded SVM predictions by indexing the raw
integer output into CLASS_ORDER positionally, but the SVM's LabelEncoder sorts
classes alphabetically (Generative=0, Mature=1, Vegetative=2), a different
order than CLASS_ORDER. That mismatch inflated the original SVM bench figure
to 47/51; fixed 2026-07-07 by decoding through the label encoder.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path("bab4_figures")
OUT.mkdir(exist_ok=True)
DATA = json.load(open("vlm_predictions.json"))

# Cohort & supervised numbers (mirror Bab 4 final cohort)
VLM_COHORT = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gpt-4o", "gpt-4o-mini"]
VLM_LABEL = {
    "gemini-2.5-flash":      "Gemini Flash",
    "gemini-2.5-flash-lite": "Gemini Flash Lite",
    "gpt-4o":                "GPT-4o",
    "gpt-4o-mini":           "GPT-4o mini",
}
SUPERVISED_BENCH = {
    "Hybrid+ExGR":   (51, 51),  # 51/51 = 1.000
    "SVM-RBF+ExGR":  (0, 51),   # 0/51 = 0.000 (label-decode bug fixed 2026-07-07)
}
SUPERVISED_TEST = {
    "Hybrid+ExGR":   0.9553,  # bootstrap mean (canonical, see thesis/canonical_numbers.md)
    "SVM-RBF+ExGR":  0.9507,  # bootstrap mean (canonical)
}


def partition(prefix: str) -> dict:
    return {k: v for k, v in DATA.items() if k.startswith(prefix)}


bench = partition("bench_mature")
nonrice = partition("non_rice_probe")
test = {k: v for k, v in DATA.items() if k.startswith("test")}

# ── Fig 4.5.2 Gatekeeper FAR + FRR ────────────────────────────────────────────

def is_rejected(r: dict) -> bool:
    """Treat anything that is not an explicit accept as a reject.
    Covers accepted=false, predicted=N/A, and accepted=None (parser returned null fields).
    Matches the convention used by Tabel 4.2 where Flash Lite Accept rate = 0.952 (5 rejects)."""
    return r.get("predicted") == "N/A" or r.get("accepted") is not True


def gatekeeper_stats():
    out = {}
    for m in VLM_COHORT:
        # FRR on rice test set: not-accepted / total responded
        rice_resp = rice_reject = 0
        for v in test.values():
            r = v["results"].get(m)
            if not r or r.get("error"): continue
            rice_resp += 1
            if is_rejected(r):
                rice_reject += 1
        frr = rice_reject / rice_resp if rice_resp else 0
        # FAR on non-rice: accepted / total responded
        nr_resp = nr_accept = 0
        for v in nonrice.values():
            r = v["results"].get(m)
            if not r or r.get("error"): continue
            nr_resp += 1
            if not is_rejected(r):
                nr_accept += 1
        far = nr_accept / nr_resp if nr_resp else 0
        out[m] = {"frr": frr, "frr_n": rice_resp, "far": far, "far_n": nr_resp}
    return out


def fig_gatekeeper():
    stats = gatekeeper_stats()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(VLM_COHORT))
    w = 0.35
    frrs = [stats[m]["frr"] * 100 for m in VLM_COHORT]
    fars = [stats[m]["far"] * 100 for m in VLM_COHORT]
    b1 = ax.bar(x - w / 2, frrs, w, label="False Reject Rate (rice test, n=104)",
                color="#d62728", alpha=0.85)
    b2 = ax.bar(x + w / 2, fars, w, label="False Accept Rate (non-rice probe, n=12)",
                color="#1f77b4", alpha=0.85)
    for rect, val in zip(b1, frrs):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.15,
                f"{val:.2f}%", ha="center", fontsize=9)
    for rect, val in zip(b2, fars):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.15,
                f"{val:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([VLM_LABEL[m] for m in VLM_COHORT])
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, max(max(frrs), max(fars)) + 2 if max(frrs + fars) > 0 else 5)
    ax.set_title("VLM gatekeeper performance on rice test set and non-rice probes")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = OUT / "Fig 4.5.2 Gatekeeper.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ── Fig 4.5.3 Domain Shift (test vs bench Mature Recall) ──────────────────────

def bench_mat_recall(m: str) -> tuple[int, int]:
    """Mature Recall on bench: numerator counts only predicted=Mature, denominator
    is full bench size (51). API errors and rejects both count against recall to
    match the convention used by Tabel 4.5 (Flash Lite reports 45/51 = 0.8824)."""
    correct = 0
    total = len(bench)
    for v in bench.values():
        r = v["results"].get(m)
        if not r or r.get("error"): continue
        if r.get("predicted") == "Mature":
            correct += 1
    return correct, total


def test_mat_recall(m: str) -> tuple[int, int]:
    """Mature Recall on curated test (n=21): same convention — denominator fixed
    to total true-Mature count, rejects and errors do not reduce the denominator."""
    correct = 0
    total = sum(1 for v in test.values() if v.get("true_label") == "Mature")
    for v in test.values():
        if v.get("true_label") != "Mature": continue
        r = v["results"].get(m)
        if not r or r.get("error"): continue
        if r.get("predicted") == "Mature":
            correct += 1
    return correct, total


def fig_domain_shift():
    labels = list(SUPERVISED_BENCH.keys()) + [VLM_LABEL[m] for m in VLM_COHORT]
    test_recall = []
    bench_recall = []
    # Supervised
    for sup in SUPERVISED_BENCH:
        c, n = SUPERVISED_BENCH[sup]
        bench_recall.append(c / n * 100)
        test_recall.append(SUPERVISED_TEST[sup] * 100)
    # VLM
    for m in VLM_COHORT:
        bc, bn = bench_mat_recall(m)
        tc, tn = test_mat_recall(m)
        bench_recall.append(bc / bn * 100 if bn else 0)
        test_recall.append(tc / tn * 100 if tn else 0)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, test_recall, w, label="Test set (public, n=21 Mature)",
                color="#1f77b4", alpha=0.85)
    b2 = ax.bar(x + w / 2, bench_recall, w, label="Bench-scale ESP32-CAM (n=51 Mature)",
                color="#2ca02c", alpha=0.85)
    for rect, val in zip(b1, test_recall):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=8)
    for rect, val in zip(b2, bench_recall):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mature Recall (%)")
    ax.set_ylim(0, 110)
    ax.axhspan(95, 100, alpha=0.08, color="green")
    ax.set_title("Paired Mature Recall: public test set vs bench-scale ESP32-CAM capture")
    ax.legend(loc="lower left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = OUT / "Fig 4.5.3 Domain Shift.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ── Fig 4.5.3 VLM Rejection Examples ──────────────────────────────────────────

def fig_rejections():
    """4-panel grid of bench Mature images rejected by GPT-4o mini."""
    rejected = []
    for k, v in bench.items():
        r = v["results"].get("gpt-4o-mini")
        if not r or r.get("error"): continue
        if r.get("predicted") == "N/A":
            rejected.append((k, r.get("reason", ""), v["results"].get("gpt-4o", {}).get("predicted", "?")))
    if not rejected:
        print("[skip] no rejected bench images")
        return
    sample = rejected[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (path, reason, gpt4o_pred) in zip(axes.flat, sample):
        try:
            img = Image.open(path)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"cannot load: {e}", ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        reason_short = reason if len(reason) <= 55 else reason[:52] + "..."
        cap = (f"True: Mature  |  GPT-4o mini -> N/A  |  GPT-4o -> {gpt4o_pred}\n"
               f'Mini: "{reason_short}"')
        ax.set_title(cap, fontsize=10, loc="center", pad=6)
    fig.suptitle("Bench Mature rejected by GPT-4o mini, accepted by GPT-4o",
                 fontsize=14, fontweight="bold", y=0.97)
    plt.subplots_adjust(top=0.90, bottom=0.02, hspace=0.20, wspace=0.08)
    out = OUT / "Fig 4.5.3 VLM Rejections.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    fig_gatekeeper()
    fig_domain_shift()
    fig_rejections()
    print(f"\nAll figures in {OUT.resolve()}")
