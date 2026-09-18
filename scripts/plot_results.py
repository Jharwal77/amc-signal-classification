import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


os.makedirs("results", exist_ok=True)

CLASS_NAMES = [
    "16QAM",
    "64QAM",
    "8PSK",
    "AM",
    "BPSK",
    "FM",
    "GFSK",
    "QPSK",
]


# =========================================================
# Final end-to-end visualizations
# =========================================================

# 3. End-to-End Modulation Confusion Matrix
# =========================================================

e2e = pd.read_csv(
    "models/end_to_end_results.csv"
)

# Detect actual column names robustly
print()
print("End-to-end columns:")
print(list(e2e.columns))

true_mod_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "true_modulation",
        "actual_modulation",
        "modulation_true",
        "true_label",
    ]
)

pred_mod_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "predicted_modulation",
        "pred_modulation",
        "modulation_pred",
        "predicted_label",
    ]
)

true_mod = e2e[true_mod_col].astype(str)
pred_mod = e2e[pred_mod_col].astype(str)

cm_e2e = confusion_matrix(
    true_mod,
    pred_mod,
    labels=CLASS_NAMES,
)

fig, ax = plt.subplots(figsize=(9, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm_e2e,
    display_labels=CLASS_NAMES,
)

display.plot(
    ax=ax,
    xticks_rotation=45,
)

ax.set_title(
    "End-to-End Modulation Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/end_to_end_modulation_confusion_matrix.png",
    dpi=300,
)

plt.close()

print(
    "Saved: results/end_to_end_modulation_confusion_matrix.png"
)


# =========================================================
# 4. End-to-End Modulation Accuracy by Class
# =========================================================

class_accuracy = []

for label in CLASS_NAMES:

    mask = true_mod == label

    if mask.sum() == 0:
        acc = 0.0
    else:
        acc = (
            pred_mod[mask] == label
        ).mean() * 100

    class_accuracy.append(acc)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    CLASS_NAMES,
    class_accuracy,
)

plt.xlabel("Modulation")
plt.ylabel("Accuracy (%)")
plt.title(
    "End-to-End Modulation Classification Accuracy by Class"
)

plt.ylim(0, 100)
plt.xticks(rotation=30)
plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    class_accuracy
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.tight_layout()

plt.savefig(
    "results/end_to_end_modulation_accuracy_by_class.png",
    dpi=300,
)

plt.close()

print(
    "Saved: results/end_to_end_modulation_accuracy_by_class.png"
)


# =========================================================
# 5. SNR MAE vs True SNR
# =========================================================

true_snr_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "true_snr_db",
    ]
)

pred_snr_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "pred_snr_db",
    ]
)

true_snr = e2e[true_snr_col].astype(float)
pred_snr = e2e[pred_snr_col].astype(float)

snr_values = sorted(
    true_snr.unique()
)

snr_mae = []

for value in snr_values:

    mask = true_snr == value

    mae = np.mean(
        np.abs(
            pred_snr[mask]
            - true_snr[mask]
        )
    )

    snr_mae.append(mae)

plt.figure(figsize=(9, 6))

plt.plot(
    snr_values,
    snr_mae,
    marker="o",
    linewidth=2,
)

for x, y in zip(
    snr_values,
    snr_mae
):
    plt.annotate(
        f"{y:.3f}",
        (x, y),
        textcoords="offset points",
        xytext=(0, 8),
        ha="center",
    )

plt.xlabel("True SNR (dB)")
plt.ylabel("SNR MAE (dB)")
plt.title("End-to-End SNR Estimation Error")
plt.grid(True, alpha=0.3)
plt.xticks(snr_values)
plt.tight_layout()

plt.savefig(
    "results/end_to_end_snr_mae_vs_snr.png",
    dpi=300,
)

plt.close()

print(
    "Saved: results/end_to_end_snr_mae_vs_snr.png"
)


# =========================================================
# 6. CFO MAE vs CFO Range
# =========================================================

true_cfo_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "true_cfo_hz",
    ]
)

pred_cfo_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "pred_cfo_hz",
    ]
)

true_cfo = e2e[true_cfo_col].astype(float)
pred_cfo = e2e[pred_cfo_col].astype(float)

cfo_ranges = [
    (-3000, -2000),
    (-2000, -1000),
    (-1000, 0),
    (0, 1000),
    (1000, 2000),
    (2000, 3000),
]

range_labels = []
cfo_mae = []

for low, high in cfo_ranges:

    mask = (
        (true_cfo >= low)
        & (true_cfo < high)
    )

    if mask.sum() == 0:
        continue

    mae = np.mean(
        np.abs(
            pred_cfo[mask]
            - true_cfo[mask]
        )
    )

    range_labels.append(
        f"{low} to {high}"
    )

    cfo_mae.append(mae)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    range_labels,
    cfo_mae,
)

plt.xlabel("True CFO Range (Hz)")
plt.ylabel("CFO MAE (Hz)")
plt.title("End-to-End CFO Estimation Error")
plt.xticks(rotation=30)
plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    cfo_mae
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(cfo_mae) * 0.02,
        f"{value:.0f}",
        ha="center",
        fontsize=9,
    )

plt.tight_layout()

plt.savefig(
    "results/end_to_end_cfo_mae_by_range.png",
    dpi=300,
)

plt.close()

print(
    "Saved: results/end_to_end_cfo_mae_by_range.png"
)


# =========================================================
# 7. Symbol-Rate Confusion Matrix
# =========================================================

true_rate_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "true_symbol_rate_hz",
    ]
)

pred_rate_col = next(
    c for c in e2e.columns
    if c.lower() in [
        "pred_symbol_rate_hz",
    ]
)

true_rate = e2e[true_rate_col].astype(float)
pred_rate = e2e[pred_rate_col].astype(float)

RATE_VALUES = np.array([
    50_000.0,
    1e6 / 13,
    100_000.0,
    125_000.0,
    1e6 / 7,
    200_000.0,
])

RATE_NAMES = [
    "50 kHz",
    "76.923 kHz",
    "100 kHz",
    "125 kHz",
    "142.857 kHz",
    "200 kHz",
]

true_rate_class = [
    int(np.argmin(np.abs(RATE_VALUES - x)))
    for x in true_rate
]

pred_rate_class = [
    int(np.argmin(np.abs(RATE_VALUES - x)))
    for x in pred_rate
]

cm_rate = confusion_matrix(
    true_rate_class,
    pred_rate_class,
    labels=list(range(len(RATE_VALUES))),
)

fig, ax = plt.subplots(figsize=(9, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm_rate,
    display_labels=RATE_NAMES,
)

display.plot(
    ax=ax,
    xticks_rotation=45,
)

ax.set_title(
    "End-to-End Symbol-Rate Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "results/end_to_end_symbol_rate_confusion_matrix.png",
    dpi=300,
)

plt.close()

print(
    "Saved: results/end_to_end_symbol_rate_confusion_matrix.png"
)


# =========================================================
# 8. Overall System Performance Summary
# =========================================================

mod_accuracy = (
    np.mean(
        true_mod == pred_mod
    ) * 100
)

snr_overall_mae = np.mean(
    np.abs(
        pred_snr - true_snr
    )
)

cfo_overall_mae = np.mean(
    np.abs(
        pred_cfo - true_cfo
    )
)

symbol_accuracy = (
    np.mean(
        np.asarray(true_rate_class)
        == np.asarray(pred_rate_class)
    ) * 100
)

metrics = [
    "Modulation Accuracy (%)",
    "SNR MAE (dB)",
    "CFO MAE (Hz)",
    "Symbol Rate Accuracy (%)",
]

values = [
    mod_accuracy,
    snr_overall_mae,
    cfo_overall_mae,
    symbol_accuracy,
]

# Separate scales are required because the units differ.
fig, ax = plt.subplots(figsize=(10, 6))

text = (
    f"Modulation Accuracy : {mod_accuracy:.2f}%\n"
    f"SNR MAE             : {snr_overall_mae:.3f} dB\n"
    f"CFO MAE             : {cfo_overall_mae:.2f} Hz\n"
    f"Symbol Rate Accuracy: {symbol_accuracy:.2f}%"
)

ax.axis("off")

ax.text(
    0.5,
    0.5,
    text,
    ha="center",
    va="center",
    fontsize=18,
)

ax.set_title(
    "AI Wireless Signal Analyzer - Final End-to-End Performance",
    fontsize=16,
    pad=20,
)

plt.tight_layout()

plt.savefig(
    "results/final_system_performance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved: results/final_system_performance.png"
)


# =========================================================
# Complete
# =========================================================

print()
print("=" * 70)
print("FINAL VISUALIZATION GENERATION COMPLETE")
print("=" * 70)

print()
print("Generated files:")

for filename in sorted(
    os.listdir("results")
):
    if filename.endswith(".png"):
        print(
            " - results/" + filename
        )
