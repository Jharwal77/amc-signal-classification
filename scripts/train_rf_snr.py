import os
import random
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from amc.preprocessing import to_features


SEED = 42
DATA_PATH = "data/snr_dataset.npz"
MODEL_PATH = "models/rf_snr.joblib"

os.makedirs("models", exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

print("=" * 70)
print("RANDOM FOREST - CONTROLLED SNR EXPERIMENT")
print("=" * 70)

data = np.load(DATA_PATH, allow_pickle=True)

X_raw = data["X"]
y = data["y"].astype(str)
meta = data["meta"]

snr = np.array([
    float(item["dataset_snr_db"])
    for item in meta
])

print("Signals:", X_raw.shape)
print("Extracting features...")

X_features = np.asarray([
    to_features(signal)
    for signal in X_raw
], dtype=np.float32)

print("Feature matrix:", X_features.shape)

# ---------------------------------------------------------
# Same deterministic split strategy
# ---------------------------------------------------------

indices = np.arange(len(X_raw))

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

# The CNN used 70/15/15.
# For RF we use the same 70% training portion and
# evaluate on the remaining 30% for a baseline comparison.
#
# For the final comparison we will use a common test
# partition in the next experiment.

X_train = X_features[train_indices]
y_train = y[train_indices]

X_test = X_features[test_indices]
y_test = y[test_indices]

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# ---------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced",
)

print()
print("Training Random Forest...")

model.fit(
    X_train,
    y_train,
)

print("Training complete.")

# ---------------------------------------------------------
# Overall evaluation
# ---------------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print()
print("=" * 70)
print("OVERALL RF TEST RESULT")
print("=" * 70)

print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
)

# ---------------------------------------------------------
# SNR-wise evaluation
# ---------------------------------------------------------

test_snr = snr[test_indices]

snr_values = sorted(
    np.unique(test_snr)
)

results = {}

print()
print("=" * 70)
print("RF SNR-WISE RESULTS")
print("=" * 70)

for snr_value in snr_values:

    mask = test_snr == snr_value

    snr_accuracy = accuracy_score(
        y_test[mask],
        predictions[mask],
    )

    results[float(snr_value)] = float(
        snr_accuracy
    )

    print(
        f"SNR {snr_value:>5.0f} dB | "
        f"Samples: {np.sum(mask):4d} | "
        f"Accuracy: {snr_accuracy * 100:6.2f}%"
    )

# ---------------------------------------------------------
# Save model and results
# ---------------------------------------------------------

joblib.dump(
    model,
    MODEL_PATH,
)

np.savez(
    "models/rf_snr_results.npz",
    snr=np.array(list(results.keys())),
    accuracy=np.array(list(results.values())),
)

cm = confusion_matrix(
    y_test,
    predictions,
)

np.save(
    "models/rf_snr_confusion_matrix.npy",
    cm,
)

print()
print("Model saved:", MODEL_PATH)
print("SNR results saved: models/rf_snr_results.npz")
print("Confusion matrix saved: models/rf_snr_confusion_matrix.npy")
print()
print("RF experiment complete.")
