import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from amc.preprocessing import to_features


DATA_PATH = "data/snr_dataset.npz"
SPLIT_PATH = "data/common_split.npz"
MODEL_PATH = "models/rf_common.pjoblib"

data = np.load(DATA_PATH, allow_pickle=True)
split = np.load(SPLIT_PATH)

X_raw = data["X"]
y = data["y"].astype(str)
meta = data["meta"]

train_idx = split["train_idx"]
test_idx = split["test_idx"]

print("=" * 60)
print("RANDOM FOREST - COMMON BENCHMARK")
print("=" * 60)

print("Extracting 15 features from 14,000 signals...")

X = np.asarray(
    [to_features(signal) for signal in X_raw],
    dtype=np.float32,
)

print("Feature matrix:", X.shape)

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("Training:", len(X_train))
print("Testing :", len(X_test))

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

print()
print("Training Random Forest...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions,
)

print()
print("=" * 60)
print("COMMON TEST RESULT")
print("=" * 60)
print(
    f"RF accuracy: {accuracy * 100:.2f}%"
)

joblib.dump(
    model,
    MODEL_PATH,
)

print("Model saved:", MODEL_PATH)

# SNR-wise accuracy
test_snr = np.array([
    float(meta[i]["dataset_snr_db"])
    for i in test_idx
])

print()
print("SNR-WISE RF ACCURACY")
print("-" * 40)

snr_results = []

for snr in sorted(np.unique(test_snr)):

    mask = test_snr == snr

    acc = accuracy_score(
        y_test[mask],
        predictions[mask],
    )

    snr_results.append(
        [float(snr), float(acc)]
    )

    print(
        f"{snr:>5.0f} dB : "
        f"{acc * 100:.2f}%"
    )

snr_results = np.asarray(snr_results)

np.save(
    "models/rf_common_snr_results.npy",
    snr_results,
)

np.save(
    "models/rf_common_predictions.npy",
    predictions,
)

np.save(
    "models/rf_common_test_labels.npy",
    y_test,
)

print()
print("Saved:")
print("models/rf_common_snr_results.npy")
print("models/rf_common_predictions.npy")
print("models/rf_common_test_labels.npy")
print()
print("Common RF benchmark complete.")
