import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from amc.cnn import AMCNet

DATA_PATH = "data/snr_dataset.npz"
SPLIT_PATH = "data/common_split.npz"
MODEL_PATH = "models/cnn_common.pth"

CLASS_NAMES = ["16QAM", "64QAM", "8PSK", "AM", "BPSK", "FM", "GFSK", "QPSK"]


class CommonTestDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]

        iq = np.stack(
            [signal.real, signal.imag],
            axis=0
        ).astype(np.float32)

        scale = np.sqrt(np.mean(iq ** 2))

        if scale > 0:
            iq = iq / scale

        label = CLASS_NAMES.index(str(self.y[idx]))

        return (
            torch.from_numpy(iq),
            torch.tensor(label, dtype=torch.long)
        )


print("=" * 70)
print("CNN - COMMON BENCHMARK EVALUATION")
print("=" * 70)

# Load dataset
data = np.load(DATA_PATH, allow_pickle=True)
X = data["X"]
y = data["y"].astype(str)

# Load common test split
split = np.load(SPLIT_PATH, allow_pickle=True)
test_idx = split["test_idx"]

X_test = X[test_idx]
y_test = y[test_idx]

print(f"Total dataset: {len(X)}")
print(f"Test samples : {len(X_test)}")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AMCNet(num_classes=len(CLASS_NAMES))

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

print(f"Device       : {device}")
print(f"Model        : {MODEL_PATH}")

# Test loader
dataset = CommonTestDataset(X_test, y_test)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False
)

# Prediction
all_predictions = []
all_labels = []

with torch.no_grad():
    for signals, labels in loader:

        signals = signals.to(device)

        outputs = model(signals)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

all_predictions = np.asarray(all_predictions)
all_labels = np.asarray(all_labels)

# Overall accuracy
accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print()
print("=" * 70)
print("COMMON TEST RESULT")
print("=" * 70)

print(f"CNN accuracy: {accuracy * 100:.2f}%")

# Classification report
print()
print("CLASSIFICATION REPORT")
print("-" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)

# SNR-wise accuracy
meta = data["meta"]

test_snrs = np.array([
    float(meta[i]["snr_db"])
    for i in test_idx
])

snr_values = sorted(
    np.unique(test_snrs)
)

snr_results = []

print()
print("=" * 70)
print("SNR-WISE CNN ACCURACY")
print("=" * 70)

for snr in snr_values:

    mask = test_snrs == snr

    snr_acc = accuracy_score(
        all_labels[mask],
        all_predictions[mask]
    )

    count = int(np.sum(mask))

    snr_results.append([
        snr,
        snr_acc * 100,
        count
    ])

    print(
        f"SNR {snr:>5.0f} dB : "
        f"{snr_acc * 100:6.2f}% "
        f"({count} samples)"
    )

# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

# Save results
os.makedirs("models", exist_ok=True)

np.save(
    "models/cnn_common_predictions.npy",
    all_predictions
)

np.save(
    "models/cnn_common_test_labels.npy",
    all_labels
)

np.save(
    "models/cnn_common_confusion_matrix.npy",
    cm
)

np.save(
    "models/cnn_common_snr_results.npy",
    np.asarray(snr_results)
)

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print("models/cnn_common_predictions.npy")
print("models/cnn_common_test_labels.npy")
print("models/cnn_common_confusion_matrix.npy")
print("models/cnn_common_snr_results.npy")

print()
print("=" * 70)
print("CNN COMMON EVALUATION COMPLETE")
print("=" * 70)
