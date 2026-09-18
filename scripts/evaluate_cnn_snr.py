import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from amc.cnn import AMCNet
from amc.cnn_dataset import AMCDataset


DATA_PATH = "data/snr_dataset.npz"
MODEL_PATH = "models/cnn_snr.pth"

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

SEED = 42
TEST_FRACTION = 0.15
TRAIN_FRACTION = 0.70


print("=" * 70)
print("CNN SNR-WISE TEST EVALUATION")
print("=" * 70)

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"].astype(str)
meta = data["meta"]

# Recover SNR from metadata
snr_values = np.array([
    float(m["dataset_snr_db"])
    for m in meta
])

label_to_index = {
    label: i
    for i, label in enumerate(CLASS_NAMES)
}

y_index = np.array([
    label_to_index[label]
    for label in y
])

# Reproduce the exact split used during training
rng = np.random.default_rng(SEED)

indices = np.arange(len(X))
rng.shuffle(indices)

train_size = int(TRAIN_FRACTION * len(X))
val_size = int(TEST_FRACTION * len(X))

test_indices = indices[train_size + val_size:]

print("Total samples:", len(X))
print("Test samples:", len(test_indices))
print("SNR levels:", sorted(np.unique(snr_values)))

# Load model
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model = AMCNet(
    num_classes=len(CLASS_NAMES)
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Device:", device)
print()

results = {}

with torch.no_grad():

    for snr in sorted(np.unique(snr_values)):

        mask = (
            snr_values[test_indices] == snr
        )

        indices_snr = test_indices[mask]

        signals = X[indices_snr]

        # Complex -> I/Q
        iq = np.stack(
            [
                signals.real,
                signals.imag
            ],
            axis=1
        ).astype(np.float32)

        # Same normalization used by AMCDataset
        scale = np.sqrt(
            np.mean(iq ** 2, axis=2, keepdims=True)
        )

        iq = iq / np.maximum(scale, 1e-12)

        tensor = torch.from_numpy(iq).to(device)

        outputs = model(tensor)

        predictions = (
            outputs.argmax(dim=1)
            .cpu()
            .numpy()
        )

        true_labels = y_index[indices_snr]

        accuracy = accuracy_score(
            true_labels,
            predictions
        )

        results[float(snr)] = float(accuracy)

        print(
            f"SNR {snr:>5.0f} dB | "
            f"Samples: {len(indices_snr):4d} | "
            f"Accuracy: {accuracy * 100:6.2f}%"
        )

print()
print("=" * 70)
print("SNR-WISE RESULTS")
print("=" * 70)

for snr, accuracy in results.items():
    print(
        f"{snr:>5.0f} dB : "
        f"{accuracy * 100:.2f}%"
    )

np.savez(
    "models/cnn_snr_results.npz",
    snr=np.array(list(results.keys())),
    accuracy=np.array(list(results.values())),
)

print()
print("Results saved:")
print("models/cnn_snr_results.npz")

# Overall test evaluation
test_signals = X[test_indices]

iq = np.stack(
    [
        test_signals.real,
        test_signals.imag
    ],
    axis=1
).astype(np.float32)

scale = np.sqrt(
    np.mean(iq ** 2, axis=2, keepdims=True)
)

iq = iq / np.maximum(scale, 1e-12)

with torch.no_grad():
    outputs = model(
        torch.from_numpy(iq).to(device)
    )

predictions = (
    outputs.argmax(dim=1)
    .cpu()
    .numpy()
)

true_labels = y_index[test_indices]

overall_accuracy = accuracy_score(
    true_labels,
    predictions
)

print()
print("=" * 70)
print("OVERALL TEST RESULT")
print("=" * 70)
print(
    f"Test accuracy: "
    f"{overall_accuracy * 100:.2f}%"
)

print()
print("Classification Report:")
print(
    classification_report(
        true_labels,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

cm = confusion_matrix(
    true_labels,
    predictions,
)

np.save(
    "models/cnn_snr_confusion_matrix.npy",
    cm
)

print("Confusion matrix saved:")
print("models/cnn_snr_confusion_matrix.npy")
