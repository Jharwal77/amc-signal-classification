import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from amc.cnn import AMCNet
from amc.cnn_dataset import AMCDataset


SEED = 42
DATA_PATH = "data/demo.npz"
MODEL_PATH = "models/cnn.pth"

dataset = AMCDataset(DATA_PATH)

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(SEED)

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model = AMCNet(num_classes=len(dataset.CLASS_NAMES)).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for signals, labels in test_loader:
        signals = signals.to(device)

        outputs = model(signals)
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_predictions.extend(predictions)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_predictions)

print()
print("=" * 60)
print("CNN TEST RESULTS")
print("=" * 60)
print(f"Test samples: {len(all_labels)}")
print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print()

print("Classification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.CLASS_NAMES,
        zero_division=0,
    )
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
)

print("Confusion Matrix:")
print("Classes:", dataset.CLASS_NAMES)
print(cm)

np.save("models/cnn_confusion_matrix.npy", cm)

print()
print("Confusion matrix saved:")
print("models/cnn_confusion_matrix.npy")
