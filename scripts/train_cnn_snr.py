import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from amc.cnn import AMCNet
from amc.cnn_dataset import AMCDataset


SEED = 42
DATA_PATH = "data/snr_dataset.npz"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_snr.pth")

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AMC CNN - CONTROLLED SNR TRAINING")
print("=" * 60)
print("Device:", device)

dataset = AMCDataset(DATA_PATH)

train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Total samples:", len(dataset))
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

model = AMCNet(
    num_classes=len(dataset.CLASS_NAMES)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in train_loader:

        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(signals)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = total_loss / total
    train_accuracy = correct / total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for signals, labels in val_loader:

            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": dataset.CLASS_NAMES,
                "input_length": int(dataset.X.shape[1]),
                "seed": SEED,
            },
            MODEL_PATH,
        )

        print(
            f"  -> Best model saved "
            f"({best_val_accuracy * 100:.2f}%)"
        )

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)
print("Model:", MODEL_PATH)
print("=" * 60)
