import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from amc.cnn import AMCNet


SEED = 42
DATA_PATH = "data/snr_dataset.npz"
SPLIT_PATH = "data/common_split.npz"
MODEL_PATH = "models/cnn_common.pth"

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3

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

LABEL_TO_INDEX = {
    label: i for i, label in enumerate(CLASS_NAMES)
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class CommonAMCDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):

        signal = self.X[index]

        iq = np.stack(
            [
                signal.real,
                signal.imag
            ],
            axis=0,
        ).astype(np.float32)

        scale = np.sqrt(
            np.mean(iq ** 2)
        )

        if scale > 0:
            iq = iq / scale

        label = LABEL_TO_INDEX[
            str(self.y[index])
        ]

        return (
            torch.from_numpy(iq),
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


print("=" * 70)
print("CNN - COMMON BENCHMARK TRAINING")
print("=" * 70)

data = np.load(
    DATA_PATH,
    allow_pickle=True,
)

split = np.load(
    SPLIT_PATH,
)

X = data["X"]
y = data["y"].astype(str)

train_idx = split["train_idx"]
test_idx = split["test_idx"]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("Total:", len(X))
print("Training:", len(X_train))
print("Testing:", len(X_test))

train_dataset = CommonAMCDataset(
    X_train,
    y_train,
)

test_dataset = CommonAMCDataset(
    X_test,
    y_test,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

model = AMCNet(
    num_classes=len(CLASS_NAMES)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_train_accuracy = 0.0

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

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()
        optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = total_loss / total
    train_accuracy = correct / total

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}%"
    )

    if train_accuracy > best_train_accuracy:

        best_train_accuracy = train_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    CLASS_NAMES,

                "input_length":
                    int(X.shape[1]),

                "seed":
                    SEED,
            },
            MODEL_PATH,
        )

print()
print("=" * 70)
print("CNN COMMON TRAINING COMPLETE")
print("=" * 70)
print(
    f"Best training accuracy: "
    f"{best_train_accuracy * 100:.2f}%"
)
print("Model:", MODEL_PATH)
