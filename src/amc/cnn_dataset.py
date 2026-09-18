import numpy as np
import torch
from torch.utils.data import Dataset


class AMCDataset(Dataset):
    """
    Dataset for Automatic Modulation Classification.

    NPZ format:
        X: complex64 array of shape (N, signal_length)
        y: string modulation labels of shape (N,)

    CNN output:
        X -> float32 tensor of shape (2, signal_length)
        y -> integer class index
    """

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

    CLASS_TO_INDEX = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.X = data["X"]
        self.labels = data["y"].astype(str)

        if self.X.ndim != 2:
            raise ValueError(
                f"Expected X to have shape (N, L), got {self.X.shape}"
            )

        unknown = set(self.labels) - set(self.CLASS_TO_INDEX)

        if unknown:
            raise ValueError(
                f"Unknown modulation labels: {sorted(unknown)}"
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        signal = self.X[index]

        # Convert complex signal into two channels: I and Q
        iq = np.stack(
            [signal.real, signal.imag],
            axis=0
        ).astype(np.float32)

        # Per-sample normalization
        scale = np.sqrt(np.mean(iq ** 2))

        if scale > 0:
            iq = iq / scale

        label = self.CLASS_TO_INDEX[self.labels[index]]

        return (
            torch.from_numpy(iq),
            torch.tensor(label, dtype=torch.long),
        )
