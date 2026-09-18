import os
import numpy as np
from sklearn.model_selection import train_test_split


DATA_PATH = "data/parameter_dataset.npz"
OUTPUT_PATH = "data/parameter_common_split.npz"


def main():

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    y = data["y"]
    meta = data["meta"]

    indices = np.arange(len(y))

    # Stratify by modulation class
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    np.savez(
        OUTPUT_PATH,
        train_idx=train_idx,
        test_idx=test_idx
    )

    print("=" * 70)
    print("PARAMETER DATASET COMMON SPLIT")
    print("=" * 70)

    print(f"Total : {len(indices)}")
    print(f"Train : {len(train_idx)}")
    print(f"Test  : {len(test_idx)}")

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
