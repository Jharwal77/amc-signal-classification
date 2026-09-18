import os
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "data/parameter_dataset_v2.npz"
OUT_PATH = "data/parameter_v2_common_split.npz"

data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]
y = data["y"]
meta = data["meta"]

labels = np.asarray([
    m["label"]
    for m in meta
])

indices = np.arange(len(X))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

os.makedirs("data", exist_ok=True)

np.savez_compressed(
    OUT_PATH,
    train_idx=train_idx,
    test_idx=test_idx
)

print("=" * 70)
print("PARAMETER DATASET V2 - COMMON SPLIT")
print("=" * 70)

print(f"Total : {len(indices)}")
print(f"Train : {len(train_idx)}")
print(f"Test  : {len(test_idx)}")

print()
print("Split saved:")
print(OUT_PATH)

print()
print("=" * 70)
print("COMMON SPLIT COMPLETE")
print("=" * 70)
