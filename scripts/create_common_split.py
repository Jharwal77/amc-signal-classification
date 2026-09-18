import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "data/snr_dataset.npz"
OUTPUT_PATH = "data/common_split.npz"

SEED = 42

data = np.load(DATA_PATH, allow_pickle=True)

y = data["y"].astype(str)

indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=y,
)

np.savez(
    OUTPUT_PATH,
    train_idx=train_idx,
    test_idx=test_idx,
)

print("=" * 60)
print("COMMON BENCHMARK SPLIT")
print("=" * 60)
print("Total:", len(indices))
print("Train:", len(train_idx))
print("Test:", len(test_idx))
print("Saved:", OUTPUT_PATH)
