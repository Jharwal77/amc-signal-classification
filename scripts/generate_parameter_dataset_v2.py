import os
import numpy as np

from amc.signal_generator import generate, CLASSES


OUT_PATH = "data/parameter_dataset_v2.npz"

SNR_VALUES = [-10, -5, 0, 5, 10, 15, 20]

# Integer samples/symbol gives physically represented symbol rates.
SPS_VALUES = [20, 13, 10, 8, 7, 5]

SAMPLE_RATE = 1e6
LENGTH = 1024
N_PER_CONDITION = 100

rng = np.random.default_rng(42)

X = []
y = []
meta = []

print("=" * 70)
print("GENERATING VARIABLE-PARAMETER DATASET V2")
print("=" * 70)

for label in CLASSES:

    for snr_db in SNR_VALUES:

        for sps in SPS_VALUES:

            actual_symbol_rate = SAMPLE_RATE / sps

            for _ in range(N_PER_CONDITION):

                cfo = float(
                    rng.uniform(-3000, 3000)
                )

                # generate() internally derives SPS from symbol_rate.
                x, _ = generate(
                    label=label,
                    num_samples=LENGTH,
                    snr_db=float(snr_db),
                    sample_rate=SAMPLE_RATE,
                    symbol_rate=actual_symbol_rate,
                    cfo_hz=cfo,
                    seed=int(rng.integers(1_000_000_000))
                )

                X.append(x)
                y.append(label)

                meta.append({
                    "label": label,
                    "snr_db": float(snr_db),
                    "sample_rate": SAMPLE_RATE,
                    "symbol_rate": actual_symbol_rate,
                    "samples_per_symbol": sps,
                    "cfo_hz": cfo
                })

X = np.asarray(X, dtype=np.complex64)
y = np.asarray(y)

meta = np.asarray(meta, dtype=object)

os.makedirs("data", exist_ok=True)

np.savez_compressed(
    OUT_PATH,
    X=X,
    y=y,
    meta=meta
)

print()
print("Dataset saved:")
print(OUT_PATH)

print()
print("Shape:")
print("X:", X.shape)
print("y:", y.shape)

print()
print("Actual symbol rates:")

for sps in SPS_VALUES:
    print(
        f"{sps:>2} samples/symbol -> "
        f"{SAMPLE_RATE / sps:,.2f} Hz"
    )

print()
print("=" * 70)
print("DATASET V2 COMPLETE")
print("=" * 70)
