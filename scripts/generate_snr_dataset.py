import numpy as np
from pathlib import Path
from amc.signal_generator import CLASSES, generate


def make_snr_dataset(
    path,
    n_per_class=250,
    length=1024,
    snr_values=(-10, -5, 0, 5, 10, 15, 20),
    cfo_max_hz=3000,
    sample_rate=1e6,
    symbol_rate=1e5,
    seed=12345,
):
    """
    Generate a balanced AMC dataset with explicitly controlled SNR.

    For every SNR value, each modulation class receives
    exactly n_per_class signals.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    X = []
    y = []
    meta = []

    rng = np.random.default_rng(seed)

    for snr_db in snr_values:

        print(f"Generating SNR = {snr_db} dB")

        for label in CLASSES:

            for _ in range(n_per_class):

                cfo = float(
                    rng.uniform(
                        -cfo_max_hz,
                        cfo_max_hz,
                    )
                )

                signal, metadata = generate(
                    label=label,
                    num_samples=length,
                    snr_db=float(snr_db),
                    sample_rate=sample_rate,
                    symbol_rate=symbol_rate,
                    cfo_hz=cfo,
                    seed=int(rng.integers(1_000_000_000)),
                )

                X.append(signal)
                y.append(label)

                metadata["dataset_snr_db"] = float(snr_db)
                metadata["cfo_hz"] = cfo

                meta.append(metadata)

    X = np.asarray(X, dtype=np.complex64)
    y = np.asarray(y)
    meta = np.asarray(meta, dtype=object)

    np.savez_compressed(
        path,
        X=X,
        y=y,
        meta=meta,
    )

    print()
    print("=" * 60)
    print("CONTROLLED DATASET CREATED")
    print("=" * 60)
    print("File:", path)
    print("Samples:", len(X))
    print("Signal length:", X.shape[1])
    print("Classes:", len(CLASSES))
    print("SNR values:", list(snr_values))
    print("Samples per class per SNR:", n_per_class)
    print("Total expected samples:",
          len(CLASSES) * len(snr_values) * n_per_class)
    print("=" * 60)


if __name__ == "__main__":
    make_snr_dataset(
        path="data/snr_dataset.npz",
        n_per_class=250,
        length=1024,
    )
