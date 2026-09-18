import os
import numpy as np

from amc.signal_generator import CLASSES, generate


OUTPUT = "data/parameter_dataset.npz"

SNR_VALUES = [-10, -5, 0, 5, 10, 15, 20]
SYMBOL_RATES = [50e3, 75e3, 100e3, 125e3, 150e3, 200e3]

N_PER_COMBINATION = 100
LENGTH = 1024
SAMPLE_RATE = 1e6

CFO_MIN = -3000
CFO_MAX = 3000

SEED = 123


def main():

    rng = np.random.default_rng(SEED)

    X = []
    y = []
    meta = []

    total = (
        len(CLASSES)
        * len(SNR_VALUES)
        * len(SYMBOL_RATES)
        * N_PER_COMBINATION
    )

    counter = 0

    print("=" * 70)
    print("VARIABLE-PARAMETER DATASET GENERATION")
    print("=" * 70)

    print(f"Modulations       : {len(CLASSES)}")
    print(f"SNR values        : {SNR_VALUES}")
    print(f"Symbol rates      : {SYMBOL_RATES}")
    print(f"Samples/condition : {N_PER_COMBINATION}")
    print(f"Total signals     : {total}")
    print()

    for label in CLASSES:

        for snr in SNR_VALUES:

            for symbol_rate in SYMBOL_RATES:

                for _ in range(N_PER_COMBINATION):

                    cfo = float(
                        rng.uniform(
                            CFO_MIN,
                            CFO_MAX
                        )
                    )

                    seed = int(
                        rng.integers(
                            0,
                            2**31 - 1
                        )
                    )

                    signal, metadata = generate(
                        label=label,
                        num_samples=LENGTH,
                        snr_db=float(snr),
                        sample_rate=SAMPLE_RATE,
                        symbol_rate=float(symbol_rate),
                        cfo_hz=cfo,
                        seed=seed
                    )

                    X.append(signal)
                    y.append(label)

                    metadata["sample_rate"] = SAMPLE_RATE
                    metadata["symbol_rate"] = float(symbol_rate)
                    metadata["cfo_hz"] = cfo

                    meta.append(metadata)

                    counter += 1

        print(
            f"{label:>6s} complete | "
            f"{counter}/{total}"
        )

    X = np.asarray(
        X,
        dtype=np.complex64
    )

    y = np.asarray(
        y
    )

    meta = np.asarray(
        meta,
        dtype=object
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT,
        X=X,
        y=y,
        meta=meta
    )

    print()
    print("=" * 70)
    print("DATASET COMPLETE")
    print("=" * 70)

    print(f"X shape    : {X.shape}")
    print(f"y shape    : {y.shape}")
    print(f"metadata   : {meta.shape}")
    print(f"Output     : {OUTPUT}")

    print()
    print("Parameter ranges:")
    print(
        f"SNR        : "
        f"{min(SNR_VALUES)} to "
        f"{max(SNR_VALUES)} dB"
    )

    print(
        f"CFO        : "
        f"{CFO_MIN} to "
        f"{CFO_MAX} Hz"
    )

    print(
        f"Symbol rate: "
        f"{min(SYMBOL_RATES)/1000:.0f} to "
        f"{max(SYMBOL_RATES)/1000:.0f} kHz"
    )


if __name__ == "__main__":
    main()
