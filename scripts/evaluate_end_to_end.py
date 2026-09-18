import os
import numpy as np
import pandas as pd

from amc.inference import SignalAnalyzer
from amc.signal_generator import generate, CLASSES


OUTPUT = "models/end_to_end_results.csv"

SNR_VALUES = [-10, -5, 0, 5, 10, 15, 20]

SYMBOL_RATES = [
    50_000.0,
    1e6 / 13,
    100_000.0,
    125_000.0,
    1e6 / 7,
    200_000.0
]

N_PER_CONDITION = 5

SAMPLE_RATE = 1e6
LENGTH = 1024

rng = np.random.default_rng(2026)


def main():

    print("=" * 70)
    print("END-TO-END SYSTEM EVALUATION")
    print("=" * 70)

    analyzer = SignalAnalyzer()

    rows = []

    total = (
        len(CLASSES)
        * len(SNR_VALUES)
        * len(SYMBOL_RATES)
        * N_PER_CONDITION
    )

    count = 0

    print()
    print(
        f"Total evaluation signals: {total}"
    )

    for modulation in CLASSES:

        for snr in SNR_VALUES:

            for symbol_rate in SYMBOL_RATES:

                for _ in range(
                    N_PER_CONDITION
                ):

                    count += 1

                    cfo = float(
                        rng.uniform(
                            -3000,
                            3000
                        )
                    )

                    signal, metadata = generate(
                        label=modulation,
                        num_samples=LENGTH,
                        snr_db=float(snr),
                        sample_rate=SAMPLE_RATE,
                        symbol_rate=symbol_rate,
                        cfo_hz=cfo,
                        seed=int(
                            rng.integers(
                                1_000_000_000
                            )
                        )
                    )

                    result = analyzer.analyze(
                        signal,
                        sample_rate=SAMPLE_RATE
                    )

                    rows.append({
                        "true_modulation":
                            modulation,

                        "pred_modulation":
                            result[
                                "modulation"
                            ],

                        "modulation_confidence":
                            result[
                                "modulation_confidence"
                            ],

                        "true_snr_db":
                            float(snr),

                        "pred_snr_db":
                            result[
                                "snr_db"
                            ],

                        "true_cfo_hz":
                            cfo,

                        "pred_cfo_hz":
                            result[
                                "cfo_hz"
                            ],

                        "true_symbol_rate_hz":
                            symbol_rate,

                        "pred_symbol_rate_hz":
                            result[
                                "symbol_rate_hz"
                            ]
                    })

                    if count % 100 == 0:

                        print(
                            f"{count}/{total}"
                        )

    df = pd.DataFrame(rows)

    os.makedirs(
        "models",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    # ------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------

    modulation_accuracy = np.mean(
        df["true_modulation"]
        == df["pred_modulation"]
    )

    snr_mae = np.mean(
        np.abs(
            df["true_snr_db"]
            - df["pred_snr_db"]
        )
    )

    snr_rmse = np.sqrt(
        np.mean(
            (
                df["true_snr_db"]
                - df["pred_snr_db"]
            ) ** 2
        )
    )

    cfo_mae = np.mean(
        np.abs(
            df["true_cfo_hz"]
            - df["pred_cfo_hz"]
        )
    )

    cfo_rmse = np.sqrt(
        np.mean(
            (
                df["true_cfo_hz"]
                - df["pred_cfo_hz"]
            ) ** 2
        )
    )

    symbol_mae = np.mean(
        np.abs(
            df["true_symbol_rate_hz"]
            - df["pred_symbol_rate_hz"]
        )
    )

    symbol_accuracy = np.mean(
        df["true_symbol_rate_hz"]
        == df["pred_symbol_rate_hz"]
    )

    print()
    print("=" * 70)
    print("END-TO-END RESULTS")
    print("=" * 70)

    print(
        f"Modulation Accuracy : "
        f"{modulation_accuracy * 100:.2f}%"
    )

    print(
        f"SNR MAE             : "
        f"{snr_mae:.3f} dB"
    )

    print(
        f"SNR RMSE            : "
        f"{snr_rmse:.3f} dB"
    )

    print(
        f"CFO MAE             : "
        f"{cfo_mae:.2f} Hz"
    )

    print(
        f"CFO RMSE            : "
        f"{cfo_rmse:.2f} Hz"
    )

    print(
        f"Symbol Rate MAE     : "
        f"{symbol_mae:.2f} Hz"
    )

    print(
        f"Symbol Rate Accuracy: "
        f"{symbol_accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Modulation confusion
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODULATION ACCURACY BY CLASS")
    print("=" * 70)

    for modulation in CLASSES:

        subset = df[
            df["true_modulation"]
            == modulation
        ]

        accuracy = np.mean(
            subset["pred_modulation"]
            == subset["true_modulation"]
        )

        print(
            f"{modulation:>6}: "
            f"{accuracy * 100:.2f}%"
        )

    # ------------------------------------------------------------
    # SNR performance
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SNR ERROR BY TRUE SNR")
    print("=" * 70)

    for snr in SNR_VALUES:

        subset = df[
            df["true_snr_db"]
            == snr
        ]

        mae = np.mean(
            np.abs(
                subset["true_snr_db"]
                - subset["pred_snr_db"]
            )
        )

        print(
            f"{snr:>4} dB: "
            f"MAE = {mae:.3f} dB"
        )

    # ------------------------------------------------------------
    # CFO performance
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CFO ERROR BY RANGE")
    print("=" * 70)

    bins = [
        (-3000, -2000),
        (-2000, -1000),
        (-1000, 0),
        (0, 1000),
        (1000, 2000),
        (2000, 3000)
    ]

    for low, high in bins:

        subset = df[
            (df["true_cfo_hz"] >= low)
            & (df["true_cfo_hz"] < high)
        ]

        if len(subset) == 0:
            continue

        mae = np.mean(
            np.abs(
                subset["true_cfo_hz"]
                - subset["pred_cfo_hz"]
            )
        )

        print(
            f"{low:>5} to {high:>5} Hz: "
            f"MAE = {mae:.2f} Hz"
        )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(OUTPUT)

    print()
    print("=" * 70)
    print("END-TO-END EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
