import os
import numpy as np


DATA_PATH = "data/snr_dataset.npz"
RESULT_PATH = "models/symbol_rate_acf_results.npz"


def estimate_symbol_rate(signal, sample_rate=1e6):

    x = np.asarray(signal, dtype=np.complex64)

    if len(x) < 64:
        return 0.0

    # Remove DC
    x = x - np.mean(x)

    # Normalize
    power = np.mean(np.abs(x) ** 2)

    if power <= 1e-12:
        return 0.0

    x = x / np.sqrt(power)

    # Use amplitude-squared to reduce carrier/CFO influence.
    feature = np.abs(x) ** 2

    # Remove mean
    feature = feature - np.mean(feature)

    # Autocorrelation
    corr = np.correlate(
        feature,
        feature,
        mode="full"
    )

    corr = corr[len(feature) - 1:]

    if corr[0] <= 1e-12:
        return 0.0

    corr = corr / corr[0]

    # Expected practical range:
    # 2 to 100 samples/symbol
    min_sps = 2
    max_sps = min(100, len(feature) // 4)

    if max_sps < min_sps:
        return 0.0

    candidates = np.arange(
        min_sps,
        max_sps + 1
    )

    scores = []

    for lag in candidates:

        if lag >= len(corr):
            scores.append(-1.0)
            continue

        # Fundamental periodicity
        score = corr[lag]

        # Also consider harmonics around the candidate
        harmonic_scores = []

        for h in [1, 2, 3]:
            idx = lag * h

            if idx < len(corr):
                harmonic_scores.append(
                    corr[idx]
                )

        if harmonic_scores:
            score = (
                0.7 * corr[lag] +
                0.3 * np.mean(harmonic_scores)
            )

        scores.append(score)

    best_index = int(
        np.argmax(scores)
    )

    best_sps = int(
        candidates[best_index]
    )

    return float(
        sample_rate / best_sps
    )


def main():

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    meta = data["meta"]

    true_rate = np.array([
        float(m["symbol_rate"])
        for m in meta
    ])

    classes = np.array([
        str(m["label"])
        for m in meta
    ])

    snrs = np.array([
        float(m["snr_db"])
        for m in meta
    ])

    print("=" * 70)
    print("AUTOCORRELATION SYMBOL-RATE ESTIMATOR")
    print("=" * 70)

    print(
        f"Signals: {len(X)}"
    )

    estimated_rate = np.array([
        estimate_symbol_rate(signal)
        for signal in X
    ])

    error = (
        estimated_rate -
        true_rate
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"True symbol rate: "
        f"{np.mean(true_rate):.2f} Hz"
    )

    print(
        f"Estimated mean: "
        f"{np.mean(estimated_rate):.2f} Hz"
    )

    print(
        f"MAE  : {mae:.2f} Hz"
    )

    print(
        f"RMSE : {rmse:.2f} Hz"
    )

    print()
    print("=" * 70)
    print("BY MODULATION")
    print("=" * 70)

    for modulation in sorted(
        np.unique(classes)
    ):

        mask = classes == modulation

        e = error[mask]

        print(
            f"{modulation:>6s} | "
            f"MAE: {np.mean(np.abs(e)):9.2f} Hz | "
            f"RMSE: {np.sqrt(np.mean(e**2)):9.2f} Hz | "
            f"N={np.sum(mask)}"
        )

    print()
    print("=" * 70)
    print("BY SNR")
    print("=" * 70)

    for snr in sorted(
        np.unique(snrs)
    ):

        mask = snrs == snr

        e = error[mask]

        print(
            f"SNR {snr:>5.0f} dB | "
            f"MAE: {np.mean(np.abs(e)):9.2f} Hz | "
            f"RMSE: {np.sqrt(np.mean(e**2)):9.2f} Hz | "
            f"N={np.sum(mask)}"
        )

    os.makedirs(
        "models",
        exist_ok=True
    )

    np.savez_compressed(
        RESULT_PATH,
        true_rate=true_rate,
        estimated_rate=estimated_rate,
        mae=mae,
        rmse=rmse
    )

    print()
    print("=" * 70)
    print(
        f"Saved: {RESULT_PATH}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
