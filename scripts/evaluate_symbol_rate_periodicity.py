import os
import numpy as np


DATA_PATH = "data/parameter_dataset_v2.npz"
SPLIT_PATH = "data/parameter_v2_common_split.npz"

RESULT_PATH = "models/symbol_rate_periodicity_results.npz"

SAMPLE_RATE = 1e6

CANDIDATE_RATES = np.array([
    50_000.0,
    1e6 / 13,
    100_000.0,
    125_000.0,
    1e6 / 7,
    200_000.0
])


def normalize(v):

    v = np.asarray(v, dtype=np.float64)

    v = v - np.mean(v)

    std = np.std(v)

    if std < 1e-12:
        return np.zeros_like(v)

    return v / std


def periodicity_score(signal, symbol_rate):

    x = np.asarray(
        signal,
        dtype=np.complex64
    )

    x = x - np.mean(x)

    power = np.mean(
        np.abs(x) ** 2
    )

    if power < 1e-12:
        return 0.0

    x = x / np.sqrt(power)

    # ------------------------------------------------------------
    # Expected samples per symbol
    # ------------------------------------------------------------

    sps = SAMPLE_RATE / symbol_rate

    # Examine several nearby integer lags because the actual
    # generated waveform uses integer samples/symbol.
    candidate_lags = np.arange(
        max(2, int(np.floor(sps)) - 1),
        int(np.ceil(sps)) + 2
    )

    candidate_lags = candidate_lags[
        candidate_lags < len(x)
    ]

    if len(candidate_lags) == 0:
        return 0.0

    scores = []

    # ------------------------------------------------------------
    # 1. IQ transition activity
    # ------------------------------------------------------------

    iq_diff = np.abs(
        x[1:] - x[:-1]
    )

    iq_diff = normalize(
        iq_diff
    )

    # ------------------------------------------------------------
    # 2. Envelope transition activity
    # ------------------------------------------------------------

    envelope = np.abs(x)

    env_diff = np.abs(
        np.diff(envelope)
    )

    env_diff = normalize(
        env_diff
    )

    # ------------------------------------------------------------
    # 3. Phase transition activity
    # ------------------------------------------------------------

    phase = np.unwrap(
        np.angle(x)
    )

    phase_diff = np.diff(
        phase
    )

    # Remove large CFO-like mean component
    phase_diff = (
        phase_diff
        - np.median(phase_diff)
    )

    phase_activity = np.abs(
        phase_diff
    )

    phase_activity = normalize(
        phase_activity
    )

    # ------------------------------------------------------------
    # Evaluate periodicity at candidate lags
    # ------------------------------------------------------------

    for lag in candidate_lags:

        lag = int(lag)

        values = []

        for seq in [
            iq_diff,
            env_diff,
            phase_activity
        ]:

            if len(seq) <= lag:
                continue

            a = seq[lag:]
            b = seq[:-lag]

            denom = (
                np.sqrt(
                    np.mean(a ** 2)
                    * np.mean(b ** 2)
                )
                + 1e-12
            )

            corr = np.mean(
                a * b
            ) / denom

            values.append(
                max(0.0, corr)
            )

        if values:
            scores.append(
                np.mean(values)
            )

    if not scores:
        return 0.0

    return float(
        np.max(scores)
    )


def frequency_periodicity_score(signal, symbol_rate):

    x = np.asarray(
        signal,
        dtype=np.complex64
    )

    x = x - np.mean(x)

    # Transition sequence
    diff = np.abs(
        x[1:] - x[:-1]
    )

    diff = diff - np.mean(diff)

    n = len(diff)

    if n < 32:
        return 0.0

    window = np.hanning(n)

    spectrum = np.abs(
        np.fft.rfft(
            diff * window
        )
    ) ** 2

    freqs = np.fft.rfftfreq(
        n,
        d=1.0 / SAMPLE_RATE
    )

    # Symbol rate may appear at Rs and harmonics.
    harmonics = [
        1,
        2,
        3,
        4
    ]

    score = 0.0

    for harmonic in harmonics:

        target = symbol_rate * harmonic

        if target >= SAMPLE_RATE / 2:
            continue

        bandwidth = max(
            1500.0,
            symbol_rate * 0.08
        )

        mask = (
            np.abs(freqs - target)
            <= bandwidth
        )

        if np.any(mask):

            local = np.max(
                spectrum[mask]
            )

            background = np.median(
                spectrum[
                    (freqs > 1000)
                    & (
                        freqs
                        < SAMPLE_RATE / 2
                    )
                ]
            ) + 1e-12

            score += (
                local / background
            ) / harmonic

    return float(
        np.log1p(score)
    )


def estimate_symbol_rate(signal):

    scores = []

    for rate in CANDIDATE_RATES:

        s1 = periodicity_score(
            signal,
            rate
        )

        s2 = frequency_periodicity_score(
            signal,
            rate
        )

        # Equal contribution.
        score = (
            0.5 * s1
            + 0.5 * s2
        )

        scores.append(score)

    scores = np.asarray(
        scores
    )

    best = np.argmax(
        scores
    )

    return (
        CANDIDATE_RATES[best],
        scores
    )


def main():

    print("=" * 70)
    print(
        "SYMBOL RATE ESTIMATION - "
        "SIGNAL PROCESSING"
    )
    print("=" * 70)

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    meta = data["meta"]

    true_rates = np.asarray([
        float(m["symbol_rate"])
        for m in meta
    ])

    split = np.load(
        SPLIT_PATH
    )

    test_idx = split["test_idx"]

    actual = true_rates[
        test_idx
    ]

    predicted = []
    confidence_scores = []

    print()
    print(
        f"Test samples: {len(test_idx)}"
    )

    print()
    print(
        "Estimating symbol rates..."
    )

    for count, idx in enumerate(
        test_idx,
        start=1
    ):

        if count % 1000 == 0:
            print(
                f"{count}/{len(test_idx)}"
            )

        rate, scores = estimate_symbol_rate(
            X[idx]
        )

        predicted.append(
            rate
        )

        confidence_scores.append(
            scores
        )

    predicted = np.asarray(
        predicted
    )

    confidence_scores = np.asarray(
        confidence_scores
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    error = (
        predicted
        - actual
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(
        error ** 2
    )

    ss_tot = np.sum(
        (
            actual
            - np.mean(actual)
        ) ** 2
    )

    r2 = (
        1.0
        - ss_res / ss_tot
    )

    correlation = np.corrcoef(
        actual,
        predicted
    )[0, 1]

    accuracy = np.mean(
        np.isclose(
            actual,
            predicted
        )
    )

    print()
    print("=" * 70)
    print(
        "SYMBOL RATE RESULTS"
    )
    print("=" * 70)

    print(
        f"MAE         : {mae:.2f} Hz"
    )

    print(
        f"RMSE        : {rmse:.2f} Hz"
    )

    print(
        f"R2          : {r2:.4f}"
    )

    print(
        f"Correlation : {correlation:.4f}"
    )

    print(
        f"Exact-class accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # Error by symbol rate
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ERROR BY SYMBOL RATE"
    )
    print("=" * 70)

    rate_results = []

    for rate in CANDIDATE_RATES:

        mask = np.isclose(
            actual,
            rate
        )

        if not np.any(mask):
            continue

        err = (
            predicted[mask]
            - actual[mask]
        )

        r_mae = np.mean(
            np.abs(err)
        )

        r_rmse = np.sqrt(
            np.mean(err ** 2)
        )

        rate_accuracy = np.mean(
            np.isclose(
                predicted[mask],
                rate
            )
        )

        count = np.sum(mask)

        rate_results.append([
            rate,
            r_mae,
            r_rmse,
            rate_accuracy,
            count
        ])

        print(
            f"{rate:>12,.2f} Hz | "
            f"MAE: {r_mae:>9.2f} | "
            f"RMSE: {r_rmse:>9.2f} | "
            f"Accuracy: {rate_accuracy * 100:>6.2f}% | "
            f"N={count}"
        )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    confusion = np.zeros(
        (
            len(CANDIDATE_RATES),
            len(CANDIDATE_RATES)
        ),
        dtype=np.int32
    )

    for a, p in zip(
        actual,
        predicted
    ):

        true_class = np.argmin(
            np.abs(
                CANDIDATE_RATES
                - a
            )
        )

        pred_class = np.argmin(
            np.abs(
                CANDIDATE_RATES
                - p
            )
        )

        confusion[
            true_class,
            pred_class
        ] += 1

    print()
    print(
        "Confusion matrix:"
    )

    print(confusion)

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    np.savez_compressed(
        RESULT_PATH,
        actual_symbol_rate=actual,
        predicted_symbol_rate=predicted,
        confidence_scores=confidence_scores,
        candidate_rates=CANDIDATE_RATES,
        mae=mae,
        rmse=rmse,
        r2=r2,
        correlation=correlation,
        accuracy=accuracy,
        rate_results=np.asarray(
            rate_results
        ),
        confusion_matrix=confusion,
        test_idx=test_idx
    )

    print()
    print("=" * 70)
    print("FILE SAVED")
    print("=" * 70)

    print(
        RESULT_PATH
    )

    print()
    print("=" * 70)
    print(
        "SIGNAL-PROCESSING "
        "SYMBOL RATE ESTIMATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
