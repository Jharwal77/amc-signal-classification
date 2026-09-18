import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/parameter_dataset_v2.npz"
SPLIT_PATH = "data/parameter_v2_common_split.npz"

MODEL_PATH = "models/symbol_rate_parameter_rf.joblib"
RESULT_PATH = "models/symbol_rate_parameter_rf_results.npz"


def extract_features(signal, sample_rate=1e6):

    x = np.asarray(signal, dtype=np.complex64)

    x = x - np.mean(x)

    power = np.mean(np.abs(x) ** 2)

    if power <= 1e-12:
        return np.zeros(40, dtype=np.float32)

    x = x / np.sqrt(power)

    features = []

    # ------------------------------------------------------------
    # Amplitude-envelope features
    # ------------------------------------------------------------

    envelope = np.abs(x)

    features.extend([
        np.mean(envelope),
        np.std(envelope),
        np.median(envelope),
        np.percentile(envelope, 10),
        np.percentile(envelope, 25),
        np.percentile(envelope, 75),
        np.percentile(envelope, 90),
    ])

    # ------------------------------------------------------------
    # Autocorrelation features
    # ------------------------------------------------------------

    real_x = np.real(x)

    real_x = real_x - np.mean(real_x)

    acf = np.correlate(
        real_x,
        real_x,
        mode="full"
    )

    acf = acf[len(real_x)-1:]

    if acf[0] > 1e-12:
        acf = acf / acf[0]

    search_end = min(
        len(acf),
        512
    )

    # Ignore zero lag
    search = acf[1:search_end]

    if len(search) > 0:

        peak_lag = np.argmax(
            search
        ) + 1

        peak_value = search[
            peak_lag - 1
        ]

    else:

        peak_lag = 0
        peak_value = 0

    features.extend([
        float(peak_lag),
        float(peak_value)
    ])

    # ------------------------------------------------------------
    # Complex-signal autocorrelation
    # ------------------------------------------------------------

    max_lag = min(
        256,
        len(x) - 1
    )

    complex_acf = []

    for lag in range(
        1,
        max_lag + 1
    ):

        value = np.mean(
            x[lag:] * np.conj(x[:-lag])
        )

        complex_acf.append(
            np.abs(value)
        )

    complex_acf = np.asarray(
        complex_acf
    )

    if len(complex_acf) > 0:

        # Top correlation peaks
        top_indices = np.argsort(
            complex_acf
        )[-5:]

        top_indices = np.sort(
            top_indices
        )

        for idx in top_indices:
            features.append(
                float(idx + 1)
            )

            features.append(
                float(
                    complex_acf[idx]
                )
            )

    # ------------------------------------------------------------
    # Envelope spectrum
    # ------------------------------------------------------------

    env = envelope - np.mean(
        envelope
    )

    n = min(
        len(env),
        1024
    )

    window = np.hanning(n)

    spectrum = np.fft.rfft(
        env[:n] * window
    )

    power_spectrum = (
        np.abs(spectrum) ** 2
    )

    freqs = np.fft.rfftfreq(
        n,
        d=1.0 / sample_rate
    )

    if len(power_spectrum) > 1:

        power_spectrum[0] = 0

        peak = np.argmax(
            power_spectrum
        )

        peak_freq = freqs[
            peak
        ]

        total_power = (
            np.sum(power_spectrum)
            + 1e-12
        )

        centroid = (
            np.sum(
                freqs
                * power_spectrum
            )
            / total_power
        )

        spread = np.sqrt(
            np.sum(
                (
                    freqs
                    - centroid
                ) ** 2
                * power_spectrum
            )
            / total_power
        )

        features.extend([
            float(peak_freq),
            float(centroid),
            float(spread),
            float(
                power_spectrum[peak]
                / total_power
            )
        ])

    else:

        features.extend([
            0.0,
            0.0,
            0.0,
            0.0
        ])

    # ------------------------------------------------------------
    # Magnitude-squared spectrum
    # ------------------------------------------------------------

    magnitude = np.abs(x)

    magnitude = (
        magnitude
        - np.mean(magnitude)
    )

    spec2 = np.fft.rfft(
        magnitude[:n] * window
    )

    power2 = (
        np.abs(spec2) ** 2
    )

    if len(power2) > 1:

        power2[0] = 0

        peak2 = np.argmax(
            power2
        )

        total2 = (
            np.sum(power2)
            + 1e-12
        )

        features.extend([
            float(freqs[peak2]),
            float(
                power2[peak2]
                / total2
            ),
            float(
                np.percentile(
                    power2,
                    90
                )
                / total2
            )
        ])

    else:

        features.extend([
            0.0,
            0.0,
            0.0
        ])

    # ------------------------------------------------------------
    # Higher-order envelope statistics
    # ------------------------------------------------------------

    centered = (
        envelope
        - np.mean(envelope)
    )

    std = (
        np.std(centered)
        + 1e-12
    )

    standardized = (
        centered / std
    )

    features.extend([
        float(
            np.mean(
                standardized ** 3
            )
        ),
        float(
            np.mean(
                standardized ** 4
            )
        )
    ])

    # ------------------------------------------------------------
    # Fixed-length feature vector
    # ------------------------------------------------------------

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if len(features) < 40:

        features = np.pad(
            features,
            (0, 40 - len(features))
        )

    return features[:40]


def main():

    print("=" * 70)
    print(
        "SYMBOL RATE ESTIMATION - "
        "VARIABLE PARAMETER DATASET V2"
    )
    print("=" * 70)

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    meta = data["meta"]

    true_symbol_rate = np.asarray([
        float(m["symbol_rate"])
        for m in meta
    ])

    split = np.load(
        SPLIT_PATH
    )

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    print(
        f"Train: {len(train_idx)}"
    )

    print(
        f"Test : {len(test_idx)}"
    )

    print()
    print(
        "Symbol-rate targets:"
    )

    unique_rates = np.unique(
        true_symbol_rate
    )

    for rate in unique_rates:
        print(
            f"{rate:,.2f} Hz"
        )

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------

    print()
    print(
        "Extracting symbol-rate features..."
    )

    features = []

    for i, signal in enumerate(X):

        if (i + 1) % 2000 == 0:

            print(
                f"{i + 1}/{len(X)}"
            )

        features.append(
            extract_features(signal)
        )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    print()
    print(
        "Training Random Forest Regressor..."
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    model.fit(
        features[train_idx],
        true_symbol_rate[train_idx]
    )

    # ------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------

    predicted = model.predict(
        features[test_idx]
    )

    actual = true_symbol_rate[
        test_idx
    ]

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    correlation = np.corrcoef(
        actual,
        predicted
    )[0, 1]

    print()
    print("=" * 70)
    print("SYMBOL RATE REGRESSION RESULTS")
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

    # ------------------------------------------------------------
    # Error by symbol rate
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR BY SYMBOL RATE")
    print("=" * 70)

    rate_results = []

    for rate in unique_rates:

        mask = np.isclose(
            actual,
            rate
        )

        if not np.any(mask):
            continue

        error = (
            predicted[mask]
            - actual[mask]
        )

        r_mae = np.mean(
            np.abs(error)
        )

        r_rmse = np.sqrt(
            np.mean(error ** 2)
        )

        bias = np.mean(
            error
        )

        count = np.sum(mask)

        rate_results.append([
            rate,
            r_mae,
            r_rmse,
            bias,
            count
        ])

        print(
            f"{rate:>12,.2f} Hz | "
            f"MAE: {r_mae:>9.2f} | "
            f"RMSE: {r_rmse:>9.2f} | "
            f"Bias: {bias:>9.2f} | "
            f"N={count}"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    np.savez_compressed(
        RESULT_PATH,
        actual_symbol_rate=actual,
        predicted_symbol_rate=predicted,
        mae=mae,
        rmse=rmse,
        r2=r2,
        correlation=correlation,
        rate_results=np.asarray(
            rate_results
        ),
        test_idx=test_idx
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        MODEL_PATH
    )

    print(
        RESULT_PATH
    )

    print()
    print("=" * 70)
    print(
        "SYMBOL RATE ESTIMATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
