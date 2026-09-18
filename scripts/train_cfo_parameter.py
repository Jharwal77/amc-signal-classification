import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from amc.preprocessing import to_features


DATA_PATH = "data/parameter_dataset.npz"
SPLIT_PATH = "data/parameter_common_split.npz"

MODEL_PATH = "models/cfo_parameter_rf.joblib"
RESULT_PATH = "models/cfo_parameter_rf_results.npz"


def extract_features(signal, sample_rate=1e6):

    x = np.asarray(
        signal,
        dtype=np.complex64
    )

    x = x - np.mean(x)

    power = np.mean(
        np.abs(x) ** 2
    )

    if power <= 1e-12:
        return np.zeros(30, dtype=np.float32)

    x = x / np.sqrt(power)

    # Phase-difference features
    phase_diff = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    inst_freq = (
        phase_diff
        * sample_rate
        / (2.0 * np.pi)
    )

    # Robust frequency statistics
    features = [
        np.mean(inst_freq),
        np.median(inst_freq),
        np.std(inst_freq),
        np.percentile(inst_freq, 5),
        np.percentile(inst_freq, 10),
        np.percentile(inst_freq, 25),
        np.percentile(inst_freq, 75),
        np.percentile(inst_freq, 90),
        np.percentile(inst_freq, 95),
    ]

    # Multiple-lag phase increments
    for lag in [2, 4, 8, 16, 32]:

        if len(x) <= lag:
            features.append(0.0)
            continue

        pd = np.angle(
            x[lag:] * np.conj(x[:-lag])
        )

        freq = (
            pd
            * sample_rate
            / (2.0 * np.pi * lag)
        )

        features.extend([
            np.mean(freq),
            np.median(freq),
            np.std(freq)
        ])

    # FFT spectral features
    n = min(
        len(x),
        1024
    )

    window = np.hanning(n)

    spectrum = np.fft.fftshift(
        np.fft.fft(
            x[:n] * window
        )
    )

    power_spectrum = (
        np.abs(spectrum) ** 2
    )

    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1.0 / sample_rate
        )
    )

    total = (
        np.sum(power_spectrum)
        + 1e-12
    )

    peak = np.argmax(
        power_spectrum
    )

    spectral_centroid = (
        np.sum(
            freqs * power_spectrum
        )
        / total
    )

    spectral_spread = np.sqrt(
        np.sum(
            (
                freqs
                - spectral_centroid
            ) ** 2
            * power_spectrum
        )
        / total
    )

    features.extend([
        freqs[peak],
        spectral_centroid,
        spectral_spread,
        np.max(power_spectrum) / total,
        np.percentile(
            power_spectrum,
            90
        )
    ])

    # Pad/truncate to fixed length
    features = np.asarray(
        features,
        dtype=np.float32
    )

    if len(features) < 30:
        features = np.pad(
            features,
            (0, 30 - len(features))
        )

    return features[:30]


def main():

    print("=" * 70)
    print("CFO ESTIMATION - VARIABLE PARAMETER DATASET")
    print("=" * 70)

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    meta = data["meta"]

    true_cfo = np.array([
        float(m["cfo_hz"])
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

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    print()
    print("Extracting CFO features...")

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

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

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
        true_cfo[train_idx]
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    predicted = model.predict(
        features[test_idx]
    )

    actual = true_cfo[test_idx]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

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
    print("CFO REGRESSION RESULTS")
    print("=" * 70)

    print(
        f"MAE         : {mae:.4f} Hz"
    )

    print(
        f"RMSE        : {rmse:.4f} Hz"
    )

    print(
        f"R2          : {r2:.4f}"
    )

    print(
        f"Correlation : {correlation:.4f}"
    )

    # ---------------------------------------------------------
    # Error by CFO range
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR BY CFO RANGE")
    print("=" * 70)

    ranges = [
        (-3000, -2000),
        (-2000, -1000),
        (-1000, 0),
        (0, 1000),
        (1000, 2000),
        (2000, 3000)
    ]

    range_results = []

    for low, high in ranges:

        mask = (
            (actual >= low)
            & (actual < high)
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

        range_results.append([
            low,
            high,
            r_mae,
            r_rmse,
            bias,
            count
        ])

        print(
            f"{low:>5} to {high:>5} Hz | "
            f"MAE: {r_mae:8.2f} | "
            f"RMSE: {r_rmse:8.2f} | "
            f"Bias: {bias:8.2f} | "
            f"N={count}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

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
        actual_cfo=actual,
        predicted_cfo=predicted,
        mae=mae,
        rmse=rmse,
        r2=r2,
        correlation=correlation,
        range_results=np.asarray(
            range_results
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
    print("CFO ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
