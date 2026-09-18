import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


DATA_PATH = "data/snr_dataset.npz"
MODEL_PATH = "models/cfo_rf.joblib"
RESULT_PATH = "models/cfo_rf_results.npz"


def extract_cfo_features(signal, sample_rate=1e6):
    """
    Extract frequency-related features from a complex baseband signal.
    """

    x = np.asarray(signal, dtype=np.complex64)

    # Remove DC
    x = x - np.mean(x)

    # Phase
    phase = np.unwrap(np.angle(x))

    # Phase differences
    dphase = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    # Instantaneous frequency
    inst_freq = (
        dphase * sample_rate / (2.0 * np.pi)
    )

    # Robust statistics
    features = [
        np.mean(inst_freq),
        np.median(inst_freq),
        np.std(inst_freq),
        np.percentile(inst_freq, 10),
        np.percentile(inst_freq, 25),
        np.percentile(inst_freq, 75),
        np.percentile(inst_freq, 90),
    ]

    # FFT-based frequency information
    n = min(len(x), 1024)

    window = np.hanning(n)

    spectrum = np.fft.fftshift(
        np.fft.fft(
            x[:n] * window
        )
    )

    power = np.abs(spectrum) ** 2

    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1.0 / sample_rate
        )
    )

    peak_index = np.argmax(power)

    peak_frequency = freqs[peak_index]

    total_power = np.sum(power) + 1e-12

    spectral_centroid = (
        np.sum(freqs * power) /
        total_power
    )

    spectral_spread = np.sqrt(
        np.sum(
            ((freqs - spectral_centroid) ** 2)
            * power
        ) /
        total_power
    )

    features.extend([
        peak_frequency,
        spectral_centroid,
        spectral_spread
    ])

    return np.asarray(
        features,
        dtype=np.float32
    )


def main():

    print("=" * 70)
    print("CARRIER FREQUENCY OFFSET ESTIMATION")
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

    print(f"Signals: {len(X)}")

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    print()
    print("Extracting CFO features...")

    features = []

    for i, signal in enumerate(X):

        if (i + 1) % 1000 == 0:
            print(
                f"Extracting features: "
                f"{i + 1}/{len(X)}"
            )

        features.append(
            extract_cfo_features(signal)
        )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42
    )

    print()
    print(
        f"Training samples: {len(train_idx)}"
    )

    print(
        f"Testing samples : {len(test_idx)}"
    )

    # ---------------------------------------------------------
    # Train RF regressor
    # ---------------------------------------------------------

    print()
    print(
        "Training Random Forest Regressor..."
    )

    model = RandomForestRegressor(
        n_estimators=300,
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

    predicted_cfo = model.predict(
        features[test_idx]
    )

    actual_cfo = true_cfo[test_idx]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        actual_cfo,
        predicted_cfo
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_cfo,
            predicted_cfo
        )
    )

    r2 = r2_score(
        actual_cfo,
        predicted_cfo
    )

    print()
    print("=" * 70)
    print("CFO REGRESSION RESULTS")
    print("=" * 70)

    print(
        f"MAE  : {mae:.4f} Hz"
    )

    print(
        f"RMSE : {rmse:.4f} Hz"
    )

    print(
        f"R2   : {r2:.4f}"
    )

    # ---------------------------------------------------------
    # CFO range analysis
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("CFO RANGE")
    print("=" * 70)

    print(
        f"True CFO range: "
        f"{actual_cfo.min():.2f} Hz "
        f"to {actual_cfo.max():.2f} Hz"
    )

    print(
        f"Estimated CFO range: "
        f"{predicted_cfo.min():.2f} Hz "
        f"to {predicted_cfo.max():.2f} Hz"
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
        actual_cfo=actual_cfo,
        predicted_cfo=predicted_cfo,
        mae=mae,
        rmse=rmse,
        r2=r2,
        test_idx=test_idx
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(MODEL_PATH)
    print(RESULT_PATH)

    print()
    print("=" * 70)
    print("CFO ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
