import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_PATH = "data/parameter_dataset_v2.npz"
SPLIT_PATH = "data/parameter_v2_common_split.npz"

MODEL_PATH = "models/symbol_rate_classifier.joblib"
RESULT_PATH = "models/symbol_rate_classifier_results.npz"


RATES = np.array([
    50_000.0,
    1e6 / 13,
    100_000.0,
    125_000.0,
    1e6 / 7,
    200_000.0
])


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
        return np.zeros(
            40,
            dtype=np.float32
        )

    x = x / np.sqrt(power)

    features = []

    envelope = np.abs(x)

    # Envelope statistics
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
    # Autocorrelation
    # ------------------------------------------------------------

    real_x = np.real(x)
    real_x -= np.mean(real_x)

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

    search = acf[1:search_end]

    if len(search):

        peak_lag = (
            np.argmax(search) + 1
        )

        peak_value = search[
            peak_lag - 1
        ]

        # Top ACF peaks
        top = np.argsort(
            search
        )[-8:]

        top = np.sort(top)

        for idx in top:
            features.append(
                float(idx + 1)
            )
            features.append(
                float(search[idx])
            )

    else:

        peak_lag = 0
        peak_value = 0

    features.extend([
        float(peak_lag),
        float(peak_value)
    ])

    # ------------------------------------------------------------
    # Complex autocorrelation
    # ------------------------------------------------------------

    for lag in [
        1, 2, 3, 4, 5,
        6, 7, 8, 10,
        12, 16, 20,
        24, 32, 40,
        50, 64, 80,
        100, 128
    ]:

        if lag >= len(x):
            features.append(0.0)
            continue

        corr = np.mean(
            x[lag:]
            * np.conj(x[:-lag])
        )

        features.append(
            float(np.abs(corr))
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

        total = (
            np.sum(power_spectrum)
            + 1e-12
        )

        centroid = (
            np.sum(
                freqs
                * power_spectrum
            )
            / total
        )

        spread = np.sqrt(
            np.sum(
                (
                    freqs
                    - centroid
                ) ** 2
                * power_spectrum
            )
            / total
        )

        features.extend([
            float(freqs[peak]),
            float(centroid),
            float(spread),
            float(
                power_spectrum[peak]
                / total
            )
        ])

    else:

        features.extend([
            0.0, 0.0, 0.0, 0.0
        ])

    # Fixed-size feature vector
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
        "SYMBOL RATE CLASSIFICATION"
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

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    # Convert Hz → class index
    rate_to_class = {
        rate: i
        for i, rate in enumerate(RATES)
    }

    labels = np.asarray([
        rate_to_class[
            min(
                RATES,
                key=lambda r: abs(
                    r - rate
                )
            )
        ]
        for rate in true_rates
    ])

    print(
        f"Train: {len(train_idx)}"
    )

    print(
        f"Test : {len(test_idx)}"
    )

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------

    print()
    print(
        "Extracting features..."
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
    # Train classifier
    # ------------------------------------------------------------

    print()
    print(
        "Training Random Forest Classifier..."
    )

    model = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced"
    )

    model.fit(
        features[train_idx],
        labels[train_idx]
    )

    predicted_classes = model.predict(
        features[test_idx]
    )

    actual_classes = labels[
        test_idx
    ]

    # Convert class → Hz
    actual_rates = RATES[
        actual_classes
    ]

    predicted_rates = RATES[
        predicted_classes
    ]

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    accuracy = accuracy_score(
        actual_classes,
        predicted_classes
    )

    mae = mean_absolute_error(
        actual_rates,
        predicted_rates
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_rates,
            predicted_rates
        )
    )

    r2 = r2_score(
        actual_rates,
        predicted_rates
    )

    correlation = np.corrcoef(
        actual_rates,
        predicted_rates
    )[0, 1]

    print()
    print("=" * 70)
    print(
        "SYMBOL RATE CLASSIFICATION RESULTS"
    )
    print("=" * 70)

    print(
        f"Accuracy    : {accuracy * 100:.2f}%"
    )

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
    # Classification report
    # ------------------------------------------------------------

    names = [
        "50.000 kHz",
        "76.923 kHz",
        "100.000 kHz",
        "125.000 kHz",
        "142.857 kHz",
        "200.000 kHz"
    ]

    print()
    print(
        classification_report(
            actual_classes,
            predicted_classes,
            target_names=names,
            digits=4
        )
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        actual_classes,
        predicted_classes
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

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
        actual_classes=actual_classes,
        predicted_classes=predicted_classes,
        actual_symbol_rate=actual_rates,
        predicted_symbol_rate=predicted_rates,
        candidate_rates=RATES,
        accuracy=accuracy,
        mae=mae,
        rmse=rmse,
        r2=r2,
        correlation=correlation,
        confusion_matrix=cm,
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
        "SYMBOL RATE CLASSIFICATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
