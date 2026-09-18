import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from amc.preprocessing import to_features


DATA_PATH = "data/snr_dataset.npz"
MODEL_PATH = "models/snr_rf.joblib"
RESULT_PATH = "models/snr_rf_results.npz"


def extract_features(X):
    features = []

    for i, signal in enumerate(X):
        if (i + 1) % 1000 == 0:
            print(f"Extracting features: {i + 1}/{len(X)}")

        features.append(to_features(signal))

    return np.asarray(features, dtype=np.float32)


def main():

    print("=" * 70)
    print("SUPERVISED SNR ESTIMATION - RANDOM FOREST REGRESSOR")
    print("=" * 70)

    data = np.load(DATA_PATH, allow_pickle=True)

    X = data["X"]
    meta = data["meta"]

    true_snr = np.array([
        float(m["snr_db"])
        for m in meta
    ])

    print(f"Signals: {len(X)}")

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    print()
    print("Extracting signal features...")

    X_features = extract_features(X)

    print(f"Feature matrix: {X_features.shape}")

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=42,
        stratify=true_snr
    )

    print()
    print(f"Training samples: {len(train_idx)}")
    print(f"Testing samples : {len(test_idx)}")

    # ---------------------------------------------------------
    # Train regressor
    # ---------------------------------------------------------

    print()
    print("Training Random Forest Regressor...")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    model.fit(
        X_features[train_idx],
        true_snr[train_idx]
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    predicted_snr = model.predict(
        X_features[test_idx]
    )

    actual_snr = true_snr[test_idx]

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    mae = mean_absolute_error(
        actual_snr,
        predicted_snr
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_snr,
            predicted_snr
        )
    )

    r2 = r2_score(
        actual_snr,
        predicted_snr
    )

    print()
    print("=" * 70)
    print("SNR REGRESSION RESULTS")
    print("=" * 70)

    print(f"MAE  : {mae:.4f} dB")
    print(f"RMSE : {rmse:.4f} dB")
    print(f"R2   : {r2:.4f}")

    # ---------------------------------------------------------
    # SNR-wise evaluation
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("SNR-WISE RESULTS")
    print("=" * 70)

    snr_values = sorted(
        np.unique(actual_snr)
    )

    snr_results = []

    for snr in snr_values:

        mask = actual_snr == snr

        errors = (
            predicted_snr[mask] -
            actual_snr[mask]
        )

        snr_mae = np.mean(
            np.abs(errors)
        )

        snr_rmse = np.sqrt(
            np.mean(errors ** 2)
        )

        bias = np.mean(errors)

        count = np.sum(mask)

        snr_results.append([
            snr,
            snr_mae,
            snr_rmse,
            bias,
            count
        ])

        print(
            f"SNR {snr:>5.0f} dB | "
            f"MAE: {snr_mae:6.3f} dB | "
            f"RMSE: {snr_rmse:6.3f} dB | "
            f"Bias: {bias:7.3f} dB | "
            f"N={count}"
        )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    np.savez_compressed(
        RESULT_PATH,
        actual_snr=actual_snr,
        predicted_snr=predicted_snr,
        snr_results=np.asarray(snr_results),
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
    print("SUPERVISED SNR ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
