import os
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from amc.preprocessing import to_features


DATA_PATH = "data/parameter_dataset.npz"
SPLIT_PATH = "data/parameter_common_split.npz"

MODEL_PATH = "models/snr_parameter_rf.joblib"
RESULT_PATH = "models/snr_parameter_rf_results.npz"


def main():

    data = np.load(
        DATA_PATH,
        allow_pickle=True
    )

    X = data["X"]
    meta = data["meta"]

    true_snr = np.array([
        float(m["snr_db"])
        for m in meta
    ])

    split = np.load(SPLIT_PATH)

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    print("=" * 70)
    print("SNR ESTIMATION - VARIABLE PARAMETER DATASET")
    print("=" * 70)

    print(f"Train: {len(train_idx)}")
    print(f"Test : {len(test_idx)}")

    # Feature extraction
    print()
    print("Extracting features...")

    features = []

    for i, signal in enumerate(X):

        if (i + 1) % 2000 == 0:
            print(
                f"{i + 1}/{len(X)}"
            )

        features.append(
            to_features(signal)
        )

    features = np.asarray(
        features,
        dtype=np.float32
    )

    print(
        f"Feature matrix: {features.shape}"
    )

    # Train
    print()
    print("Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    model.fit(
        features[train_idx],
        true_snr[train_idx]
    )

    # Predict
    prediction = model.predict(
        features[test_idx]
    )

    actual = true_snr[test_idx]

    # Metrics
    mae = mean_absolute_error(
        actual,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            prediction
        )
    )

    r2 = r2_score(
        actual,
        prediction
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"MAE : {mae:.4f} dB")
    print(f"RMSE: {rmse:.4f} dB")
    print(f"R2  : {r2:.4f}")

    # Save
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
        actual_snr=actual,
        predicted_snr=prediction,
        mae=mae,
        rmse=rmse,
        r2=r2,
        test_idx=test_idx
    )

    print()
    print("Saved:")
    print(MODEL_PATH)
    print(RESULT_PATH)


if __name__ == "__main__":
    main()
