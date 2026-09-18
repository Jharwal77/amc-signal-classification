import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def estimate_snr(signal):
    """
    Estimate SNR from a complex baseband signal.

    Uses a variance/power-based estimator after removing
    the estimated signal mean.
    """
    x = np.asarray(signal, dtype=np.complex64)

    if len(x) == 0:
        raise ValueError("Signal cannot be empty.")

    # Remove DC component
    x = x - np.mean(x)

    # Total received power
    total_power = np.mean(np.abs(x) ** 2)

    # Estimate noise from first differences.
    # For slowly varying digitally modulated signals, this provides
    # a practical baseline noise-power estimate.
    diff = np.diff(x)

    noise_power = np.mean(np.abs(diff) ** 2) / 2.0

    # Protect against numerical problems
    noise_power = max(float(noise_power), 1e-12)
    total_power = max(float(total_power), 1e-12)

    # Estimated signal power
    signal_power = max(total_power - noise_power, 1e-12)

    snr_linear = signal_power / noise_power

    return float(10.0 * np.log10(snr_linear))


def evaluate_snr_estimator(
    data_path="data/snr_dataset.npz",
    output_path="models/snr_estimation_results.npz"
):
    data = np.load(data_path, allow_pickle=True)

    X = data["X"]
    meta = data["meta"]

    true_snr = np.array([
        float(m["snr_db"])
        for m in meta
    ])

    print("=" * 70)
    print("SNR PARAMETER ESTIMATION")
    print("=" * 70)

    print(f"Signals: {len(X)}")

    estimated_snr = np.array([
        estimate_snr(signal)
        for signal in X
    ])

    # Metrics
    mae = mean_absolute_error(
        true_snr,
        estimated_snr
    )

    rmse = np.sqrt(
        mean_squared_error(
            true_snr,
            estimated_snr
        )
    )

    r2 = r2_score(
        true_snr,
        estimated_snr
    )

    print()
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"MAE  : {mae:.4f} dB")
    print(f"RMSE : {rmse:.4f} dB")
    print(f"R2   : {r2:.4f}")

    # SNR-wise results
    print()
    print("=" * 70)
    print("SNR-WISE ESTIMATION")
    print("=" * 70)

    snr_values = sorted(np.unique(true_snr))
    snr_results = []

    for snr in snr_values:

        mask = true_snr == snr

        errors = estimated_snr[mask] - true_snr[mask]

        snr_mae = np.mean(np.abs(errors))
        snr_rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)

        snr_results.append([
            snr,
            snr_mae,
            snr_rmse,
            bias,
            np.sum(mask)
        ])

        print(
            f"SNR {snr:>5.0f} dB | "
            f"MAE: {snr_mae:6.3f} dB | "
            f"RMSE: {snr_rmse:6.3f} dB | "
            f"Bias: {bias:7.3f} dB | "
            f"N={np.sum(mask)}"
        )

    # Save results
    np.savez_compressed(
        output_path,
        true_snr=true_snr,
        estimated_snr=estimated_snr,
        snr_results=np.asarray(snr_results),
        mae=mae,
        rmse=rmse,
        r2=r2
    )

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(output_path)

    print()
    print("=" * 70)
    print("SNR ESTIMATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_snr_estimator()
