import numpy as np


def estimate_cfo_phase(signal, sample_rate=1e6):
    """
    Baseline phase-slope CFO estimator.

    Uses the phase progression of the received complex signal.
    A robust median-based estimate is used to reduce the effect
    of noisy instantaneous-frequency samples.
    """

    x = np.asarray(signal, dtype=np.complex64)

    if len(x) < 2:
        raise ValueError("Signal must contain at least 2 samples.")

    # Remove DC
    x = x - np.mean(x)

    # Normalize amplitude
    power = np.mean(np.abs(x) ** 2)

    if power <= 1e-12:
        return 0.0

    x = x / np.sqrt(power)

    # One-sample complex phase difference
    phase_diff = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    # Convert phase increment to frequency
    inst_freq = (
        phase_diff *
        sample_rate /
        (2.0 * np.pi)
    )

    # Robust estimate
    return float(np.median(inst_freq))


def evaluate():

    data = np.load(
        "data/snr_dataset.npz",
        allow_pickle=True
    )

    X = data["X"]
    meta = data["meta"]

    true_cfo = np.array([
        float(m["cfo_hz"])
        for m in meta
    ])

    classes = np.array([
        str(m["label"])
        for m in meta
    ])

    estimated_cfo = np.array([
        estimate_cfo_phase(signal)
        for signal in X
    ])

    errors = estimated_cfo - true_cfo

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    print("=" * 70)
    print("PHASE-SLOPE CFO ESTIMATOR")
    print("=" * 70)

    print(f"Signals: {len(X)}")

    print()
    print("OVERALL")
    print("-" * 70)
    print(f"MAE  : {mae:.4f} Hz")
    print(f"RMSE : {rmse:.4f} Hz")

    print()
    print("BY MODULATION")
    print("-" * 70)

    for modulation in sorted(np.unique(classes)):

        mask = classes == modulation

        class_error = (
            estimated_cfo[mask] -
            true_cfo[mask]
        )

        class_mae = np.mean(
            np.abs(class_error)
        )

        class_rmse = np.sqrt(
            np.mean(class_error ** 2)
        )

        print(
            f"{modulation:>6s} | "
            f"MAE: {class_mae:8.2f} Hz | "
            f"RMSE: {class_rmse:8.2f} Hz | "
            f"N={np.sum(mask)}"
        )

    print()
    print("BY SNR")
    print("-" * 70)

    snrs = np.array([
        float(m["snr_db"])
        for m in meta
    ])

    for snr in sorted(np.unique(snrs)):

        mask = snrs == snr

        error = (
            estimated_cfo[mask] -
            true_cfo[mask]
        )

        print(
            f"SNR {snr:>5.0f} dB | "
            f"MAE: {np.mean(np.abs(error)):8.2f} Hz | "
            f"RMSE: {np.sqrt(np.mean(error ** 2)):8.2f} Hz | "
            f"N={np.sum(mask)}"
        )

    np.savez_compressed(
        "models/cfo_phase_results.npz",
        true_cfo=true_cfo,
        estimated_cfo=estimated_cfo,
        mae=mae,
        rmse=rmse
    )

    print()
    print("=" * 70)
    print("Saved: models/cfo_phase_results.npz")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
