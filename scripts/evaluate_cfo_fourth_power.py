import numpy as np


def estimate_cfo_fourth_power(signal, sample_rate=1e6):

    x = np.asarray(signal, dtype=np.complex64)

    if len(x) < 16:
        return 0.0

    # Remove mean/DC
    x = x - np.mean(x)

    # Normalize
    p = np.mean(np.abs(x) ** 2)

    if p <= 1e-12:
        return 0.0

    x = x / np.sqrt(p)

    # Fourth-power transformation
    z = x ** 4

    # Remove mean from transformed signal
    z = z - np.mean(z)

    n = len(z)

    # Window
    window = np.hanning(n)

    spectrum = np.fft.fftshift(
        np.fft.fft(z * window)
    )

    power = np.abs(spectrum) ** 2

    freqs = np.fft.fftshift(
        np.fft.fftfreq(
            n,
            d=1.0 / sample_rate
        )
    )

    # Search only within a physically meaningful range.
    # Fourth power multiplies CFO by 4.
    valid = np.abs(freqs) <= 12000

    if not np.any(valid):
        return 0.0

    valid_indices = np.where(valid)[0]

    peak_index = valid_indices[
        np.argmax(power[valid])
    ]

    peak_frequency = freqs[peak_index]

    # Undo fourth-power multiplication
    return float(peak_frequency / 4.0)


def main():

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

    snrs = np.array([
        float(m["snr_db"])
        for m in meta
    ])

    print("=" * 70)
    print("FOURTH-POWER CFO ESTIMATOR")
    print("=" * 70)

    estimated = np.array([
        estimate_cfo_fourth_power(signal)
        for signal in X
    ])

    error = estimated - true_cfo

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))

    print()
    print("OVERALL")
    print("-" * 70)
    print(f"MAE  : {mae:.2f} Hz")
    print(f"RMSE : {rmse:.2f} Hz")
    print(
        f"Correlation: "
        f"{np.corrcoef(true_cfo, estimated)[0,1]:.4f}"
    )

    print()
    print("BY MODULATION")
    print("-" * 70)

    for modulation in sorted(np.unique(classes)):

        mask = classes == modulation

        e = error[mask]

        print(
            f"{modulation:>6s} | "
            f"MAE: {np.mean(np.abs(e)):8.2f} Hz | "
            f"RMSE: {np.sqrt(np.mean(e**2)):8.2f} Hz | "
            f"N={np.sum(mask)}"
        )

    print()
    print("BY SNR")
    print("-" * 70)

    for snr in sorted(np.unique(snrs)):

        mask = snrs == snr

        e = error[mask]

        print(
            f"SNR {snr:>5.0f} dB | "
            f"MAE: {np.mean(np.abs(e)):8.2f} Hz | "
            f"RMSE: {np.sqrt(np.mean(e**2)):8.2f} Hz | "
            f"N={np.sum(mask)}"
        )

    np.savez_compressed(
        "models/cfo_fourth_power_results.npz",
        true_cfo=true_cfo,
        estimated_cfo=estimated,
        mae=mae,
        rmse=rmse
    )

    print()
    print("=" * 70)
    print("Saved: models/cfo_fourth_power_results.npz")
    print("=" * 70)


if __name__ == "__main__":
    main()
