import numpy as np


# ============================================================
# SNR FEATURES
# Exact feature definition used by the original SNR model
# ============================================================

def normalize_iq(x):
    x = np.asarray(x, dtype=np.complex64)
    x = x - np.mean(x)
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


def snr_features(x, nfft=256):
    x = normalize_iq(x)

    i = np.real(x)
    q = np.imag(x)
    amp = np.abs(x)

    ph = np.unwrap(np.angle(x))
    dph = np.diff(ph)

    window = np.hanning(min(len(x), nfft))

    p = np.abs(
        np.fft.fft(
            x[:nfft] * window,
            n=nfft
        )
    ) ** 2

    return np.array(
        [
            i.mean(),
            q.mean(),
            i.std(),
            q.std(),
            amp.mean(),
            amp.std(),
            np.mean(dph),
            np.std(dph),
            np.max(p),
            np.argmax(p) / nfft,
            *np.quantile(
                amp,
                [.1, .25, .5, .75, .9]
            )
        ],
        dtype=np.float32
    )


# ============================================================
# CFO FEATURES
# Exact feature definition used during CFO model training
# ============================================================

def cfo_features(signal, sample_rate=1e6):

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
            30,
            dtype=np.float32
        )

    x = x / np.sqrt(power)

    phase_diff = np.angle(
        x[1:] * np.conj(x[:-1])
    )

    inst_freq = (
        phase_diff
        * sample_rate
        / (2.0 * np.pi)
    )

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

        features.extend(
            [
                np.mean(freq),
                np.median(freq),
                np.std(freq)
            ]
        )

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

    features.extend(
        [
            freqs[peak],
            spectral_centroid,
            spectral_spread,
            np.max(power_spectrum) / total,
            np.percentile(
                power_spectrum,
                90
            )
        ]
    )

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


# ============================================================
# SYMBOL RATE FEATURES
# Exact feature definition used during symbol-rate training
# ============================================================

def symbol_rate_features(signal, sample_rate=1e6):

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
    features.extend(
        [
            np.mean(envelope),
            np.std(envelope),
            np.median(envelope),
            np.percentile(envelope, 10),
            np.percentile(envelope, 25),
            np.percentile(envelope, 75),
            np.percentile(envelope, 90),
        ]
    )

    # Autocorrelation
    real_x = np.real(x)
    real_x -= np.mean(real_x)

    acf = np.correlate(
        real_x,
        real_x,
        mode="full"
    )

    acf = acf[len(real_x) - 1:]

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

    features.extend(
        [
            float(peak_lag),
            float(peak_value)
        ]
    )

    # Complex autocorrelation
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

    # Envelope spectrum
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

        features.extend(
            [
                float(freqs[peak]),
                float(centroid),
                float(spread),
                float(
                    power_spectrum[peak]
                    / total
                )
            ]
        )

    else:

        features.extend(
            [
                0.0,
                0.0,
                0.0,
                0.0
            ]
        )

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
