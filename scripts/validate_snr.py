from amc.inference import SignalAnalyzer
from amc.signal_generator import generate

analyzer = SignalAnalyzer()

snr_values = [-10, -5, 0, 5, 10, 15, 20]

print("SNR VALIDATION")
print("=" * 75)

for snr in snr_values:
    signal, metadata = generate(
        label="QPSK",
        num_samples=1024,
        snr_db=snr,
        sample_rate=1e6,
        symbol_rate=1e5,
        cfo_hz=500,
        seed=42,
    )

    result = analyzer.analyze(signal)

    error = abs(result["snr_db"] - snr)

    print(
        f"Input SNR={snr:6.1f} dB | "
        f"Estimated={result['snr_db']:7.2f} dB | "
        f"Error={error:6.2f} dB | "
        f"Mod={result['modulation']:6s} | "
        f"Confidence={result['modulation_confidence']*100:6.2f}%"
    )

print("=" * 75)
print("SNR VALIDATION COMPLETE")
