from amc.inference import SignalAnalyzer
from amc.signal_generator import generate

analyzer = SignalAnalyzer()

cfo_values = [-3000, -2000, -1000, 0, 1000, 2000, 3000]

print("CFO VALIDATION")
print("=" * 85)

for cfo in cfo_values:
    signal, metadata = generate(
        label="QPSK",
        num_samples=1024,
        snr_db=10,
        sample_rate=1e6,
        symbol_rate=1e5,
        cfo_hz=cfo,
        seed=42,
    )

    result = analyzer.analyze(signal)

    error = abs(result["cfo_hz"] - cfo)

    print(
        f"Input CFO={cfo:8.1f} Hz | "
        f"Estimated={result['cfo_hz']:10.2f} Hz | "
        f"Error={error:9.2f} Hz"
    )

print("=" * 85)
print("CFO VALIDATION COMPLETE")
