from amc.inference import SignalAnalyzer
from amc.signal_generator import generate

analyzer = SignalAnalyzer()

symbol_rates = [
    50_000,
    1e6 / 13,
    100_000,
    125_000,
    1e6 / 7,
    200_000,
]

print("SYMBOL RATE VALIDATION")
print("=" * 90)

for rate in symbol_rates:
    signal, metadata = generate(
        label="QPSK",
        num_samples=1024,
        snr_db=10,
        sample_rate=1e6,
        symbol_rate=rate,
        cfo_hz=500,
        seed=42,
    )

    result = analyzer.analyze(signal)

    estimated = result["symbol_rate_hz"]
    error = abs(estimated - rate)

    print(
        f"Input={rate/1000:9.3f} kHz | "
        f"Estimated={estimated/1000:9.3f} kHz | "
        f"Error={error/1000:8.3f} kHz"
    )

print("=" * 90)
print("SYMBOL RATE VALIDATION COMPLETE")
