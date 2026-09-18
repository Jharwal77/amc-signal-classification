from amc.inference import SignalAnalyzer
from amc.signal_generator import generate, CLASSES

analyzer = SignalAnalyzer()

print("MODULATION VALIDATION")
print("=" * 100)

for mod in CLASSES:
    signal, metadata = generate(
        label=mod,
        num_samples=1024,
        snr_db=10,
        sample_rate=1e6,
        symbol_rate=1e5,
        cfo_hz=500,
        seed=42,
    )

    result = analyzer.analyze(signal)

    print(
        f"{mod:6s} -> "
        f"{result['modulation']:6s} | "
        f"Confidence={result['modulation_confidence']*100:6.2f}% | "
        f"SNR={result['snr_db']:7.2f} dB | "
        f"CFO={result['cfo_hz']:9.2f} Hz | "
        f"SR={result['symbol_rate_khz']:8.3f} kHz"
    )

print("=" * 100)
print("VALIDATION COMPLETE")
