import numpy as np
from amc.signal_generator import generate

signal, metadata = generate(
    label="QPSK",
    num_samples=1024,
    snr_db=10,
    sample_rate=1e6,
    symbol_rate=1e5,
    cfo_hz=500,
    seed=123,
)

np.save("data/test_qpsk.npy", signal)

np.savez_compressed(
    "data/test_qpsk.npz",
    iq=signal
)

csv_data = np.column_stack([
    signal.real,
    signal.imag
])

np.savetxt(
    "data/test_qpsk.csv",
    csv_data,
    delimiter=",",
    header="I,Q",
    comments=""
)

print("UPLOAD TEST FILES CREATED")
print("data/test_qpsk.npy")
print("data/test_qpsk.npz")
print("data/test_qpsk.csv")
print(f"Samples: {len(signal)}")
print(f"Expected modulation: {metadata['label']}")
print(f"Expected SNR: {metadata['snr_db']} dB")
print(f"Expected CFO: {metadata['cfo_hz']} Hz")
print(f"Expected symbol rate: {metadata['symbol_rate']/1000:.3f} kHz")
