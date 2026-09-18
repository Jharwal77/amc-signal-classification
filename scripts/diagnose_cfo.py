import numpy as np

d = np.load("models/cfo_rf_results.npz")

true = d["actual_cfo"]
pred = d["predicted_cfo"]

bins = [
    (-3000, -2000),
    (-2000, -1000),
    (-1000, 0),
    (0, 1000),
    (1000, 2000),
    (2000, 3000),
]

print("=" * 70)
print("CFO ERROR BY CFO RANGE")
print("=" * 70)

for low, high in bins:

    mask = (true >= low) & (true < high)

    if np.sum(mask) == 0:
        continue

    error = pred[mask] - true[mask]

    print(
        f"{low:>5} to {high:>5} Hz | "
        f"MAE: {np.mean(np.abs(error)):8.2f} Hz | "
        f"RMSE: {np.sqrt(np.mean(error**2)):8.2f} Hz | "
        f"Bias: {np.mean(error):8.2f} Hz | "
        f"N={np.sum(mask)}"
    )
