import joblib

models = [
    "models/snr_parameter_rf.joblib",
    "models/cfo_parameter_rf.joblib",
    "models/symbol_rate_classifier.joblib"
]

print("=" * 70)
print("MODEL FEATURE AUDIT")
print("=" * 70)

for path in models:

    model = joblib.load(path)

    print()
    print(path)
    print("-" * 70)

    print(
        "Model type:",
        type(model).__name__
    )

    print(
        "n_features_in_:",
        getattr(
            model,
            "n_features_in_",
            "N/A"
        )
    )

    print(
        "n_estimators:",
        getattr(
            model,
            "n_estimators",
            "N/A"
        )
    )

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
