import os
import numpy as np
import joblib
import torch

from amc.cnn import AMCNet
from amc.feature_extractors import (
    snr_features,
    cfo_features,
    symbol_rate_features
)

CLASS_NAMES = ["16QAM", "64QAM", "8PSK", "AM", "BPSK", "FM", "GFSK", "QPSK"]

SYMBOL_RATE_CLASSES = np.array([
    50_000.0,
    1e6 / 13,
    100_000.0,
    125_000.0,
    1e6 / 7,
    200_000.0
])


# ============================================================
# Paths
# ============================================================

CNN_PATH = "models/cnn_common.pth"
SNR_PATH = "models/snr_parameter_rf.joblib"
CFO_PATH = "models/cfo_parameter_rf.joblib"
SYMBOL_PATH = "models/symbol_rate_classifier.joblib"


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


class SignalAnalyzer:

    def __init__(self):

        print("Loading models...")

        # ----------------------------------------------------
        # CNN
        # ----------------------------------------------------

        checkpoint = torch.load(
            CNN_PATH,
            map_location=DEVICE
        )

        self.cnn = AMCNet(
            num_classes=len(CLASS_NAMES)
        )

        self.cnn.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.cnn.to(DEVICE)

        self.cnn.eval()

        # ----------------------------------------------------
        # Regression / classification models
        # ----------------------------------------------------

        self.snr_model = joblib.load(
            SNR_PATH
        )

        self.cfo_model = joblib.load(
            CFO_PATH
        )

        self.symbol_model = joblib.load(
            SYMBOL_PATH
        )

        print("All models loaded successfully.")


    # ========================================================
    # General signal preprocessing
    # ========================================================

    def preprocess(self, signal):

        x = np.asarray(
            signal,
            dtype=np.complex64
        )

        if x.ndim != 1:
            x = x.flatten()

        if len(x) == 0:
            raise ValueError(
                "Signal is empty."
            )

        x = x - np.mean(x)

        power = np.mean(
            np.abs(x) ** 2
        )

        if power <= 1e-12:
            raise ValueError(
                "Signal has near-zero power."
            )

        x = x / np.sqrt(power)

        return x


    # ========================================================
    # CNN classification
    # Exact preprocessing used during CNN training/evaluation
    # ========================================================

    def cnn_classify(self, signal):

        target_length = 1024

        signal = np.asarray(
            signal,
            dtype=np.complex64
        ).flatten()

        if len(signal) >= target_length:
            signal_cnn = signal[:target_length]
        else:
            signal_cnn = np.pad(
                signal,
                (0, target_length - len(signal))
            )

        # Convert complex IQ to two channels
        iq = np.stack(
            [
                signal_cnn.real,
                signal_cnn.imag
            ],
            axis=0
        ).astype(np.float32)

        # EXACT normalization used by train_cnn_common.py
        scale = np.sqrt(
            np.mean(iq ** 2)
        )

        if scale > 0:
            iq = iq / scale

        tensor = torch.from_numpy(
            iq
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            logits = self.cnn(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[0]

        class_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        modulation = CLASS_NAMES[
            class_index
        ]

        confidence = float(
            probabilities[
                class_index
            ].item()
        )

        probability_dict = {
            CLASS_NAMES[i]:
            float(probabilities[i].item())
            for i in range(
                len(CLASS_NAMES)
            )
        }

        return (
            modulation,
            confidence,
            probability_dict
        )


    # ========================================================
    # Canonical feature extraction
    # ========================================================

    def extract_snr_features(self, x):
        return snr_features(x)

    def extract_cfo_features(self, x, sample_rate=1e6):
        return cfo_features(
            x,
            sample_rate=sample_rate
        )

    def extract_symbol_features(self, x, sample_rate=1e6):
        return symbol_rate_features(
            x,
            sample_rate=sample_rate
        )

    # ========================================================
    # Complete analysis
    # ========================================================

    def analyze(
        self,
        signal,
        sample_rate=1e6
    ):

        raw_signal = np.asarray(
            signal,
            dtype=np.complex64
        ).flatten()

        if len(raw_signal) == 0:
            raise ValueError(
                "Signal is empty."
            )

        # CNN uses its own training-equivalent preprocessing.
        modulation, confidence, probabilities = (
            self.cnn_classify(raw_signal)
        )

        # RF estimators use the common complex normalization.
        x = self.preprocess(
            raw_signal
        )

        # SNR
        snr_feature_vector = (
            self.extract_snr_features(x)
            .reshape(1, -1)
        )

        snr = float(
            self.snr_model.predict(
                snr_feature_vector
            )[0]
        )

        # CFO
        cfo_feature_vector = (
            self.extract_cfo_features(
                x,
                sample_rate
            )
            .reshape(1, -1)
        )

        cfo = float(
            self.cfo_model.predict(
                cfo_feature_vector
            )[0]
        )

        # Symbol rate
        symbol_feature_vector = (
            self.extract_symbol_features(
                x,
                sample_rate
            )
            .reshape(1, -1)
        )

        symbol_class = int(
            self.symbol_model.predict(
                symbol_feature_vector
            )[0]
        )

        symbol_rate = float(
            SYMBOL_RATE_CLASSES[symbol_class]
        )

        return {

            "modulation": modulation,

            "modulation_confidence":
                confidence,

            "modulation_probabilities":
                probabilities,

            "snr_db": snr,

            "cfo_hz": cfo,

            "symbol_rate_hz":
                symbol_rate,

            "symbol_rate_khz":
                symbol_rate / 1000.0,

            "sample_rate_hz":
                sample_rate,

            "num_samples":
                len(x)
        }


# ============================================================
# Simple command-line demo
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "AI WIRELESS SIGNAL ANALYZER"
    )
    print("=" * 70)

    analyzer = SignalAnalyzer()

    # Demo signal
    from amc.signal_generator import generate

    signal, metadata = generate(
        label="QPSK",
        num_samples=1024,
        snr_db=10,
        symbol_rate=100e3,
        cfo_hz=500,
        seed=123
    )

    result = analyzer.analyze(
        signal
    )

    print()
    print("=" * 70)
    print("ANALYSIS RESULT")
    print("=" * 70)

    print(
        f"Modulation : "
        f"{result['modulation']}"
    )

    print(
        f"Confidence : "
        f"{result['modulation_confidence'] * 100:.2f}%"
    )

    print(
        f"SNR        : "
        f"{result['snr_db']:.2f} dB"
    )

    print(
        f"CFO        : "
        f"{result['cfo_hz']:.2f} Hz"
    )

    print(
        f"Symbol Rate: "
        f"{result['symbol_rate_khz']:.3f} kHz"
    )

    print(
        f"Samples    : "
        f"{result['num_samples']}"
    )

    print()
    print("=" * 70)
    print(
        "END-TO-END INFERENCE COMPLETE"
    )
    print("=" * 70)


