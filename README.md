# AI-Based Automatic Modulation Classification and Signal Parameter Estimation for Wireless Communication Systems

## Overview

This project implements an AI-based wireless signal analysis system for **Automatic Modulation Classification (AMC)** and **signal parameter estimation** from complex baseband I/Q samples.

The system analyzes a received complex I/Q signal and estimates:

- Modulation type
- Modulation confidence
- Signal-to-Noise Ratio (SNR)
- Carrier Frequency Offset (CFO)
- Symbol rate
- Sampling rate
- Number of samples

A Streamlit-based dashboard provides an interactive interface for signal generation, I/Q signal upload, analysis, visualization, and performance evaluation.

---

## Supported Modulation Classes

The system supports eight modulation classes:

1. BPSK
2. QPSK
3. 8PSK
4. 16QAM
5. 64QAM
6. GFSK
7. AM
8. FM

---

## System Architecture

```text
                  Input I/Q Signal
                         │
             ┌───────────┴───────────┐
             │                       │
       Preprocessing            Signal Metadata
             │
             ▼
     ┌─────────────────┐
     │ Feature / I-Q   │
     │ Representation  │
     └────────┬────────┘
              │
       ┌──────┴───────────────────────────────┐
       │                                      │
       ▼                                      ▼
 CNN Modulation Classifier          Parameter Estimation
       │                                      │
       │                         ┌────────────┼─────────────┐
       │                         │            │             │
       ▼                         ▼            ▼             ▼
 Modulation +              SNR Estimator  CFO Estimator  Symbol-Rate
 Confidence                                              Classifier
       │                         │            │             │
       └─────────────────────────┴────────────┴─────────────┘
                                  │
                                  ▼
                         Signal Analysis Result
                                  │
                                  ▼
                         Streamlit Dashboard
```

---

## Main Components

### 1. I/Q Signal Generation

`src/amc/signal_generator.py`

The project can generate synthetic complex baseband signals with configurable:

- Modulation
- SNR
- Carrier frequency offset
- Symbol rate
- Sampling rate
- Number of samples
- Random seed

The generated signals contain complex I/Q samples suitable for training and testing.

---

### 2. Preprocessing

`src/amc/preprocessing.py`

The preprocessing pipeline includes:

- Complex I/Q normalization
- DC removal
- Amplitude extraction
- Phase extraction
- Phase-difference features
- FFT-based spectral features
- Statistical amplitude features

---

### 3. Modulation Classification

`src/amc/cnn.py`

A 1-D convolutional neural network is used for modulation classification.

The CNN processes two-channel I/Q input:

```text
Input: (batch, 2, 1024)
```

The network contains multiple convolutional blocks followed by pooling and fully connected layers.

The final classifier predicts one of the eight supported modulation classes.

Final runtime model:

```text
models/cnn_common.pth
```

---

### 4. SNR Estimation

`src/amc/feature_extractors.py`

The SNR estimator uses engineered signal features and a Random Forest regression model.

Runtime model:

```text
models/snr_parameter_rf.joblib
```

Final end-to-end SNR performance:

```text
MAE  = 0.47 dB
RMSE = 1.07 dB
```

---

### 5. Carrier Frequency Offset Estimation

CFO is estimated using engineered signal features and a Random Forest regression model.

Runtime model:

```text
models/cfo_parameter_rf.joblib
```

Final end-to-end performance:

```text
MAE  = 688.56 Hz
RMSE = 1034.11 Hz
```

CFO estimation is currently the most challenging parameter-estimation component of the system, particularly toward the edges of the supported CFO range.

---

### 6. Symbol Rate Estimation

The system performs symbol-rate classification using engineered signal features.

Supported symbol rates:

```text
50,000 Hz
76,923.08 Hz
100,000 Hz
125,000 Hz
142,857.14 Hz
200,000 Hz
```

Runtime model:

```text
models/symbol_rate_classifier.joblib
```

Final end-to-end symbol-rate classification accuracy:

```text
70.95%
```

The symbol-rate model currently operates over the six supported rate classes.

---

## Dataset

The project uses synthetically generated complex I/Q signals.

The datasets include variation in:

- Modulation
- SNR
- Carrier frequency offset
- Symbol rate
- Random signal realization

Important datasets include:

```text
data/demo.npz
data/snr_dataset.npz
data/common_split.npz
data/parameter_dataset.npz
data/parameter_dataset_v2.npz
data/parameter_common_split.npz
data/parameter_v2_common_split.npz
```

---

## Final End-to-End Performance

The complete system was evaluated using a controlled end-to-end test set covering:

- 8 modulation classes
- 7 SNR levels
- 6 symbol rates
- Multiple signal realizations

Final measured performance:

| Component | Metric | Result |
|---|---:|---:|
| Modulation Classification | Accuracy | **75.24%** |
| SNR Estimation | MAE | **0.47 dB** |
| SNR Estimation | RMSE | **1.07 dB** |
| CFO Estimation | MAE | **688.56 Hz** |
| CFO Estimation | RMSE | **1034.11 Hz** |
| Symbol Rate Classification | Accuracy | **70.95%** |

### Modulation Accuracy by Class

| Modulation | Accuracy |
|---|---:|
| BPSK | 93.81% |
| QPSK | 68.57% |
| 8PSK | 60.48% |
| 16QAM | 56.67% |
| 64QAM | 33.33% |
| GFSK | 97.62% |
| AM | 96.19% |
| FM | 95.24% |

The results show that lower-order and waveform-distinct modulation classes are generally easier to classify, while 64QAM and some higher-order digital modulation classes remain more challenging under the tested conditions.

---

## Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard supports:

### Signal Input

- Synthetic signal generation
- `.npy` upload
- `.npz` upload
- `.csv` upload

### Analysis

- Modulation classification
- Modulation confidence
- SNR estimation
- CFO estimation
- Symbol-rate estimation

### Visualizations

- I/Q waveform
- Constellation diagram
- Frequency spectrum
- Amplitude
- Phase
- Model performance
- Modulation accuracy versus SNR
- Modulation accuracy by class
- Modulation confusion matrix
- Symbol-rate confusion matrix

### Run Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

The dashboard will normally be available at:

```text
http://localhost:8501
```

---

## Installation

### Windows PowerShell

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Set the source path:

```powershell
$env:PYTHONPATH="src"
```

---

## Run Tests

From the project root:

```powershell
pytest -q
```

The validated test suite currently passes:

```text
3 passed
```

---

## Direct Inference Test

A simple QPSK inference test can be performed with:

```powershell
$env:PYTHONPATH="src"

python -c "from amc.inference import SignalAnalyzer; from amc.signal_generator import generate; x,_=generate('QPSK',1024,10,cfo_hz=500,seed=123); print(SignalAnalyzer().analyze(x))"
```

The system loads:

```text
models/cnn_common.pth
models/snr_parameter_rf.joblib
models/cfo_parameter_rf.joblib
models/symbol_rate_classifier.joblib
```

and returns the predicted modulation and estimated signal parameters.

---

## Project Structure

```text
amc_project/
│
├── dashboard/
│   ├── app.py
│   └── path_setup.py
│
├── data/
│   ├── demo.npz
│   ├── snr_dataset.npz
│   ├── common_split.npz
│   ├── parameter_dataset.npz
│   ├── parameter_dataset_v2.npz
│   ├── parameter_common_split.npz
│   ├── parameter_v2_common_split.npz
│   └── test signal files
│
├── models/
│   ├── cnn_common.pth
│   ├── snr_parameter_rf.joblib
│   ├── cfo_parameter_rf.joblib
│   ├── symbol_rate_classifier.joblib
│   └── final evaluation artifacts
│
├── results/
│   ├── final_system_performance.png
│   ├── end_to_end_modulation_accuracy_by_class.png
│   ├── end_to_end_modulation_confusion_matrix.png
│   ├── end_to_end_cfo_mae_by_range.png
│   ├── end_to_end_snr_mae_vs_snr.png
│   └── end_to_end_symbol_rate_confusion_matrix.png
│
├── scripts/
│   ├── dataset generation
│   ├── model training
│   ├── model evaluation
│   └── validation scripts
│
├── src/
│   └── amc/
│       ├── cnn.py
│       ├── cnn_dataset.py
│       ├── estimators.py
│       ├── feature_extractors.py
│       ├── inference.py
│       ├── preprocessing.py
│       ├── signal_generator.py
│       ├── snr_estimator.py
│       └── train.py
│
├── tests/
│   └── test_amc.py
│
├── archive/
│   ├── old_models/
│   ├── old_results/
│   └── old_scripts/
│
├── IMPLEMENTATION_STATUS.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Reproducibility

The project uses fixed random seeds in its dataset-generation and validation workflows where required.

The scripts directory contains the workflows used for:

```text
Signal generation
      ↓
Dataset creation
      ↓
Train/test splitting
      ↓
Feature extraction
      ↓
Model training
      ↓
Model evaluation
      ↓
End-to-end validation
```

The trained runtime models are included in the `models/` directory.

---

## Validation

The final implementation was validated using:

```text
pytest
```

and dedicated validation scripts for:

```text
All modulation classes
SNR estimation
CFO estimation
Symbol-rate estimation
End-to-end system performance
```

The final runtime model-loading path was also verified successfully.

---

## Limitations

The current system has several important limitations:

1. Training and evaluation primarily use synthetically generated signals.
2. Modulation classification performance varies across modulation classes.
3. Higher-order QAM classification remains challenging.
4. CFO estimation has substantially larger error than SNR estimation.
5. Symbol-rate estimation is currently formulated as classification over six supported symbol rates.
6. Real-world channel effects such as multipath fading, hardware impairments, timing offsets, phase noise, and nonlinear amplifier effects are not comprehensively modeled.

These limitations provide clear directions for future improvement.

---

## Future Work

Potential extensions include:

- Training on real-world RF/IQ datasets
- Rayleigh and Rician fading simulation
- Multipath channel modeling
- Timing-offset estimation
- Phase-noise modeling
- Improved CFO estimation
- Transformer-based or attention-based AMC models
- More modulation classes
- Joint multi-task neural-network estimation
- GPU-accelerated training
- Real-time SDR integration
- Deployment on edge/embedded hardware

---

## Final Status

The project currently contains a validated end-to-end AI-based signal analysis pipeline with:

- 8 supported modulation classes
- CNN-based modulation classification
- SNR estimation
- CFO estimation
- Symbol-rate classification
- Synthetic I/Q signal generation
- File-based I/Q input
- Interactive Streamlit dashboard
- Automated tests
- End-to-end evaluation
- Performance visualizations
- Reproducible training and validation scripts

The final runtime models have been separated from older experimental models, which are retained under `archive/`.