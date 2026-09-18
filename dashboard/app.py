from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from amc.inference import SignalAnalyzer
from amc.signal_generator import generate


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="AI Wireless Signal Analyzer",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-title {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
    }

    .status {
        padding: 0.6rem 1rem;
        border-radius: 8px;
        border: 1px solid rgba(128,128,128,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Header
# ============================================================

st.title("AI Wireless Signal Analyzer")

st.markdown(
    """
    **AI-Based Automatic Modulation Classification and Signal
    Parameter Estimation for Wireless Communication Systems**
    """
)

st.caption(
    "CNN-based modulation classification | Random Forest parameter estimation"
)


# ============================================================
# Load analyzer once
# ============================================================

@st.cache_resource
def load_analyzer():
    return SignalAnalyzer()


try:
    analyzer = load_analyzer()
except Exception as e:
    st.error(f"Failed to load AI models: {e}")
    st.stop()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Signal Input")

input_mode = st.sidebar.radio(
    "Input Mode",
    [
        "Generate Synthetic Signal",
        "Upload IQ Signal",
    ],
)


# ============================================================
# Signal generation
# ============================================================

signal = None
sample_rate = 1e6
signal_description = ""


if input_mode == "Generate Synthetic Signal":

    st.sidebar.subheader("Generator Parameters")

    modulation_input = st.sidebar.selectbox(
        "Modulation",
        [
            "BPSK",
            "QPSK",
            "8PSK",
            "16QAM",
            "64QAM",
            "GFSK",
            "AM",
            "FM",
        ],
        index=1,
    )

    snr_input = st.sidebar.slider(
        "SNR (dB)",
        min_value=-10.0,
        max_value=20.0,
        value=10.0,
        step=1.0,
    )

    cfo_input = st.sidebar.slider(
        "CFO (Hz)",
        min_value=-3000,
        max_value=3000,
        value=500,
        step=100,
    )

    symbol_rate_options = {
        "50 kHz": 50_000.0,
        "76.923 kHz": 1e6 / 13,
        "100 kHz": 100_000.0,
        "125 kHz": 125_000.0,
        "142.857 kHz": 1e6 / 7,
        "200 kHz": 200_000.0,
    }

    symbol_rate_label = st.sidebar.selectbox(
        "Symbol Rate",
        list(symbol_rate_options.keys()),
        index=2,
    )

    symbol_rate_input = symbol_rate_options[
        symbol_rate_label
    ]

    samples_input = st.sidebar.selectbox(
        "Number of Samples",
        [1024, 2048, 4096, 8192],
        index=0,
    )

    seed_input = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=123,
        step=1,
    )

    generate_button = st.sidebar.button(
        "Generate & Analyze",
        type="primary",
        width="stretch",
    )

    if generate_button or "signal" not in st.session_state:

        signal, metadata = generate(
            label=modulation_input,
            num_samples=samples_input,
            snr_db=snr_input,
            symbol_rate=symbol_rate_input,
            cfo_hz=cfo_input,
            sample_rate=sample_rate,
            seed=int(seed_input),
        )

        st.session_state.signal = signal
        st.session_state.sample_rate = sample_rate
        st.session_state.signal_description = (
            f"Generated {modulation_input} signal"
        )

    signal = st.session_state.signal
    sample_rate = st.session_state.sample_rate
    signal_description = st.session_state.signal_description


# ============================================================
# Upload IQ signal
# ============================================================

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload IQ data",
        type=["npy", "npz", "csv"],
        help="Upload a complex NumPy array or IQ data.",
    )

    sample_rate = st.sidebar.number_input(
        "Sample Rate (Hz)",
        min_value=1.0,
        value=1_000_000.0,
        step=1000.0,
    )

    if uploaded_file is not None:

        try:

            suffix = Path(uploaded_file.name).suffix.lower()

            if suffix == ".npy":

                signal = np.load(
                    uploaded_file,
                    allow_pickle=False,
                )

            elif suffix == ".npz":

                data = np.load(
                    uploaded_file,
                    allow_pickle=False,
                )

                if len(data.files) == 0:
                    raise ValueError(
                        "NPZ file contains no arrays."
                    )

                signal = data[data.files[0]]

            elif suffix == ".csv":

                data = np.loadtxt(
                    uploaded_file,
                    delimiter=",",
                )

                if np.iscomplexobj(data):

                    signal = data

                elif data.ndim == 2 and data.shape[1] >= 2:

                    signal = (
                        data[:, 0]
                        + 1j * data[:, 1]
                    )

                elif data.ndim == 1:

                    signal = data.astype(
                        np.complex64
                    )

                else:

                    raise ValueError(
                        "CSV must contain either one real column "
                        "or at least two columns containing I and Q."
                    )

            else:

                raise ValueError(
                    "Unsupported file format."
                )

            signal = np.asarray(
                signal,
                dtype=np.complex64,
            ).flatten()

            if signal.size == 0:

                raise ValueError(
                    "The uploaded file contains no samples."
                )

            if not np.all(np.isfinite(signal.real)):

                raise ValueError(
                    "I data contains NaN or infinite values."
                )

            if not np.all(np.isfinite(signal.imag)):

                raise ValueError(
                    "Q data contains NaN or infinite values."
                )

            st.session_state.signal = signal

            st.session_state.sample_rate = float(
                sample_rate
            )

            st.session_state.signal_description = (
                f"Uploaded: {uploaded_file.name}"
            )

            st.sidebar.success(
                f"Loaded {len(signal):,} IQ samples"
            )

            st.sidebar.caption(
                f"Format: {suffix.upper()} | "
                f"Sample rate: {sample_rate / 1e6:.3f} MHz"
            )

        except Exception as e:

            st.sidebar.error(
                f"Could not load file: {e}"
            )

            st.session_state.pop(
                "signal",
                None,
            )

            st.session_state.pop(
                "analysis_result",
                None,
            )

    signal = st.session_state.get(
        "signal",
        None,
    )

    signal_description = st.session_state.get(
        "signal_description",
        "",
    )


# ============================================================
# Analyze button
# ============================================================

if signal is None:

    st.info(
        "Configure a signal in the sidebar and click "
        "**Generate & Analyze**."
    )

    st.stop()


analyze_button = st.button(
    "?? Analyze Signal",
    type="primary",
)


if (
    analyze_button
    or "analysis_result" not in st.session_state
):

    with st.spinner(
        "Running AI signal analysis..."
    ):

        try:

            result = analyzer.analyze(
                signal,
                sample_rate=sample_rate,
            )

            st.session_state.analysis_result = result

        except Exception as e:

            st.error(
                f"Signal analysis failed: {e}"
            )

            st.stop()


result = st.session_state.analysis_result


# ============================================================
# Status
# ============================================================

st.markdown(
    f"""
    <div class="status">
    <b>Analysis Complete</b> &nbsp; | &nbsp;
    {signal_description} &nbsp; | &nbsp;
    {len(signal):,} samples
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")


# ============================================================
# Prediction metrics
# ============================================================

st.subheader("AI Signal Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Modulation",
        result["modulation"],
    )

    st.caption(
        f"Confidence: "
        f"{result['modulation_confidence'] * 100:.2f}%"
    )

with col2:

    st.metric(
        "SNR",
        f"{result['snr_db']:.2f} dB",
    )

with col3:

    st.metric(
        "Carrier Frequency Offset",
        f"{result['cfo_hz']:.2f} Hz",
    )

with col4:

    st.metric(
        "Symbol Rate",
        f"{result['symbol_rate_khz']:.3f} kHz",
    )


# ============================================================
# Input vs AI Estimated Parameters
# ============================================================

if input_mode == "Generate Synthetic Signal":

    st.subheader("Input vs AI Estimated Parameters")

    comparison_data = {
        "Parameter": [
            "Modulation",
            "SNR",
            "Carrier Frequency Offset",
            "Symbol Rate",
        ],
        "Input": [
            modulation_input,
            f"{snr_input:.2f} dB",
            f"{cfo_input:.2f} Hz",
            f"{symbol_rate_input / 1000:.3f} kHz",
        ],
        "AI Estimate": [
            result["modulation"],
            f"{result['snr_db']:.2f} dB",
            f"{result['cfo_hz']:.2f} Hz",
            f"{result['symbol_rate_khz']:.3f} kHz",
        ],
    }

    st.table(comparison_data)


# ============================================================
# Parameter Estimation Error
# ============================================================

if input_mode == "Generate Synthetic Signal":

    st.subheader("Parameter Estimation Error")

    snr_error = abs(
        result["snr_db"] - snr_input
    )

    cfo_error = abs(
        result["cfo_hz"] - cfo_input
    )

    symbol_rate_error = abs(
        result["symbol_rate_khz"]
        - (symbol_rate_input / 1000)
    )

    modulation_correct = (
        result["modulation"] == modulation_input
    )

    error_col1, error_col2, error_col3, error_col4 = st.columns(4)

    with error_col1:
        st.metric(
            "SNR Absolute Error",
            f"{snr_error:.2f} dB",
        )

    with error_col2:
        st.metric(
            "CFO Absolute Error",
            f"{cfo_error:.2f} Hz",
        )

    with error_col3:
        st.metric(
            "Symbol Rate Error",
            f"{symbol_rate_error:.3f} kHz",
        )

    with error_col4:
        st.metric(
            "Modulation",
            "Correct" if modulation_correct else "Mismatch",
        )


# ============================================================
# Probability distribution
# ============================================================
# ============================================================

st.subheader("Modulation Confidence Distribution")

probabilities = result[
    "modulation_probabilities"
]

probability_items = sorted(
    probabilities.items(),
    key=lambda item: item[1],
    reverse=True,
)

probability_names = [
    item[0] for item in probability_items
]

probability_values = [
    item[1] * 100
    for item in probability_items
]

fig, ax = plt.subplots(
    figsize=(10, 4)
)

ax.barh(
    probability_names[::-1],
    probability_values[::-1],
)

ax.set_xlabel(
    "Probability (%)"
)

ax.set_xlim(
    0,
    100,
)

ax.grid(
    axis="x",
    alpha=0.25,
)

fig.tight_layout()

st.pyplot(
    fig,
    width="stretch",
)

plt.close(fig)


# ============================================================
# Signal preparation for visualization
# ============================================================

x = np.asarray(
    signal,
    dtype=np.complex64,
).flatten()

display_length = min(
    len(x),
    2048,
)

x_display = x[:display_length]

time_axis = (
    np.arange(display_length)
    / sample_rate
    * 1e6
)


# ============================================================
# I/Q waveform
# ============================================================

st.subheader("I/Q Waveform")

fig, ax = plt.subplots(
    figsize=(12, 4)
)

ax.plot(
    time_axis,
    np.real(x_display),
    label="I",
)

ax.plot(
    time_axis,
    np.imag(x_display),
    label="Q",
)

ax.set_xlabel(
    "Time (s)"
)

ax.set_ylabel(
    "Amplitude"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

st.pyplot(
    fig,
    width="stretch",
)

plt.close(fig)


# ============================================================
# Constellation + Spectrum
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Constellation")

    constellation_length = min(
        len(x),
        4096,
    )

    constellation = x[
        :constellation_length
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.scatter(
        np.real(constellation),
        np.imag(constellation),
        s=8,
        alpha=0.45,
    )

    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")

    ax.axhline(
        0,
        linewidth=0.7,
    )

    ax.axvline(
        0,
        linewidth=0.7,
    )

    ax.grid(
        alpha=0.25
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)


with col2:

    st.subheader("Frequency Spectrum")

    nfft = min(
        len(x),
        4096,
    )

    spectrum_signal = x[:nfft]

    window = np.hanning(
        nfft
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            spectrum_signal * window
        )
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            nfft,
            d=1 / sample_rate,
        )
    )

    magnitude = 20 * np.log10(
        np.abs(spectrum) + 1e-12
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        frequencies / 1000,
        magnitude,
    )

    ax.set_xlabel(
        "Frequency (kHz)"
    )

    ax.set_ylabel(
        "Magnitude (dB)"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)


# ============================================================
# Amplitude and phase
# ============================================================

st.subheader("Amplitude and Phase")

col1, col2 = st.columns(2)

amplitude = np.abs(
    x_display
)

phase = np.unwrap(
    np.angle(x_display)
)

with col1:

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        time_axis,
        amplitude,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)


with col2:

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.plot(
        time_axis,
        phase,
    )

    ax.set_xlabel(
        "Time (s)"
    )

    ax.set_ylabel(
        "Unwrapped Phase (rad)"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)


# ============================================================
# Technical information
# ============================================================

st.subheader("Technical Information")

info1, info2, info3 = st.columns(3)

with info1:

    st.write(
        "**Modulation Classifier**"
    )

    st.write(
        "1D Convolutional Neural Network"
    )

    st.write(
        "8 supported modulation classes"
    )

with info2:

    st.write(
        "**Parameter Estimation**"
    )

    st.write(
        "Random Forest regression/classification"
    )

    st.write(
        "SNR + CFO + Symbol Rate"
    )

with info3:

    st.write(
        "**Signal**"
    )

    st.write(
        f"Sample Rate: {sample_rate / 1e6:.3f} MHz"
    )

    st.write(
        f"Samples: {len(signal):,}"
    )

# ============================================================
# Export Analysis Report
# ============================================================

st.subheader("Export Analysis Report")

report_lines = [
    "AI WIRELESS SIGNAL ANALYSIS REPORT",
    "===================================",
    "",
    "Signal",
    "------",
    f"Description       : {signal_description}",
    f"Samples           : {len(signal):,}",
    f"Sample Rate       : {sample_rate / 1e6:.3f} MHz",
    "",
    "AI Results",
    "----------",
    f"Modulation        : {result['modulation']}",
    f"Confidence        : {result['modulation_confidence'] * 100:.2f}%",
    f"SNR               : {result['snr_db']:.2f} dB",
    f"CFO               : {result['cfo_hz']:.2f} Hz",
    f"Symbol Rate       : {result['symbol_rate_khz']:.3f} kHz",
]

if input_mode == "Generate Synthetic Signal":

    snr_error = abs(
        result["snr_db"] - snr_input
    )

    cfo_error = abs(
        result["cfo_hz"] - cfo_input
    )

    symbol_rate_error = abs(
        result["symbol_rate_khz"]
        - (symbol_rate_input / 1000)
    )

    modulation_correct = (
        result["modulation"] == modulation_input
    )

    report_lines.extend([
        "",
        "Input Parameters",
        "----------------",
        f"Modulation        : {modulation_input}",
        f"SNR               : {snr_input:.2f} dB",
        f"CFO               : {cfo_input:.2f} Hz",
        f"Symbol Rate       : {symbol_rate_input / 1000:.3f} kHz",
        "",
        "Estimation Error",
        "----------------",
        f"SNR Absolute Error         : {snr_error:.2f} dB",
        f"CFO Absolute Error         : {cfo_error:.2f} Hz",
        f"Symbol Rate Absolute Error : {symbol_rate_error:.3f} kHz",
        f"Modulation Classification  : "
        f"{'Correct' if modulation_correct else 'Mismatch'}",
    ])

report_text = "\n".join(report_lines)

st.download_button(
    label="Download Analysis Report",
    data=report_text,
    file_name="ai_signal_analysis_report.txt",
    mime="text/plain",
    width="stretch",
)


# ============================================================
# Model Performance
# ============================================================

st.subheader("Model Performance")

results_path = Path("models/end_to_end_results.csv")

if results_path.exists():

    performance_df = pd.read_csv(
        results_path
    )

    modulation_accuracy = (
        performance_df["true_modulation"]
        == performance_df["pred_modulation"]
    ).mean() * 100

    snr_mae = np.mean(
        np.abs(
            performance_df["true_snr_db"]
            - performance_df["pred_snr_db"]
        )
    )

    cfo_mae = np.mean(
        np.abs(
            performance_df["true_cfo_hz"]
            - performance_df["pred_cfo_hz"]
        )
    )

    symbol_rate_accuracy = (
        performance_df["true_symbol_rate_hz"]
        == performance_df["pred_symbol_rate_hz"]
    ).mean() * 100

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:

        st.metric(
            "Modulation Accuracy",
            f"{modulation_accuracy:.2f}%",
        )

    with perf_col2:

        st.metric(
            "SNR MAE",
            f"{snr_mae:.2f} dB",
        )

    with perf_col3:

        st.metric(
            "CFO MAE",
            f"{cfo_mae:.0f} Hz",
        )

    with perf_col4:

        st.metric(
            "Symbol Rate Accuracy",
            f"{symbol_rate_accuracy:.2f}%",
        )

    st.caption(
        "Metrics calculated from the saved end-to-end evaluation dataset."
    )

    # --------------------------------------------------------
    # Accuracy by SNR
    # --------------------------------------------------------

    st.markdown("#### Modulation Accuracy vs SNR")

    snr_values = sorted(
        performance_df["true_snr_db"].unique()
    )

    snr_accuracy_values = []

    for snr_value in snr_values:

        snr_subset = performance_df[
            performance_df["true_snr_db"] == snr_value
        ]

        accuracy = (
            snr_subset["true_modulation"]
            == snr_subset["pred_modulation"]
        ).mean() * 100

        snr_accuracy_values.append(
            accuracy
        )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.plot(
        snr_values,
        snr_accuracy_values,
        marker="o",
    )

    ax.set_xlabel(
        "True SNR (dB)"
    )

    ax.set_ylabel(
        "Modulation Accuracy (%)"
    )

    ax.set_title(
        "Modulation Classification Accuracy vs SNR"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Modulation accuracy by class
    # --------------------------------------------------------

    st.markdown("#### Modulation Accuracy by Class")

    class_names = sorted(
        performance_df["true_modulation"].unique()
    )

    class_accuracy = []

    for class_name in class_names:

        class_subset = performance_df[
            performance_df["true_modulation"] == class_name
        ]

        accuracy = (
            class_subset["true_modulation"]
            == class_subset["pred_modulation"]
        ).mean() * 100

        class_accuracy.append(
            accuracy
        )

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.bar(
        class_names,
        class_accuracy,
    )

    ax.set_xlabel(
        "Modulation Class"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Modulation Classification Accuracy by Class"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        width="stretch",
    )

    plt.close(fig)

else:

    st.warning(
        "End-to-end performance results were not found."
    )


# ============================================================
# Confusion Matrices
# ============================================================

st.subheader("Classification Confusion Matrices")

if results_path.exists():

    cm_col1, cm_col2 = st.columns(2)

    # --------------------------------------------------------
    # Modulation confusion matrix
    # --------------------------------------------------------

    with cm_col1:

        st.markdown("#### Modulation Classification")

        modulation_labels = sorted(
            performance_df["true_modulation"].unique()
        )

        modulation_cm = pd.crosstab(
            performance_df["true_modulation"],
            performance_df["pred_modulation"],
        )

        modulation_cm = modulation_cm.reindex(
            index=modulation_labels,
            columns=modulation_labels,
            fill_value=0,
        )

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        image = ax.imshow(
            modulation_cm.values,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(modulation_labels))
        )

        ax.set_yticks(
            range(len(modulation_labels))
        )

        ax.set_xticklabels(
            modulation_labels,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            modulation_labels,
        )

        ax.set_xlabel(
            "Predicted Modulation"
        )

        ax.set_ylabel(
            "True Modulation"
        )

        ax.set_title(
            "Modulation Confusion Matrix"
        )

        for i in range(len(modulation_labels)):

            for j in range(len(modulation_labels)):

                ax.text(
                    j,
                    i,
                    int(modulation_cm.iloc[i, j]),
                    ha="center",
                    va="center",
                )

        fig.colorbar(
            image,
            ax=ax,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Symbol-rate confusion matrix
    # --------------------------------------------------------

    with cm_col2:

        st.markdown("#### Symbol Rate Classification")

        symbol_labels = sorted(
            performance_df["true_symbol_rate_hz"].unique()
        )

        symbol_cm = pd.crosstab(
            performance_df["true_symbol_rate_hz"],
            performance_df["pred_symbol_rate_hz"],
        )

        symbol_cm = symbol_cm.reindex(
            index=symbol_labels,
            columns=symbol_labels,
            fill_value=0,
        )

        symbol_label_text = [
            f"{value / 1000:.3f}"
            for value in symbol_labels
        ]

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        image = ax.imshow(
            symbol_cm.values,
            aspect="auto",
        )

        ax.set_xticks(
            range(len(symbol_labels))
        )

        ax.set_yticks(
            range(len(symbol_labels))
        )

        ax.set_xticklabels(
            symbol_label_text,
            rotation=45,
            ha="right",
        )

        ax.set_yticklabels(
            symbol_label_text,
        )

        ax.set_xlabel(
            "Predicted Symbol Rate (kHz)"
        )

        ax.set_ylabel(
            "True Symbol Rate (kHz)"
        )

        ax.set_title(
            "Symbol Rate Confusion Matrix"
        )

        for i in range(len(symbol_labels)):

            for j in range(len(symbol_labels)):

                ax.text(
                    j,
                    i,
                    int(symbol_cm.iloc[i, j]),
                    ha="center",
                    va="center",
                )

        fig.colorbar(
            image,
            ax=ax,
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            width="stretch",
        )

        plt.close(fig)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "AI-Based Automatic Modulation Classification and "
    "Signal Parameter Estimation for Wireless Communication Systems"
)


st.divider()

st.caption(
    "AI-Based Automatic Modulation Classification and "
    "Signal Parameter Estimation for Wireless Communication Systems"
)

