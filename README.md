# NTRO SIH26147: Automated .IQ & .WAV Signal Analyzer

[![SIH Problem Statement](https://img.shields.io/badge/SIH-SIH26147-blue.svg)](https://www.sih.gov.in/)
[![Theme](https://img.shields.io/badge/Theme-Space%20Technology-purple.svg)]()
[![Organization](https://img.shields.io/badge/Organization-NTRO-red.svg)]()
[![Python](https://img.shields.io/badge/Python-3.14%2B-green.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)]()

> **Automated model for analysis of `.IQ` and `.wav` files along with blind signal parameter extraction, automated modulation classification (AMC), demodulation, de-interleaving, forward error correction (FEC), and bitstream correlation.**

Developed for the **National Technical Research Organisation (NTRO)** under the **Smart India Hackathon (SIH)** — Space Technology Theme.

---

## Key Capabilities & Features

### 1. Multi-Format Ingestion
- **`.WAV` Files**: Native support for 8, 16, 24, and 32-bit PCM as well as 32-bit float; stereo I/Q channel parsing or mono analytic signal construction via Hilbert transform.
- **`.IQ` / `.BIN` / `.RAW` Files**: Formats including `complex64` (GNU Radio), `int16` (HackRF/USRP/BladeRF), `uint8` (RTL-SDR `cu8`), and `int8` (`cs8`).
- **SigMF Standard**: Automatic discovery and parsing of `.sigmf-meta` companion files to extract sample rates, center frequencies, and data types.
- **Signal Conditioning**: Automated DC offset removal (carrier leakage suppression) and AGC RMS power normalization.

### 2. High-Precision Parameter Extraction
- **Carrier Frequency Offset (CFO)**: Peak spectral tracking and non-linear squaring / fourth-power carrier recovery.
- **Blind Symbol / Baud Rate ($R_s$)**: Non-linear envelope spectrum cyclostationary peak detection and wavelet derivative analysis.
- **Signal-to-Noise Ratio (SNR)**: Moment-based $M_2 M_4$ estimator combined with spectral noise floor integration.
- **Spectral Metrics**: 99% Occupied Bandwidth (OBW) and 3dB roll-off bandwidth calculations.

### 3. Automated Modulation Classification (AMC)
- **Higher-Order Cumulants (HOC)**: Computes $C_{20}$, $C_{40}$, $C_{42}$, $C_{41}$, and $C_{63}$ after coarse CFO de-rotation.
- **Instantaneous Azzouz-Nandi Features**: Amplitude spectral peak $\gamma_{\max}$, phase variance $\sigma_{ap}$, envelope variance $\sigma_{aa}$, and instantaneous frequency hopping variance $\sigma_{af}$.
- **Supported Modulation Classes**: `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `2FSK`, `4FSK`, `CW`, and `NOISE`.

### 4. Demodulation Suite
- **FSK Demodulator**: Instantaneous frequency discriminator with tone-matched filtering and symbol slicing for 2-FSK and 4-FSK.
- **PSK Demodulator**: Root-Raised Cosine (RRC) matched filter, Costas Loop for carrier phase tracking, Gardner Timing Error Detector, and Gray-coded constellation slicers for BPSK, QPSK, and 8PSK.
- **QAM Demodulator**: Adaptive Constant Modulus Algorithm (CMA) / LMS equalizer with multi-threshold grid slicers for 16-QAM and 64-QAM.

### 5. De-Interleaving Engine
- **Block De-interleaver**: Matrix $M \times N$ transpositions.
- **Convolutional De-interleaver**: Ramsey / Forney multi-branch shift registers with complementary delay branches.
- **Diagonal De-interleaver**: Diagonal block mapping for satellite and terrestrial links.
- **Pseudo-Random (PR) De-interleaver**: LFSR and seed-based deterministic permutation inverters.
- **Blind Detection**: Bitstream autocorrelation analysis for interleaver depth/period estimation.

### 6. Forward Error Correction (FEC)
- **Convolutional Code & Viterbi Decoder**: NASA/CCSDS standard ($K=7$, Rate 1/2, polynomials $G_1=171_8, G_2=133_8$) with trellis maximum-likelihood path traceback.
- **Reed-Solomon (RS) Codec**: Galois Field $GF(2^8)$ arithmetic with Berlekamp-Massey syndrome solver and exact $GF(2^8)$ Gaussian elimination error magnitude evaluation. Defaults to CCSDS RS(255, 223) and DVB RS(204, 188).
- **Concatenated Codes**: Outer RS + Interleaver + Inner Viterbi deep-space communications codec.
- **LDPC Decoder**: Tanner graph message-passing Belief Propagation / Min-Sum LLR decoding.

### 7. Bitstream Correlation & Framing
- **Preamble Correlation**: Sliding-window cross-correlation with Hamming distance tolerance for:
  - CCSDS Attached Sync Marker (ASM): `0x1ACFFC1D`
  - Barker Codes (Barker-11, Barker-13)
  - AX.25 HDLC Flag: `0x7E`
  - DVB-S Sync Byte: `0x47`
- **Frame Segmentation**: Header extraction, payload extraction, CRC verification, and Shannon entropy analysis.
- **Hex/ASCII Inspector**: Colorized hex dump and ASCII side-by-side view.

---

## Directory Structure

```
sih26147-signal-analyzer/
├── requirements.txt            # Frozen dependencies
├── README.md                   # System documentation
├── launch.py                   # One-click browser GUI launcher
├── ntro_cli.py                 # Headless batch CLI processor
├── core/
│   ├── __init__.py
│   ├── ingestion.py            # .WAV & .IQ file parser and preprocessor
│   ├── synthetic_generator.py  # Realistic RF signal synthesizer with impairments
│   ├── parameter_extraction.py # PSD, CFO, Baud rate, OBW, and M2M4 SNR
│   ├── amc_classifier.py       # Higher-order cumulants & AMC classification
│   ├── demodulator.py          # FSK, PSK (Costas loop), and QAM demodulators
│   ├── deinterleaver.py        # Block, Conv, Diag, and PR de-interleavers
│   ├── fec_decoder.py          # Viterbi K=7, Reed-Solomon GF(2^8), Concatenated, LDPC
│   ├── bitstream_correlator.py # Cross-correlation, header/payload extraction, entropy
│   └── pipeline.py             # Complete end-to-end signal processing pipeline
├── gui/
│   ├── __init__.py
│   ├── server.py               # REST API & static file web server
│   └── static/
│       ├── index.html          # Cyber-defense telemetry UI
│       ├── style.css           # Glassmorphism & military dark theme
│       └── app.js              # HTML5 Canvas Waterfall, PSD, Constellation, Eye Diagram
├── samples/                    # Ready-to-use sample dataset
│   ├── satellite_telemetry_qpsk.iq
│   ├── terrestrial_2fsk.wav
│   ├── deepspace_bpsk.iq
│   └── deepspace_bpsk.sigmf-meta
└── tests/
    ├── __init__.py
    └── test_signal_analyzer.py # 100% passing automated test suite
```

---

## Getting Started

### 1. Launch Interactive Web Dashboard (Recommended)
Run the launcher script to start the local server and automatically open the GUI in your default browser:

```powershell
.venv\Scripts\python.exe launch.py
```
Or with standard Python:
```powershell
python launch.py
```
Navigate to: **`http://127.0.0.1:8080/`**

### 2. Run Headless Command Line Interface (CLI)

**Analyze a file on disk:**
```powershell
.venv\Scripts\python.exe ntro_cli.py --input samples\terrestrial_2fsk.wav
```

**Analyze raw `.IQ` with SigMF metadata:**
```powershell
.venv\Scripts\python.exe ntro_cli.py --input samples\deepspace_bpsk.iq
```

**Synthesize and evaluate a test signal on the fly:**
```powershell
.venv\Scripts\python.exe ntro_cli.py --generate QPSK --snr 22 --cfo 2500 --export report.json
```

---

## Running the Automated Test Suite

Execute the comprehensive unit and integration test suite:

```powershell
.venv\Scripts\python.exe -m unittest tests\test_signal_analyzer.py -v
```

All tests verify:
- Ingestion and reading of both `.wav` and `.iq` formats.
- Exact bit recovery across all 4 interleaver types (Block, Conv, Diag, PR).
- Error correction under transmission bit-flips for the Viterbi $K=7$ decoder.
- Reed-Solomon $GF(2^8)$ Berlekamp-Massey and Gaussian elimination syndrome correction.
- Tanner graph LDPC belief propagation decoding.
- AMC classification accuracy on BPSK, QPSK, 16QAM, and 2FSK.
- Preamble synchronization on CCSDS markers.
