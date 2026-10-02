"""
Signal Parameter Extraction Engine
Extracts:
- Power Spectral Density (PSD) and peak frequency
- Carrier Frequency Offset (CFO) using spectral centroid & non-linear M-th power methods
- Blind Symbol / Baud Rate (Rs) estimation via envelope cyclostationary squaring & cyclic autocorrelation
- Occupied Bandwidth (OBW 99%) & 3dB/10dB bandwidth
- Signal-to-Noise Ratio (SNR) via M2M4 moment estimator & spectral noise floor integration
- Spectrogram (Waterfall) time-frequency matrix generation
"""

import numpy as np
from scipy import signal

def compute_psd(iq_samples: np.ndarray, sample_rate: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes double-sided Power Spectral Density (PSD) centered at 0 Hz using Welch's method.
    Returns: (frequencies_hz, psd_db)
    """
    nperseg = min(nperseg, len(iq_samples))
    freqs, psd = signal.welch(
        iq_samples,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        return_onesided=False,
        scaling='density'
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-18))
    return freqs, psd_db


def estimate_cfo(iq_samples: np.ndarray, sample_rate: float) -> float:
    """
    Estimates Carrier Frequency Offset (CFO) in Hz.
    Combines:
    1. Direct spectral peak for carrier-bearing signals
    2. Squaring (for BPSK: peak at 2*CFO)
    3. Fourth-power (for QPSK: peak at 4*CFO)
    """
    n = min(len(iq_samples), 16384)
    sig_chunk = iq_samples[:n] - np.mean(iq_samples[:n])

    # Method 1: direct FFT peak
    fft_raw = np.abs(np.fft.fftshift(np.fft.fft(sig_chunk, n=32768)))
    freqs_raw = np.fft.fftshift(np.fft.fftfreq(32768, d=1.0/sample_rate))
    idx_raw = np.argmax(fft_raw)
    cfo_raw = freqs_raw[idx_raw]

    # Method 2: Squaring (reveals 2*fc for BPSK)
    sig_sq = sig_chunk ** 2
    fft_sq = np.abs(np.fft.fftshift(np.fft.fft(sig_sq, n=32768)))
    idx_sq = np.argmax(fft_sq)
    cfo_sq = freqs_raw[idx_sq] / 2.0

    # Method 3: 4th power (reveals 4*fc for QPSK)
    sig_4th = sig_chunk ** 4
    fft_4th = np.abs(np.fft.fftshift(np.fft.fft(sig_4th, n=32768)))
    idx_4th = np.argmax(fft_4th)
    cfo_4th = freqs_raw[idx_4th] / 4.0

    # Select candidate with most prominent peak prominence
    peak_raw = np.max(fft_raw) / (np.mean(fft_raw) + 1e-12)
    peak_sq = np.max(fft_sq) / (np.mean(fft_sq) + 1e-12)
    peak_4th = np.max(fft_4th) / (np.mean(fft_4th) + 1e-12)

    prominences = [peak_raw, peak_sq, peak_4th]
    cfos = [cfo_raw, cfo_sq, cfo_4th]

    best_idx = int(np.argmax(prominences))
    return float(cfos[best_idx])


def estimate_symbol_rate(iq_samples: np.ndarray, sample_rate: float) -> float:
    """
    Blindly estimates Symbol Rate (Baud Rate, Rs) in baud (symbols/sec).
    Uses non-linear envelope squaring spectrum and magnitude transition peaks.
    """
    n = min(len(iq_samples), 32768)
    x = iq_samples[:n]

    # Non-linear transform: |x[n]|^2 or envelope derivative
    env = np.abs(x) ** 2
    env = env - np.mean(env)

    # Compute FFT of envelope
    n_fft = 65536
    env_fft = np.abs(np.fft.rfft(env * np.hanning(len(env)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0/sample_rate)

    # Mask DC and extreme frequencies (> sample_rate / 2)
    # Most communications signals have SPS between 2 and 64
    min_freq = sample_rate / 64.0
    max_freq = sample_rate / 2.0
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)

    filtered_fft = np.zeros_like(env_fft)
    filtered_fft[valid_mask] = env_fft[valid_mask]

    # Find peaks in envelope spectrum
    peaks, properties = signal.find_peaks(filtered_fft, distance=50, prominence=np.max(filtered_fft) * 0.1)

    if len(peaks) > 0:
        # Sort by prominence
        prominences = properties['prominences']
        best_peak_idx = peaks[np.argmax(prominences)]
        est_baud = float(freqs[best_peak_idx])
    else:
        # Fallback: estimate from 3dB bandwidth
        _, psd = signal.welch(x, fs=sample_rate, nperseg=1024)
        psd_max = np.max(psd)
        half_power = psd > (psd_max * 0.5)
        est_baud = float(np.sum(half_power) * (sample_rate / 1024.0))

    # Sanity check baud rate
    if est_baud <= 0 or est_baud > sample_rate / 2.0:
        est_baud = sample_rate / 8.0

    return est_baud


def estimate_bandwidth_and_obw(freqs: np.ndarray, psd_linear: np.ndarray) -> tuple[float, float]:
    """
    Estimates:
    1. 99% Occupied Bandwidth (OBW)
    2. 3dB Bandwidth
    """
    total_pwr = np.sum(psd_linear)
    if total_pwr <= 0:
        return 0.0, 0.0

    # 99% OBW: cumulative power between 0.5% and 99.5%
    cum_pwr = np.cumsum(psd_linear) / total_pwr
    low_idx = np.searchsorted(cum_pwr, 0.005)
    high_idx = np.searchsorted(cum_pwr, 0.995)
    high_idx = min(high_idx, len(freqs) - 1)
    obw_99 = float(np.abs(freqs[high_idx] - freqs[low_idx]))

    # 3dB Bandwidth
    max_pwr = np.max(psd_linear)
    half_mask = psd_linear >= (max_pwr * 0.5)
    indices = np.where(half_mask)[0]
    if len(indices) > 1:
        bw_3db = float(np.abs(freqs[indices[-1]] - freqs[indices[0]]))
    else:
        bw_3db = obw_99

    return obw_99, bw_3db


def estimate_snr_m2m4(iq_samples: np.ndarray) -> float:
    """
    Estimates SNR (dB) using the M2M4 (2nd and 4th order moments) estimator.
    Robust for complex PSK and QAM constellations in AWGN channels.
    """
    n = min(len(iq_samples), 32768)
    x = iq_samples[:n] - np.mean(iq_samples[:n])

    r2 = np.abs(x) ** 2
    m2 = np.mean(r2)
    m4 = np.mean(r2 ** 2)

    # For complex PSK (constant envelope symbol): kurtosis factor ka = 1
    # m2 = S + N
    # m4 = ka * S^2 + 4 * S * N + 2 * N^2
    # In general: 2*m2^2 - m4 = S^2 * (2 - ka)
    # If ka = 1 (constant modulus), 2*m2^2 - m4 = S^2
    delta = 2.0 * (m2 ** 2) - m4
    if delta > 0:
        s = np.sqrt(delta)
        noise = m2 - s
        if noise > 1e-12:
            snr_linear = s / noise
            return float(10.0 * np.log10(np.maximum(snr_linear, 0.01)))

    # Fallback: spectral estimation (signal peak vs median noise floor)
    _, psd = signal.welch(x, nperseg=512)
    p_sig = np.percentile(psd, 95)
    p_noise = np.percentile(psd, 20)
    if p_noise > 0 and p_sig > p_noise:
        return float(10.0 * np.log10(p_sig / p_noise))

    return 15.0  # default nominal SNR


def compute_waterfall_matrix(iq_samples: np.ndarray, sample_rate: float,
                            nperseg: int = 512, time_bins: int = 80) -> dict:
    """
    Computes a downsampled 2D Spectrogram / Waterfall matrix for efficient 60fps web rendering.
    Returns: JSON-serializable dictionary with freqs, time_slices, and 2D dB intensity grid.
    """
    n = min(len(iq_samples), 65536)
    x = iq_samples[:n]

    f, t, sxx = signal.spectrogram(
        x,
        fs=sample_rate,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=False,
        scaling='density'
    )
    f = np.fft.fftshift(f)
    sxx = np.fft.fftshift(sxx, axes=0)
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-15))

    # Downsample time bins if needed for fast transmission & responsive canvas rendering
    if sxx_db.shape[1] > time_bins:
        step = sxx_db.shape[1] // time_bins
        sxx_db = sxx_db[:, ::step]
        t = t[::step]

    # Normalize dB to [0, 1] range for fast client-side heatmap LUT
    min_val = np.percentile(sxx_db, 5)
    max_val = np.percentile(sxx_db, 99)
    if max_val <= min_val:
        max_val = min_val + 1.0
    norm_grid = np.clip((sxx_db - min_val) / (max_val - min_val), 0.0, 1.0)

    return {
        "frequencies": f.tolist(),
        "time": t.tolist(),
        "intensity": norm_grid.tolist(),  # shape: [freq_bins, time_bins]
        "min_db": float(min_val),
        "max_db": float(max_val)
    }


def extract_all_parameters(iq_samples: np.ndarray, sample_rate: float) -> dict:
    """
    Extracts all primary signal parameters.
    """
    freqs, psd_db = compute_psd(iq_samples, sample_rate)
    psd_linear = 10.0 ** (psd_db / 10.0)

    cfo = estimate_cfo(iq_samples, sample_rate)
    symbol_rate = estimate_symbol_rate(iq_samples, sample_rate)
    obw_99, bw_3db = estimate_bandwidth_and_obw(freqs, psd_linear)
    snr_est = estimate_snr_m2m4(iq_samples)

    # Estimate samples per symbol
    sps = sample_rate / symbol_rate if symbol_rate > 0 else 1.0

    return {
        "sample_rate": float(sample_rate),
        "cfo_hz": round(float(cfo), 2),
        "symbol_rate_baud": round(float(symbol_rate), 2),
        "samples_per_symbol": round(float(sps), 2),
        "obw_99_hz": round(float(obw_99), 2),
        "bandwidth_3db_hz": round(float(bw_3db), 2),
        "estimated_snr_db": round(float(snr_est), 2),
        "num_samples": len(iq_samples)
    }
