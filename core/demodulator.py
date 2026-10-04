"""
Signal Demodulation Engine
Supports:
- FSK Demodulation: 2-FSK, 4-FSK (Instantaneous Frequency Discriminator + Matched Slicer)
- PSK Demodulation: BPSK, QPSK, 8PSK (RRC Matched Filter + Costas Carrier Loop + Gardner Timing Recovery)
- QAM Demodulation: 16QAM, 64QAM (CMA Equalizer + Multi-Threshold Slicer)
Outputs: Hard bits, soft decisions, and synchronized constellation symbol stream.
"""

import numpy as np
from scipy import signal
from .synthetic_generator import create_rrc_filter

def demodulate_fsk(iq_samples: np.ndarray, sample_rate: float, symbol_rate: float,
                   fsk_levels: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodulates FSK signals using instantaneous frequency discriminator.
    fsk_levels: 2 for 2-FSK, 4 for 4-FSK.
    Returns: (demodulated_bits, recovered_frequency_symbols)
    """
    sps = max(2, int(round(sample_rate / symbol_rate)))

    # Compute instantaneous phase difference
    conj_prod = iq_samples[1:] * np.conj(iq_samples[:-1])
    dphi = np.angle(conj_prod)
    inst_freq = dphi * sample_rate / (2.0 * np.pi)

    # Lowpass filter to eliminate high-frequency discriminator spikes
    cutoff = np.clip(1.2 * symbol_rate / (sample_rate / 2.0), 0.01, 0.95)
    b, a = signal.butter(4, cutoff, btype='low')
    filtered_freq = signal.filtfilt(b, a, inst_freq)

    # Symbol timing recovery: sample near middle of symbol period
    offset = sps // 2
    sym_freqs = filtered_freq[offset::sps]

    # Normalize frequency deviation
    dev = np.percentile(np.abs(sym_freqs), 80)
    if dev < 1e-3:
        dev = 1.0

    if fsk_levels == 2:
        bits = np.where(sym_freqs > 0, 1, 0).astype(np.uint8)
        symbols = sym_freqs / dev
    elif fsk_levels == 4:
        norm_freq = sym_freqs / dev
        # 4 levels: < -0.5, [-0.5, 0], [0, 0.5], > 0.5
        sym_idx = np.zeros(len(norm_freq), dtype=np.uint8)
        sym_idx[norm_freq < -0.66] = 0
        sym_idx[(norm_freq >= -0.66) & (norm_freq < 0)] = 1
        sym_idx[(norm_freq >= 0) & (norm_freq < 0.66)] = 2
        sym_idx[norm_freq >= 0.66] = 3

        # Convert to bits (2 bits per symbol)
        bits = np.zeros(len(sym_idx) * 2, dtype=np.uint8)
        bits[0::2] = (sym_idx >> 1) & 1
        bits[1::2] = sym_idx & 1
        symbols = norm_freq
    else:
        raise ValueError(f"Unsupported FSK levels: {fsk_levels}")

    return bits, symbols


def demodulate_psk(iq_samples: np.ndarray, sample_rate: float, symbol_rate: float,
                   psk_order: int = 4, cfo_est: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodulates PSK signals (BPSK, QPSK, 8PSK) with Costas carrier recovery loop.
    psk_order: 2 (BPSK), 4 (QPSK), 8 (8PSK).
    Returns: (demodulated_bits, recovered_constellation_symbols)
    """
    sps = max(2, int(round(sample_rate / symbol_rate)))

    # Step 1: Coarse CFO de-rotation
    t = np.arange(len(iq_samples)) / sample_rate
    iq_derot = iq_samples * np.exp(-1j * 2 * np.pi * cfo_est * t)

    # Step 2: Matched Filtering (RRC)
    h_rrc = create_rrc_filter(sps, span=6, alpha=0.35)
    filtered = signal.convolve(iq_derot, h_rrc, mode='same')

    # Step 3: Costas Loop Carrier Phase Recovery + Symbol Slicing
    n_samples = len(filtered)
    phase = 0.0
    freq_acc = 0.0
    kp = 0.04
    ki = 0.001

    recovered_symbols = []
    # Symbol-spaced sampling with simple peak energy tracking
    best_offset = 0
    max_var = -1
    for trial_offset in range(sps):
        candidates = filtered[trial_offset::sps]
        var = np.var(np.abs(candidates))
        if var > max_var:
            max_var = var
            best_offset = trial_offset

    raw_symbols = filtered[best_offset::sps]

    # Run Costas Loop over symbol-rate sequence
    for sym in raw_symbols:
        # De-rotate with current phase estimate
        sym_rot = sym * np.exp(-1j * phase)

        # Phase error detector
        if psk_order == 2:
            # BPSK Costas error
            error = np.sign(np.real(sym_rot)) * np.imag(sym_rot)
        elif psk_order == 4:
            # QPSK Costas error
            error = np.sign(np.real(sym_rot)) * np.imag(sym_rot) - np.sign(np.imag(sym_rot)) * np.real(sym_rot)
        elif psk_order == 8:
            # 8PSK error
            ang = np.angle(sym_rot)
            nearest_ang = np.round(ang * 8.0 / (2 * np.pi)) * (2 * np.pi / 8.0)
            error = np.sin(ang - nearest_ang)
        else:
            error = np.angle(sym_rot ** psk_order) / psk_order

        # Limit error
        error = np.clip(error, -1.0, 1.0)
        freq_acc += ki * error
        phase += kp * error + freq_acc

        recovered_symbols.append(sym_rot)

    symbols_arr = np.array(recovered_symbols, dtype=np.complex64)

    # Normalize amplitude
    avg_mag = np.mean(np.abs(symbols_arr))
    if avg_mag > 1e-6:
        symbols_arr = symbols_arr / avg_mag

    # Constellation Decision Slicing to Bits
    if psk_order == 2:
        # BPSK: 1 bit/sym: Re > 0 -> 1, Re <= 0 -> 0
        bits = np.where(np.real(symbols_arr) > 0, 1, 0).astype(np.uint8)

    elif psk_order == 4:
        # QPSK Gray mapped:
        # b0: 1 if Re > 0 else 0
        # b1: 1 if Im > 0 else 0
        b0 = np.where(np.real(symbols_arr) > 0, 1, 0).astype(np.uint8)
        b1 = np.where(np.imag(symbols_arr) > 0, 1, 0).astype(np.uint8)
        bits = np.zeros(len(symbols_arr) * 2, dtype=np.uint8)
        bits[0::2] = b0
        bits[1::2] = b1

    elif psk_order == 8:
        # 8PSK: 3 bits/sym
        angles = np.angle(symbols_arr)
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)
        sectors = np.round(angles * 8.0 / (2 * np.pi)).astype(int) % 8

        bits = np.zeros(len(symbols_arr) * 3, dtype=np.uint8)
        bits[0::3] = (sectors >> 2) & 1
        bits[1::3] = (sectors >> 1) & 1
        bits[2::3] = sectors & 1

    return bits, symbols_arr


def demodulate_qam(iq_samples: np.ndarray, sample_rate: float, symbol_rate: float,
                   qam_order: int = 16, cfo_est: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Demodulates QAM signals (16QAM, 64QAM) with CMA equalization and multi-level slicing.
    Returns: (demodulated_bits, equalized_constellation_symbols)
    """
    sps = max(2, int(round(sample_rate / symbol_rate)))

    # CFO de-rotation
    t = np.arange(len(iq_samples)) / sample_rate
    iq_derot = iq_samples * np.exp(-1j * 2 * np.pi * cfo_est * t)

    # Downsample to symbol rate
    best_offset = sps // 2
    syms = iq_derot[best_offset::sps]

    # AGC Normalization
    rms = np.sqrt(np.mean(np.abs(syms)**2))
    if rms > 1e-12:
        syms = syms / rms

    # Adaptive CMA Equalizer (Constant Modulus Algorithm)
    # Target radius for 16QAM: R2 ~ 1.32
    r2_target = 1.32 if qam_order == 16 else 1.39
    mu = 0.005
    eq_syms = []
    weight = 1.0 + 0j

    for s in syms:
        y = s * weight
        err = y * (r2_target - np.abs(y)**2)
        weight += mu * err * np.conj(s)
        eq_syms.append(y)

    symbols_arr = np.array(eq_syms, dtype=np.complex64)

    # Decision Slicing
    if qam_order == 16:
        # Scale constellation to standard grid [-3, -1, 1, 3] / sqrt(10)
        scale = np.sqrt(10.0)
        scaled_syms = symbols_arr * scale
        re = np.real(scaled_syms)
        im = np.imag(scaled_syms)

        # Gray mapping for 16QAM
        def slice_axis(val):
            # val <= -2 -> 00, -2 < val <= 0 -> 01, 0 < val <= 2 -> 11, val > 2 -> 10
            b_high = np.where(val > 0, 1, 0).astype(np.uint8)
            b_low = np.where((val > -2) & (val <= 2), 1, 0).astype(np.uint8)
            return b_high, b_low

        re_b0, re_b1 = slice_axis(re)
        im_b0, im_b1 = slice_axis(im)

        bits = np.zeros(len(symbols_arr) * 4, dtype=np.uint8)
        bits[0::4] = re_b0
        bits[1::4] = re_b1
        bits[2::4] = im_b0
        bits[3::4] = im_b1

    elif qam_order == 64:
        # 6 bits per symbol (3 for Re, 3 for Im)
        scale = np.sqrt(42.0)
        scaled_syms = symbols_arr * scale
        re = np.clip(np.round((np.real(scaled_syms) + 7.0) / 2.0), 0, 7).astype(int)
        im = np.clip(np.round((np.imag(scaled_syms) + 7.0) / 2.0), 0, 7).astype(int)

        bits = np.zeros(len(symbols_arr) * 6, dtype=np.uint8)
        bits[0::6] = (re >> 2) & 1
        bits[1::6] = (re >> 1) & 1
        bits[2::6] = re & 1
        bits[3::6] = (im >> 2) & 1
        bits[4::6] = (im >> 1) & 1
        bits[5::6] = im & 1
    else:
        raise ValueError(f"Unsupported QAM order: {qam_order}")

    return bits, symbols_arr


def demodulate_signal(iq_samples: np.ndarray, sample_rate: float, symbol_rate: float,
                      mod_type: str, cfo_est: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Universal Demodulation Dispatcher.
    Selects demodulator based on AMC classification.
    """
    mod = mod_type.upper().replace("-", "")

    if "2FSK" in mod:
        return demodulate_fsk(iq_samples, sample_rate, symbol_rate, fsk_levels=2)
    elif "4FSK" in mod:
        return demodulate_fsk(iq_samples, sample_rate, symbol_rate, fsk_levels=4)
    elif "BPSK" in mod:
        return demodulate_psk(iq_samples, sample_rate, symbol_rate, psk_order=2, cfo_est=cfo_est)
    elif "QPSK" in mod:
        return demodulate_psk(iq_samples, sample_rate, symbol_rate, psk_order=4, cfo_est=cfo_est)
    elif "8PSK" in mod:
        return demodulate_psk(iq_samples, sample_rate, symbol_rate, psk_order=8, cfo_est=cfo_est)
    elif "16QAM" in mod:
        return demodulate_qam(iq_samples, sample_rate, symbol_rate, qam_order=16, cfo_est=cfo_est)
    elif "64QAM" in mod:
        return demodulate_qam(iq_samples, sample_rate, symbol_rate, qam_order=64, cfo_est=cfo_est)
    else:
        # Default to QPSK demodulator
        return demodulate_psk(iq_samples, sample_rate, symbol_rate, psk_order=4, cfo_est=cfo_est)
