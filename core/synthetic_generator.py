"""
Synthetic RF Signal Generator
Generates realistic IQ signals with configurable:
- Modulation: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK, CW, Noise
- Over-the-air impairments: AWGN noise (SNR dB), Carrier Frequency Offset (CFO), Phase Noise
- Pulse shaping: Root Raised Cosine (RRC) or Rectangular
- Preambles: CCSDS ASM (0x1ACFFC1D), Barker-11, Barker-13, AX.25 (0x7E)
- Interleaving & FEC encoding integration
"""

import numpy as np
from scipy import signal
from .ingestion import SignalData

# Standard Synchronization Preambles
PREAMBLES = {
    "CCSDS": np.array([0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1], dtype=np.uint8), # 0x1ACFFC1D
    "BARKER_13": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8),
    "BARKER_11": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
    "AX25": np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8) # 0x7E
}


def create_rrc_filter(sps: int, span: int = 8, alpha: float = 0.35) -> np.ndarray:
    """Generates Root-Raised-Cosine (RRC) filter impulse response."""
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.zeros(len(t), dtype=np.float32)

    for i, ti in enumerate(t):
        if ti == 0.0:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and np.isclose(np.abs(ti), 1 / (4 * alpha)):
            h[i] = (alpha / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))) +
                ((1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
            )
        else:
            numer = np.sin(np.pi * ti * (1 - alpha)) + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
            denom = np.pi * ti * (1 - (4 * alpha * ti) ** 2)
            h[i] = numer / denom

    return h / np.sqrt(np.sum(h**2))


def map_bits_to_symbols(bits: np.ndarray, mod_type: str) -> np.ndarray:
    """Maps binary array to complex constellation symbols."""
    mod = mod_type.upper().replace("-", "")

    if mod == "BPSK":
        # 1 bit/sym: 0 -> -1, 1 -> +1
        return (2 * bits.astype(np.float32) - 1.0) + 0j

    elif mod == "QPSK":
        # 2 bits/sym: Gray coded QPSK
        pad_len = (2 - (len(bits) % 2)) % 2
        if pad_len:
            bits = np.pad(bits, (0, pad_len))
        b = bits.reshape(-1, 2)
        # Gray mapping: 00 -> (-1-j), 01 -> (-1+j), 11 -> (1+j), 10 -> (1-j)
        i = np.where(b[:, 0] == 0, -1.0, 1.0)
        q = np.where(b[:, 1] == 0, -1.0, 1.0)
        return (i + 1j * q) / np.sqrt(2.0)

    elif mod == "8PSK":
        # 3 bits/sym
        pad_len = (3 - (len(bits) % 3)) % 3
        if pad_len:
            bits = np.pad(bits, (0, pad_len))
        b = bits.reshape(-1, 3)
        int_val = b[:, 0] * 4 + b[:, 1] * 2 + b[:, 2]
        angles = 2 * np.pi * int_val / 8.0
        return np.exp(1j * angles)

    elif mod == "16QAM":
        # 4 bits/sym
        pad_len = (4 - (len(bits) % 4)) % 4
        if pad_len:
            bits = np.pad(bits, (0, pad_len))
        b = bits.reshape(-1, 4)
        map_table = {
            (0,0): -3.0,
            (0,1): -1.0,
            (1,1): 1.0,
            (1,0): 3.0
        }
        i_sym = np.array([map_table[(row[0], row[1])] for row in b])
        q_sym = np.array([map_table[(row[2], row[3])] for row in b])
        return (i_sym + 1j * q_sym) / np.sqrt(10.0)

    elif mod == "64QAM":
        # 6 bits/sym
        pad_len = (6 - (len(bits) % 6)) % 6
        if pad_len:
            bits = np.pad(bits, (0, pad_len))
        b = bits.reshape(-1, 6)
        map3 = {
            (0,0,0): -7.0, (0,0,1): -5.0, (0,1,1): -3.0, (0,1,0): -1.0,
            (1,1,0): 1.0, (1,1,1): 3.0, (1,0,1): 5.0, (1,0,0): 7.0
        }
        i_sym = np.array([map3[(r[0], r[1], r[2])] for r in b])
        q_sym = np.array([map3[(r[3], r[4], r[5])] for r in b])
        return (i_sym + 1j * q_sym) / np.sqrt(42.0)

    else:
        raise ValueError(f"Unsupported linear modulation for symbol mapping: {mod_type}")


def generate_fsk_signal(bits: np.ndarray, mod_type: str, sample_rate: float,
                       symbol_rate: float, freq_dev: float = 25000.0) -> np.ndarray:
    """Generates continuous-phase FSK (2FSK or 4FSK) signal."""
    mod = mod_type.upper().replace("-", "")
    sps = int(round(sample_rate / symbol_rate))

    if mod == "2FSK":
        # 1 bit per symbol: 0 -> -freq_dev, 1 -> +freq_dev
        freqs = np.where(bits == 1, freq_dev, -freq_dev)
    elif mod == "4FSK":
        pad = (2 - (len(bits) % 2)) % 2
        if pad:
            bits = np.pad(bits, (0, pad))
        b = bits.reshape(-1, 2)
        # 4 levels: -3, -1, +1, +3
        levels = (2 * b[:, 0] + b[:, 1]).astype(np.float32)
        # map 0,1,2,3 to -3, -1, 1, 3
        mult = 2 * levels - 3.0
        freqs = mult * (freq_dev / 3.0)
    else:
        raise ValueError(f"Unsupported FSK type: {mod_type}")

    # Upsample frequency deviations per sample
    freq_samples = np.repeat(freqs, sps)
    # Continuous phase integration
    phase = 2 * np.pi * np.cumsum(freq_samples) / sample_rate
    iq = np.exp(1j * phase).astype(np.complex64)
    return iq


def generate_synthetic_rf(
    mod_type: str = "QPSK",
    num_symbols: int = 4000,
    sample_rate: float = 1e6,
    symbol_rate: float = 1e5,
    snr_db: float = 20.0,
    cfo_hz: float = 2500.0,
    preamble: str = "CCSDS",
    rrc_alpha: float = 0.35,
    random_seed: int = 42
) -> tuple[SignalData, np.ndarray, np.ndarray]:
    """
    Generates realistic RF signal with specified modulation and channel effects.
    Returns: (SignalData, raw_bits, transmitted_symbols)
    """
    rng = np.random.default_rng(random_seed)
    sps = max(2, int(round(sample_rate / symbol_rate)))

    # Determine bits per symbol
    mod_upper = mod_type.upper().replace("-", "")
    bps_map = {"BPSK": 1, "2FSK": 1, "QPSK": 2, "4FSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6, "CW": 0, "NOISE": 0}
    bps = bps_map.get(mod_upper, 2)

    if mod_upper == "NOISE":
        noise = (rng.normal(0, 1, num_symbols * sps) + 1j * rng.normal(0, 1, num_symbols * sps)).astype(np.complex64)
        sig = SignalData(noise / np.sqrt(2.0), sample_rate, 0.0, "synthetic_noise")
        return sig, np.array([], dtype=np.uint8), np.array([], dtype=np.complex64)

    if mod_upper == "CW":
        t = np.arange(num_symbols * sps) / sample_rate
        cw = np.exp(1j * 2 * np.pi * cfo_hz * t).astype(np.complex64)
        sig = SignalData(cw, sample_rate, 0.0, "synthetic_cw")
        return sig, np.array([], dtype=np.uint8), np.array([], dtype=np.complex64)

    # Generate Payload Bits
    total_bits = num_symbols * bps
    payload_bits = rng.integers(0, 2, total_bits, dtype=np.uint8)

    # Prepend Preamble if selected
    if preamble in PREAMBLES:
        pre = PREAMBLES[preamble]
        bits = np.concatenate([pre, payload_bits])
    else:
        bits = payload_bits

    # Modulation
    if "FSK" in mod_upper:
        iq_tx = generate_fsk_signal(bits, mod_upper, sample_rate, symbol_rate, freq_dev=symbol_rate * 0.5)
        symbols = np.array([])
    else:
        symbols = map_bits_to_symbols(bits, mod_upper)
        # Upsample symbols by SPS
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex64)
        upsampled[::sps] = symbols

        # Pulse shaping filter
        if rrc_alpha > 0:
            h_rrc = create_rrc_filter(sps, span=8, alpha=rrc_alpha)
            iq_tx = signal.convolve(upsampled, h_rrc, mode='same').astype(np.complex64)
        else:
            # Rectangular pulse
            iq_tx = np.repeat(symbols, sps).astype(np.complex64)

    # Impairment 1: Carrier Frequency Offset (CFO)
    t = np.arange(len(iq_tx)) / sample_rate
    iq_cfo = iq_tx * np.exp(1j * 2 * np.pi * cfo_hz * t)

    # Impairment 2: Additive White Gaussian Noise (AWGN)
    if snr_db < 100:
        sig_pwr = np.mean(np.abs(iq_cfo)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_pwr = sig_pwr / snr_linear
        noise = (rng.normal(0, np.sqrt(noise_pwr / 2.0), len(iq_cfo)) +
                 1j * rng.normal(0, np.sqrt(noise_pwr / 2.0), len(iq_cfo))).astype(np.complex64)
        iq_rx = (iq_cfo + noise).astype(np.complex64)
    else:
        iq_rx = iq_cfo.astype(np.complex64)

    signal_data = SignalData(
        iq_samples=iq_rx,
        sample_rate=sample_rate,
        center_freq=cfo_hz,
        source_type=f"synthetic_{mod_type}"
    )

    return signal_data, bits, symbols
