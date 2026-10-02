"""
Signal Ingestion and Preprocessing Engine
Supports:
- .wav files: mono (converted to analytic signal via Hilbert transform if needed) or stereo (Ch0=I, Ch1=Q)
- .iq / .bin / .raw files: complex64, int16 (cs16), int8 (cs8), uint8 (cu8)
- SigMF metadata parsing (.sigmf-meta + .sigmf-data)
- DC offset blocker & AGC normalization
"""

import os
import json
import struct
import wave
import numpy as np
from scipy import signal

class SignalData:
    def __init__(self, iq_samples: np.ndarray, sample_rate: float, center_freq: float = 0.0,
                 source_type: str = "unknown", file_path: str = ""):
        """
        iq_samples: 1D complex numpy array (I + 1j * Q)
        sample_rate: sampling frequency in Hz
        center_freq: center/carrier frequency in Hz
        source_type: 'wav', 'iq_complex64', 'iq_int16', 'iq_uint8', 'synthetic', etc.
        """
        self.samples = np.asarray(iq_samples, dtype=np.complex64)
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.source_type = source_type
        self.file_path = file_path

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate if self.sample_rate > 0 else 0.0

    def get_i(self) -> np.ndarray:
        return np.real(self.samples)

    def get_q(self) -> np.ndarray:
        return np.imag(self.samples)


def read_wav_file(file_path: str, max_samples: int = None) -> SignalData:
    """
    Reads a .wav file.
    If 2 channels: Ch0 = I, Ch1 = Q.
    If 1 channel: Real IF or audio signal; automatically computes analytic complex signal via Hilbert transform.
    Supports 8-bit, 16-bit, 24-bit, and 32-bit PCM or 32-bit float.
    """
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        if max_samples is not None:
            n_frames = min(n_frames, max_samples)

        raw_bytes = wf.readframes(n_frames)

    # Decode bytes based on sample width
    if sampwidth == 1:
        # 8-bit unsigned PCM
        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        # 16-bit signed PCM
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit signed PCM
        raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        triplets = raw_arr.reshape(-1, 3)
        # Sign extend 24-bit to 32-bit int
        int24 = (triplets[:, 0].astype(np.int32) |
                 (triplets[:, 1].astype(np.int32) << 8) |
                 (triplets[:, 2].astype(np.int32) << 16))
        int24 = np.where(int24 & 0x800000, int24 - 0x1000000, int24)
        data = int24.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        # 32-bit signed int or float
        try:
            data = np.frombuffer(raw_bytes, dtype=np.float32)
            if np.max(np.abs(data)) > 10.0:  # likely int32
                data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        except Exception:
            data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels == 2:
        # Stereo I/Q: Ch0 = I, Ch1 = Q
        i_data = data[0::2]
        q_data = data[1::2]
        min_len = min(len(i_data), len(q_data))
        iq = (i_data[:min_len] + 1j * q_data[:min_len]).astype(np.complex64)
    elif n_channels == 1:
        # Mono: compute analytic signal via Hilbert transform
        analytic = signal.hilbert(data)
        iq = analytic.astype(np.complex64)
    else:
        # Multi-channel: take first two as I/Q
        data_matrix = data.reshape(-1, n_channels)
        iq = (data_matrix[:, 0] + 1j * data_matrix[:, 1]).astype(np.complex64)

    return SignalData(
        iq_samples=iq,
        sample_rate=float(sample_rate),
        center_freq=0.0,
        source_type="wav",
        file_path=file_path
    )


def read_sigmf_metadata(meta_path: str) -> dict:
    """Parses SigMF .sigmf-meta JSON file if available."""
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        global_info = meta.get("global", {})
        capture_info = meta.get("captures", [{}])[0] if meta.get("captures") else {}
        return {
            "datatype": global_info.get("core:datatype", "cf32_le"),
            "sample_rate": global_info.get("core:sample_rate", 1e6),
            "center_freq": capture_info.get("core:frequency", 0.0),
            "description": global_info.get("core:description", "")
        }
    except Exception:
        return {}


def read_iq_file(file_path: str, format_type: str = "auto", sample_rate: float = 1e6,
                 center_freq: float = 0.0, max_samples: int = None) -> SignalData:
    """
    Reads raw .iq / .bin / .raw files.
    format_type: 'auto', 'complex64' (float32 I/Q), 'int16' (cs16), 'int8' (cs8), 'uint8' (cu8)
    """
    # Check for accompanying SigMF metadata
    base_no_ext = os.path.splitext(file_path)[0]
    sigmf_meta_path = f"{base_no_ext}.sigmf-meta"
    if os.path.exists(sigmf_meta_path):
        meta = read_sigmf_metadata(sigmf_meta_path)
        if meta:
            sample_rate = meta.get("sample_rate", sample_rate)
            center_freq = meta.get("center_freq", center_freq)
            dtype_str = meta.get("datatype", "")
            if "cf32" in dtype_str:
                format_type = "complex64"
            elif "ci16" in dtype_str:
                format_type = "int16"
            elif "cu8" in dtype_str:
                format_type = "uint8"
            elif "ci8" in dtype_str:
                format_type = "int8"

    file_size = os.path.getsize(file_path)

    # Auto-detect format if set to 'auto'
    if format_type == "auto":
        with open(file_path, 'rb') as f:
            header_bytes = f.read(min(4096, file_size))
        # Heuristic test: try complex64 first
        floats = np.frombuffer(header_bytes, dtype=np.float32)
        if len(floats) > 0 and np.all(np.isfinite(floats)) and np.all(np.abs(floats) < 50.0):
            format_type = "complex64"
        else:
            format_type = "int16"

    # Read binary data
    if format_type == "complex64":
        count = max_samples if max_samples is not None else -1
        samples = np.fromfile(file_path, dtype=np.complex64, count=count)
    elif format_type == "int16":
        count = max_samples * 2 if max_samples is not None else -1
        raw = np.fromfile(file_path, dtype=np.int16, count=count)
        i_data = raw[0::2].astype(np.float32) / 32768.0
        q_data = raw[1::2].astype(np.float32) / 32768.0
        min_len = min(len(i_data), len(q_data))
        samples = (i_data[:min_len] + 1j * q_data[:min_len]).astype(np.complex64)
    elif format_type == "int8":
        count = max_samples * 2 if max_samples is not None else -1
        raw = np.fromfile(file_path, dtype=np.int8, count=count)
        i_data = raw[0::2].astype(np.float32) / 128.0
        q_data = raw[1::2].astype(np.float32) / 128.0
        min_len = min(len(i_data), len(q_data))
        samples = (i_data[:min_len] + 1j * q_data[:min_len]).astype(np.complex64)
    elif format_type == "uint8":
        count = max_samples * 2 if max_samples is not None else -1
        raw = np.fromfile(file_path, dtype=np.uint8, count=count)
        i_data = (raw[0::2].astype(np.float32) - 127.5) / 128.0
        q_data = (raw[1::2].astype(np.float32) - 127.5) / 128.0
        min_len = min(len(i_data), len(q_data))
        samples = (i_data[:min_len] + 1j * q_data[:min_len]).astype(np.complex64)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

    return SignalData(
        iq_samples=samples,
        sample_rate=float(sample_rate),
        center_freq=float(center_freq),
        source_type=f"iq_{format_type}",
        file_path=file_path
    )


def load_signal_file(file_path: str, format_hint: str = "auto", sample_rate: float = 1e6,
                     center_freq: float = 0.0, max_samples: int = None) -> SignalData:
    """Universal loader for .wav and .iq files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        return read_wav_file(file_path, max_samples=max_samples)
    else:
        return read_iq_file(file_path, format_type=format_hint, sample_rate=sample_rate,
                            center_freq=center_freq, max_samples=max_samples)


def remove_dc_offset(iq_samples: np.ndarray) -> np.ndarray:
    """Removes DC carrier leak (mean offset) from I and Q channels."""
    i = np.real(iq_samples) - np.mean(np.real(iq_samples))
    q = np.imag(iq_samples) - np.mean(np.imag(iq_samples))
    return (i + 1j * q).astype(np.complex64)


def normalize_power(iq_samples: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """Normalizes signal power (AGC) to target RMS."""
    rms = np.sqrt(np.mean(np.abs(iq_samples)**2))
    if rms > 1e-12:
        return (iq_samples * (target_rms / rms)).astype(np.complex64)
    return iq_samples
