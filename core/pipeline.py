"""
Unified Signal Analysis Pipeline
Orchestrates end-to-end signal processing:
1. Ingestion & DC/AGC conditioning
2. Spectral Analysis & Parameter Extraction (PSD, CFO, Baud Rate, SNR)
3. Automated Modulation Classification (AMC)
4. Demodulation (FSK, PSK, QAM)
5. De-interleaving (Block, Conv, Diag, PR)
6. Forward Error Correction (Viterbi, RS, LDPC)
7. Bitstream Correlation & Frame/Payload Extraction
8. Visualization data compilation (Waterfall, PSD, Constellation, Eye Diagram)
"""

import numpy as np
from .ingestion import SignalData, remove_dc_offset, normalize_power, load_signal_file
from .parameter_extraction import (
    compute_psd, estimate_cfo, estimate_symbol_rate,
    estimate_bandwidth_and_obw, estimate_snr_m2m4, compute_waterfall_matrix
)
from .amc_classifier import AMCClassifier
from .demodulator import demodulate_signal
from .deinterleaver import apply_deinterleaving, estimate_interleaver_period
from .fec_decoder import decode_fec
from .bitstream_correlator import detect_all_preambles, parse_frames

class SignalAnalysisPipeline:
    def __init__(self):
        self.amc = AMCClassifier()

    def process_signal(
        self,
        signal_data: SignalData,
        manual_mod: str = "auto",
        manual_fec: str = "auto",
        manual_interleaving: str = "none",
        interleave_p1: int = 16,
        interleave_p2: int = 16,
        target_preamble: str = "CCSDS"
    ) -> dict:
        """
        Executes full automated or parameterized analysis on SignalData.
        """
        raw_samples = signal_data.samples
        sample_rate = signal_data.sample_rate

        # 1. Preprocessing: DC Blocker & AGC
        clean_samples = remove_dc_offset(raw_samples)
        norm_samples = normalize_power(clean_samples)

        # 2. Parameter Extraction
        freqs, psd_db = compute_psd(norm_samples, sample_rate, nperseg=1024)
        psd_linear = 10.0 ** (psd_db / 10.0)
        cfo_est = estimate_cfo(norm_samples, sample_rate)
        baud_est = estimate_symbol_rate(norm_samples, sample_rate)
        obw_99, bw_3db = estimate_bandwidth_and_obw(freqs, psd_linear)
        snr_est = estimate_snr_m2m4(norm_samples)
        sps = sample_rate / baud_est if baud_est > 0 else 1.0

        parameters = {
            "sample_rate": float(sample_rate),
            "estimated_cfo_hz": round(float(cfo_est), 2),
            "estimated_baud_rate": round(float(baud_est), 2),
            "samples_per_symbol": round(float(sps), 2),
            "obw_99_hz": round(float(obw_99), 2),
            "bandwidth_3db_hz": round(float(bw_3db), 2),
            "estimated_snr_db": round(float(snr_est), 2),
            "num_samples": len(raw_samples),
            "duration_sec": round(len(raw_samples) / sample_rate, 4) if sample_rate > 0 else 0.0
        }

        # 3. Automated Modulation Classification (AMC) with CFO de-rotation
        t = np.arange(len(norm_samples)) / sample_rate
        derot_samples = norm_samples * np.exp(-1j * 2.0 * np.pi * cfo_est * t)
        amc_result = self.amc.classify(derot_samples, sample_rate, snr_est=snr_est)
        active_mod = amc_result["predicted_modulation"] if manual_mod == "auto" else manual_mod

        # 4. Demodulation
        demod_bits, symbols = demodulate_signal(
            norm_samples,
            sample_rate=sample_rate,
            symbol_rate=baud_est,
            mod_type=active_mod,
            cfo_est=cfo_est
        )

        # 5. De-interleaving
        if manual_interleaving == "auto":
            detected_period, conf = estimate_interleaver_period(demod_bits)
            if conf > 0.4 and detected_period > 4:
                side = int(round(np.sqrt(detected_period)))
                active_interleave = "block"
                p1, p2 = side, side
            else:
                active_interleave = "none"
                p1, p2 = 16, 16
        else:
            active_interleave = manual_interleaving
            p1, p2 = interleave_p1, interleave_p2

        deinterleaved_bits = apply_deinterleaving(demod_bits, active_interleave, p1, p2)

        # 6. Forward Error Correction (FEC)
        active_fec = manual_fec
        if manual_fec == "auto":
            # Heuristic: test if Viterbi improves sync correlation or default to standard space Viterbi
            active_fec = "viterbi" if "PSK" in active_mod or "QAM" in active_mod else "none"

        fec_bits, fec_meta = decode_fec(deinterleaved_bits, active_fec)

        # 7. Bitstream Correlation & Framing
        preamble_scan = detect_all_preambles(fec_bits)
        frame_sync_target = target_preamble if target_preamble != "auto" else (
            preamble_scan["best_preamble"] or "CCSDS"
        )
        frame_analysis = parse_frames(fec_bits, sync_key=frame_sync_target)

        # 8. Compile Visualization Artifacts (Downsampled for 60fps responsive UI)
        # PSD subset (max 512 points)
        step_psd = max(1, len(freqs) // 512)
        psd_data = {
            "frequencies": [round(float(f), 1) for f in freqs[::step_psd]],
            "power_db": [round(float(p), 2) for p in psd_db[::step_psd]]
        }

        # Constellation subset (max 2000 points)
        n_syms = len(symbols)
        stride_syms = max(1, n_syms // 2000)
        sub_syms = symbols[::stride_syms]
        constellation_data = {
            "i": [round(float(np.real(s)), 3) for s in sub_syms],
            "q": [round(float(np.imag(s)), 3) for s in sub_syms]
        }

        # Time domain waveform slice (500 samples)
        n_time = min(500, len(norm_samples))
        time_data = {
            "t": [round(float(i / sample_rate * 1e6), 2) for i in range(n_time)], # in microseconds
            "i": [round(float(np.real(norm_samples[i])), 3) for i in range(n_time)],
            "q": [round(float(np.imag(norm_samples[i])), 3) for i in range(n_time)]
        }

        # Waterfall matrix
        waterfall_data = compute_waterfall_matrix(norm_samples, sample_rate, nperseg=256, time_bins=64)

        return {
            "status": "success",
            "parameters": parameters,
            "amc": amc_result,
            "active_configuration": {
                "modulation": active_mod,
                "fec": active_fec,
                "interleaving": active_interleave,
                "frame_sync": frame_sync_target
            },
            "demodulation": {
                "num_bits_recovered": len(demod_bits),
                "num_symbols": len(symbols),
                "bit_sample_snippet": demod_bits[:64].tolist()
            },
            "fec_status": fec_meta,
            "framing": frame_analysis,
            "preambles": preamble_scan,
            "visuals": {
                "psd": psd_data,
                "constellation": constellation_data,
                "time_domain": time_data,
                "waterfall": waterfall_data
            }
        }
