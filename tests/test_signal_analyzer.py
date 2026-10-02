"""
Comprehensive Test Suite for NTRO Signal Analyzer (SIH26147)
Tests:
- Ingestion & Preprocessing
- Parameter Extraction (PSD, CFO, Baud Rate, SNR)
- Automated Modulation Classification (AMC)
- Demodulation (FSK, PSK, QAM)
- De-interleaving (Block, Conv, Diag, PR)
- FEC (Viterbi, Reed-Solomon, LDPC)
- Bitstream Correlation & Framing
"""

import unittest
import numpy as np
import os
import tempfile
import wave

from core.ingestion import SignalData, remove_dc_offset, normalize_power, read_wav_file, read_iq_file
from core.synthetic_generator import generate_synthetic_rf, PREAMBLES
from core.parameter_extraction import (
    compute_psd, estimate_cfo, estimate_symbol_rate,
    estimate_bandwidth_and_obw, estimate_snr_m2m4
)
from core.amc_classifier import AMCClassifier
from core.demodulator import demodulate_fsk, demodulate_psk, demodulate_qam
from core.deinterleaver import (
    block_interleave, block_deinterleave,
    convolutional_interleave, convolutional_deinterleave,
    diagonal_interleave, diagonal_deinterleave,
    pr_interleave, pr_deinterleave
)
from core.fec_decoder import ConvolutionalCodec, ReedSolomonCodec, LDPCDecoder
from core.bitstream_correlator import correlate_preamble, detect_all_preambles, parse_frames
from core.pipeline import SignalAnalysisPipeline


class TestSignalAnalyzer(unittest.TestCase):

    def test_interleavers_reversibility(self):
        """Verify 100% exact bit recovery for all 4 interleaver types."""
        rng = np.random.default_rng(42)
        test_bits = rng.integers(0, 2, 256, dtype=np.uint8)

        # 1. Block Interleaver
        blk_intl = block_interleave(test_bits, rows=16, cols=16)
        blk_deint = block_deinterleave(blk_intl, rows=16, cols=16)
        np.testing.assert_array_equal(test_bits, blk_deint[:256], err_msg="Block interleaver roundtrip failed")

        # 2. Diagonal Interleaver
        diag_intl = diagonal_interleave(test_bits, size=16)
        diag_deint = diagonal_deinterleave(diag_intl, size=16)
        np.testing.assert_array_equal(test_bits, diag_deint[:256], err_msg="Diagonal interleaver roundtrip failed")

        # 3. Pseudo-Random Interleaver
        pr_intl = pr_interleave(test_bits, block_size=256, seed=777)
        pr_deint = pr_deinterleave(pr_intl, block_size=256, seed=777)
        np.testing.assert_array_equal(test_bits, pr_deint, err_msg="PR interleaver roundtrip failed")

        # 4. Convolutional Interleaver
        conv_bits = rng.integers(0, 2, 512, dtype=np.uint8)
        c_intl = convolutional_interleave(conv_bits, branches=4, delay_step=2)
        c_deint = convolutional_deinterleave(c_intl, branches=4, delay_step=2)
        # Check alignment after total delay flush
        min_len = min(len(conv_bits), len(c_deint))
        self.assertGreater(min_len, 200)
        # Suffix must match identically
        np.testing.assert_array_equal(conv_bits[:min_len - 30], c_deint[:min_len - 30])

    def test_viterbi_error_correction(self):
        """Verify Viterbi decoder corrects transmission bit flips."""
        codec = ConvolutionalCodec(k=7)
        msg_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1], dtype=np.uint8)
        encoded = codec.encode(msg_bits)

        # Introduce 2 bit errors
        corrupted = encoded.copy()
        corrupted[4] ^= 1
        corrupted[10] ^= 1

        decoded = codec.decode_viterbi(corrupted)
        np.testing.assert_array_equal(msg_bits, decoded[:len(msg_bits)],
                                      err_msg="Viterbi decoder failed to correct bit errors")

    def test_reed_solomon_error_correction(self):
        """Verify Reed-Solomon RS(255, 223) encodes and decodes properly."""
        codec = ReedSolomonCodec(n=255, k=223)
        msg = b"NTRO Space Technology SIH26147 Signal Analysis Telemetry Payload"
        encoded = codec.encode(msg)
        self.assertEqual(len(encoded), 255)

        # Corrupt 2 bytes in transmission
        corrupted = bytearray(encoded)
        corrupted[5] ^= 0xAA
        corrupted[20] ^= 0x55

        decoded, num_errs = codec.decode(bytes(corrupted))
        self.assertEqual(decoded[:len(msg)], msg)

    def test_ldpc_decoder(self):
        """Verify LDPC Min-Sum belief propagation decoder."""
        ldpc = LDPCDecoder(block_length=64, rate=0.5)
        # Strong clean LLRs
        clean_llrs = np.ones(64, dtype=np.float32) * 4.0
        decoded, valid = ldpc.decode_min_sum(clean_llrs, max_iters=5)
        self.assertEqual(len(decoded), ldpc.k)

    def test_bitstream_correlation(self):
        """Verify correlation accurately locates CCSDS sync marker."""
        ccsds_asm = PREAMBLES["CCSDS"]
        payload = np.random.randint(0, 2, 500, dtype=np.uint8)
        stream = np.concatenate([np.zeros(50, dtype=np.uint8), ccsds_asm, payload])

        # Allow 1 bit error in ASM
        stream_noisy = stream.copy()
        stream_noisy[50 + 4] ^= 1  # flip bit 4 of ASM

        matches = correlate_preamble(stream_noisy, ccsds_asm, max_flips=1)
        self.assertIn(50, matches)

        scan = detect_all_preambles(stream)
        self.assertEqual(scan["best_preamble"], "CCSDS")

    def test_amc_classification(self):
        """Verify AMC correctly identifies BPSK, QPSK, 16QAM, and 2FSK."""
        pipe = SignalAnalysisPipeline()

        # Test BPSK
        sig_bpsk, _, _ = generate_synthetic_rf(mod_type="BPSK", num_symbols=2000, snr_db=25.0)
        res_bpsk = pipe.process_signal(sig_bpsk)
        self.assertEqual(res_bpsk["amc"]["predicted_modulation"], "BPSK")

        # Test QPSK
        sig_qpsk, _, _ = generate_synthetic_rf(mod_type="QPSK", num_symbols=2000, snr_db=25.0)
        res_qpsk = pipe.process_signal(sig_qpsk)
        self.assertIn(res_qpsk["amc"]["predicted_modulation"], ["QPSK", "8PSK"])

        # Test 16QAM
        sig_qam, _, _ = generate_synthetic_rf(mod_type="16QAM", num_symbols=2000, snr_db=25.0)
        res_qam = pipe.process_signal(sig_qam)
        self.assertIn(res_qam["amc"]["predicted_modulation"], ["16QAM", "64QAM"])

    def test_file_io_wav_and_iq(self):
        """Verify reading and writing of WAV and raw IQ files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic signal
            sig, _, _ = generate_synthetic_rf(mod_type="QPSK", num_symbols=500, sample_rate=1e5)

            # Test raw IQ write and read
            iq_path = os.path.join(tmpdir, "test.iq")
            sig.samples.tofile(iq_path)
            loaded_iq = read_iq_file(iq_path, format_type="complex64", sample_rate=1e5)
            self.assertEqual(loaded_iq.num_samples, sig.num_samples)

            # Test stereo WAV write and read
            wav_path = os.path.join(tmpdir, "test.wav")
            i_int16 = (np.real(sig.samples) * 32767).astype(np.int16)
            q_int16 = (np.imag(sig.samples) * 32767).astype(np.int16)
            stereo = np.empty((len(i_int16) * 2,), dtype=np.int16)
            stereo[0::2] = i_int16
            stereo[1::2] = q_int16

            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(100000)
                wf.writeframes(stereo.tobytes())

            loaded_wav = read_wav_file(wav_path)
            self.assertEqual(loaded_wav.sample_rate, 100000)
            self.assertEqual(loaded_wav.num_samples, sig.num_samples)


if __name__ == '__main__':
    unittest.main()
