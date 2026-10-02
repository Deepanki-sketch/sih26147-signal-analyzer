"""
NTRO Automated Signal Analyzer - Command Line Interface (SIH26147)
Batch headless processing, automated parameter extraction, AMC classification,
demodulation, de-interleaving, FEC decoding, and framing correlation.

Usage:
  python ntro_cli.py --generate QPSK --snr 22 --cfo 2500 --export report.json
  python ntro_cli.py --input sample.iq --rate 1000000 --format complex64
  python ntro_cli.py --input recording.wav --auto
"""

import os
import sys
import argparse
import json
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.ingestion import load_signal_file
from core.synthetic_generator import generate_synthetic_rf
from core.pipeline import SignalAnalysisPipeline

BANNER = r"""
================================================================================
       NATIONAL TECHNICAL RESEARCH ORGANISATION (NTRO) - SIH26147
       AUTOMATED .IQ & .WAV SIGNAL ANALYZER & PARAMETER EXTRACTOR
================================================================================
"""

def print_banner():
    print(BANNER)

def main():
    parser = argparse.ArgumentParser(
        description="NTRO SIH26147 - Automated Signal Parameter Extractor & AMC Demodulator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", type=str, help="Path to .iq, .wav, .bin or .raw file")
    group.add_argument("--generate", "-g", type=str,
                       choices=["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK", "CW", "NOISE"],
                       help="Generate synthetic RF signal for benchmark testing")

    parser.add_argument("--rate", "-r", type=float, default=1e6, help="Sampling rate in Hz (default: 1,000,000)")
    parser.add_argument("--format", "-f", type=str, default="auto",
                        choices=["auto", "complex64", "int16", "uint8", "int8"],
                        help="Binary format hint for raw .IQ files (default: auto)")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR in dB for synthetic generation (default: 20)")
    parser.add_argument("--cfo", type=float, default=2500.0, help="Carrier Frequency Offset in Hz for synthesis (default: 2500)")
    parser.add_argument("--preamble", type=str, default="CCSDS", help="Sync preamble to inject/seek (default: CCSDS)")
    parser.add_argument("--mod-override", type=str, default="auto", help="Manual modulation override or auto")
    parser.add_argument("--fec-override", type=str, default="auto", help="Manual FEC override or auto")
    parser.add_argument("--export", "-e", type=str, help="Export JSON analysis report to path")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output for automated scripts")

    args = parser.parse_args()

    if not args.quiet:
        print_banner()

    pipe = SignalAnalysisPipeline()

    if args.generate:
        if not args.quiet:
            print(f"[*] Synthesizing {args.generate} signal (SNR: {args.snr} dB, CFO: {args.cfo} Hz, Fs: {args.rate/1e6:.2f} MHz)...")
        sig, _, _ = generate_synthetic_rf(
            mod_type=args.generate,
            num_symbols=3000,
            sample_rate=args.rate,
            symbol_rate=args.rate / 10.0,
            snr_db=args.snr,
            cfo_hz=args.cfo,
            preamble=args.preamble
        )
    else:
        if not os.path.exists(args.input):
            print(f"[!] Error: File not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        if not args.quiet:
            print(f"[*] Ingesting file: {args.input}...")
        sig = load_signal_file(
            args.input,
            format_hint=args.format,
            sample_rate=args.rate,
            max_samples=65536
        )

    if not args.quiet:
        print("[*] Running automated parameter extraction, AMC classification, and decoding pipeline...\n")

    res = pipe.process_signal(
        sig,
        manual_mod=args.mod_override,
        manual_fec=args.fec_override,
        target_preamble=args.preamble
    )

    p = res["parameters"]
    amc = res["amc"]
    framing = res["framing"]
    fec = res["fec_status"]

    if not args.quiet:
        print("-------------------- EXTRACTED PARAMETERS --------------------")
        print(f"  Sampling Frequency (Fs):       {p['sample_rate']:,.0f} Hz ({p['sample_rate']/1e6:.2f} MHz)")
        print(f"  Carrier Frequency Offset (CFO): {p['estimated_cfo_hz']:+,.1f} Hz")
        print(f"  Estimated Baud Rate:           {p['estimated_baud_rate']:,.1f} Baud (SPS: {p['samples_per_symbol']})")
        print(f"  Estimated SNR (M2M4 Moment):   {p['estimated_snr_db']:+.1f} dB")
        print(f"  Occupied Bandwidth (99% OBW):  {p['obw_99_hz']/1e3:,.1f} kHz")
        print(f"  3dB Roll-off Bandwidth:        {p['bandwidth_3db_hz']/1e3:,.1f} kHz")
        print(f"  Total Processed Samples:       {p['num_samples']:,} ({p['duration_sec']:.4f} s)")

        print("\n---------------- AUTOMATED MODULATION CLASSIFICATION (AMC) ----------------")
        print(f"  Classified Modulation:         >>> {amc['predicted_modulation']} <<<")
        print(f"  Classification Confidence:     {amc['confidence']*100:.1f} %")
        print(f"  Higher-Order Cumulants:        C20={amc['features']['c20']:.3f}, C40={amc['features']['c40']:.3f}, C42={amc['features']['c42']:.3f}")
        print(f"  Instantaneous Stats:           gamma_max={amc['features']['gamma_max']:.2f}, sigma_aa={amc['features']['sigma_aa']:.3f}, sigma_af={amc['features']['sigma_af']:.3f}")

        print("\n----------------- DEMODULATION & DE-INTERLEAVING -----------------")
        print(f"  Active Demodulator:            {res['active_configuration']['modulation']}")
        print(f"  Recovered Symbol Count:        {res['demodulation']['num_symbols']:,}")
        print(f"  Recovered Raw Bit Count:       {res['demodulation']['num_bits_recovered']:,}")
        print(f"  De-interleaving Scheme:        {res['active_configuration']['interleaving'].upper()}")

        print("\n----------------- FORWARD ERROR CORRECTION (FEC) -----------------")
        print(f"  FEC Decoder Applied:           {fec.get('fec_applied', 'Bypass')}")
        print(f"  Decoding Status:               {fec.get('status', 'Completed')}")

        print("\n----------------- BITSTREAM CORRELATION & FRAMING ----------------")
        print(f"  Best Correlated Sync Marker:   {res['preambles']['best_preamble'] or 'None'}")
        print(f"  Synchronized Frames Detected:  {framing['num_frames_detected']}")
        print(f"  Payload Shannon Entropy:       {framing['overall_entropy']:.3f} bits/byte")
        print(f"  Total Recovered Payload:       {framing['total_bytes']:,} bytes\n")

        print("---------------- PAYLOAD HEX & ASCII PREVIEW -----------------")
        hex_lines = framing["hex_dump"].split("\n")[:8]
        for line in hex_lines:
            print(f"  {line}")
        print("================================================================================\n")

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        if not args.quiet:
            print(f"[OK] Analysis report saved to: {args.export}")

if __name__ == "__main__":
    main()
