"""
NTRO Signal Analyzer - Local Web Backend Server
Exposes REST API endpoints and serves high-performance interactive web dashboard:
- GET  /api/status
- POST /api/synthesize
- POST /api/analyze
- POST /api/upload
- GET  /api/export/<format>
- Static file serving (HTML, CSS, JS)
"""

import os
import sys
import json
import base64
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.ingestion import SignalData, load_signal_file
from core.synthetic_generator import generate_synthetic_rf
from core.pipeline import SignalAnalysisPipeline

# Global session cache
CURRENT_SIGNAL = None
CURRENT_RESULT = None
PIPELINE = SignalAnalysisPipeline()


class SignalAnalyzerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        static_dir = os.path.join(CURRENT_DIR, "static")
        super().__init__(*args, directory=static_dir, **kwargs)

    def _send_json(self, data: dict, status_code: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            global CURRENT_SIGNAL, CURRENT_RESULT
            self._send_json({
                "status": "online",
                "system": "NTRO SIH26147 Automated Signal Analyzer",
                "has_active_signal": CURRENT_SIGNAL is not None,
                "has_analysis_result": CURRENT_RESULT is not None,
                "signal_info": {
                    "sample_rate": CURRENT_SIGNAL.sample_rate if CURRENT_SIGNAL else None,
                    "num_samples": CURRENT_SIGNAL.num_samples if CURRENT_SIGNAL else 0,
                    "source_type": CURRENT_SIGNAL.source_type if CURRENT_SIGNAL else None
                } if CURRENT_SIGNAL else None
            })

        elif path.startswith("/api/export"):
            parts = path.split("/")
            export_format = parts[-1] if len(parts) > 3 else "json"
            if CURRENT_RESULT is None:
                self._send_json({"error": "No analysis result available to export"}, status_code=400)
                return

            if export_format == "json":
                # Full report export
                report_data = {
                    "parameters": CURRENT_RESULT["parameters"],
                    "amc": CURRENT_RESULT["amc"],
                    "configuration": CURRENT_RESULT["active_configuration"],
                    "fec_status": CURRENT_RESULT["fec_status"],
                    "framing": {
                        "num_frames_detected": CURRENT_RESULT["framing"]["num_frames_detected"],
                        "overall_entropy": CURRENT_RESULT["framing"]["overall_entropy"],
                        "total_bytes": CURRENT_RESULT["framing"]["total_bytes"],
                        "frames_summary": CURRENT_RESULT["framing"]["frames"]
                    }
                }
                body = json.dumps(report_data, indent=2).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Disposition', 'attachment; filename="ntro_signal_analysis_report.json"')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif export_format == "payload":
                hex_dump = CURRENT_RESULT["framing"].get("hex_dump", "")
                body = hex_dump.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Disposition', 'attachment; filename="recovered_payload_hex.txt"')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._send_json({"error": f"Unknown format {export_format}"}, status_code=400)

        else:
            # Fallback to static files
            super().do_GET()

    def do_POST(self):
        global CURRENT_SIGNAL, CURRENT_RESULT, PIPELINE
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b'{}'

        try:
            req_data = json.loads(body.decode('utf-8')) if body else {}
        except Exception:
            req_data = {}

        if path == "/api/synthesize":
            try:
                mod_type = req_data.get("modulation", "QPSK")
                snr_db = float(req_data.get("snr_db", 20.0))
                cfo_hz = float(req_data.get("cfo_hz", 2500.0))
                sample_rate = float(req_data.get("sample_rate", 1e6))
                symbol_rate = float(req_data.get("symbol_rate", 1e5))
                num_symbols = int(req_data.get("num_symbols", 3000))
                preamble = req_data.get("preamble", "CCSDS")

                sig, bits, syms = generate_synthetic_rf(
                    mod_type=mod_type,
                    num_symbols=num_symbols,
                    sample_rate=sample_rate,
                    symbol_rate=symbol_rate,
                    snr_db=snr_db,
                    cfo_hz=cfo_hz,
                    preamble=preamble
                )
                CURRENT_SIGNAL = sig

                # Auto run pipeline on synthesized signal
                result = PIPELINE.process_signal(
                    sig,
                    manual_mod=req_data.get("manual_mod", "auto"),
                    manual_fec=req_data.get("manual_fec", "auto"),
                    manual_interleaving=req_data.get("manual_interleave", "none"),
                    target_preamble=preamble
                )
                CURRENT_RESULT = result

                self._send_json({
                    "status": "success",
                    "message": f"Synthesized {mod_type} signal (SNR {snr_db} dB, CFO {cfo_hz} Hz)",
                    "result": result
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)

        elif path == "/api/analyze":
            try:
                if CURRENT_SIGNAL is None:
                    self._send_json({"error": "No signal currently loaded. Synthesize or upload one first."}, status_code=400)
                    return

                manual_mod = req_data.get("manual_mod", "auto")
                manual_fec = req_data.get("manual_fec", "auto")
                manual_interleaving = req_data.get("manual_interleave", "none")
                target_preamble = req_data.get("target_preamble", "CCSDS")

                result = PIPELINE.process_signal(
                    CURRENT_SIGNAL,
                    manual_mod=manual_mod,
                    manual_fec=manual_fec,
                    manual_interleaving=manual_interleaving,
                    target_preamble=target_preamble
                )
                CURRENT_RESULT = result

                self._send_json({
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)

        elif path == "/api/upload":
            try:
                # Expects base64 data and filename
                filename = req_data.get("filename", "uploaded_signal.iq")
                b64_content = req_data.get("content_base64", "")
                sample_rate = float(req_data.get("sample_rate", 1e6))
                format_hint = req_data.get("format_hint", "auto")

                raw_bytes = base64.b64decode(b64_content)
                upload_dir = os.path.join(PROJECT_ROOT, "uploads")
                os.makedirs(upload_dir, exist_ok=True)
                save_path = os.path.join(upload_dir, filename)

                with open(save_path, "wb") as f:
                    f.write(raw_bytes)

                sig = load_signal_file(
                    save_path,
                    format_hint=format_hint,
                    sample_rate=sample_rate,
                    max_samples=65536
                )
                CURRENT_SIGNAL = sig

                # Run automated analysis
                result = PIPELINE.process_signal(sig, manual_mod="auto")
                CURRENT_RESULT = result

                self._send_json({
                    "status": "success",
                    "filename": filename,
                    "sample_rate": sig.sample_rate,
                    "num_samples": sig.num_samples,
                    "result": result
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)

        else:
            self._send_json({"error": "Endpoint not found"}, status_code=404)


def run_server(port: int = 8080):
    server_address = ('127.0.0.1', port)
    try:
        httpd = ThreadingHTTPServer(server_address, SignalAnalyzerHandler)
        print(f"[NTRO-SERVER] Server active at http://127.0.0.1:{port}")
        return httpd
    except OSError:
        # Fallback to next port
        httpd = ThreadingHTTPServer(('127.0.0.1', port + 1), SignalAnalyzerHandler)
        print(f"[NTRO-SERVER] Server active at http://127.0.0.1:{port + 1}")
        return httpd


if __name__ == "__main__":
    server = run_server()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[NTRO-SERVER] Shutting down...")
        server.server_close()
