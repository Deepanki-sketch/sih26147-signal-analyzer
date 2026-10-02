"""
NTRO Automated Signal Analyzer (SIH26147)
One-Click Application Launcher
Boots local server and opens browser dashboard automatically.
"""

import os
import sys
import webbrowser
import time
import socket

# Ensure current directory in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from gui.server import run_server

def find_free_port(start_port: int = 8080) -> int:
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
        port += 1
    return 8080

def main():
    print("=" * 75)
    print("   NATIONAL TECHNICAL RESEARCH ORGANISATION (NTRO) - SIH26147")
    print("   Automated Model for Analysis of .IQ & .WAV Files")
    print("=" * 75)

    port = find_free_port(8080)
    url = f"http://127.0.0.1:{port}/"
    print(f"\n[*] Initializing DSP Engine & Web Server at: {url}")

    server = run_server(port)

    # Open browser automatically after a short delay
    print("[*] Opening interactive web dashboard in default browser...")
    webbrowser.open(url)

    print("\n[✓] System Online! Press Ctrl+C in this terminal to terminate.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down NTRO Signal Analyzer server...")
        server.server_close()
        print("[*] Terminated.")

if __name__ == "__main__":
    main()
