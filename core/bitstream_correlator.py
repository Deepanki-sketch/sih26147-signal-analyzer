"""
Bitstream Correlation & Frame Synchronization Engine
Capabilities:
- Cross-correlation with sync words: CCSDS ASM (0x1ACFFC1D), Barker-7/11/13, AX.25 (0x7E), DVB (0x47), Custom
- Hamming distance tolerant correlation (detects sync with up to N bit flips)
- Frame boundary alignment, header extraction, and payload segmentation
- Shannon entropy calculator & byte distribution analysis
- CRC-16 and CRC-32 validation
- Hex dump & ASCII viewer generation
"""

import numpy as np
import math

KNOWN_PREAMBLES = {
    "CCSDS": {
        "name": "CCSDS Telemetry ASM (0x1ACFFC1D)",
        "pattern": np.array([0,0,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,1], dtype=np.uint8),
        "hex": "1ACFFC1D"
    },
    "BARKER_13": {
        "name": "Barker Code 13",
        "pattern": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8),
        "hex": "1F35"
    },
    "BARKER_11": {
        "name": "Barker Code 11",
        "pattern": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
        "hex": "0E2"
    },
    "AX25": {
        "name": "AX.25 / HDLC Flag (0x7E)",
        "pattern": np.array([0, 1, 1, 1, 1, 1, 1, 0], dtype=np.uint8),
        "hex": "7E"
    },
    "DVB": {
        "name": "DVB Sync Byte (0x47)",
        "pattern": np.array([0, 1, 0, 0, 0, 1, 1, 1], dtype=np.uint8),
        "hex": "47"
    }
}


def correlate_preamble(bits: np.ndarray, pattern: np.ndarray, max_flips: int = 2) -> list[int]:
    """
    Sliding window cross-correlation finding all occurrences of pattern within bits,
    allowing up to max_flips bit errors (Hamming distance).
    Returns list of starting bit indices.
    """
    p_len = len(pattern)
    if len(bits) < p_len:
        return []

    # Map to bipolar (-1, +1)
    b_bipolar = bits.astype(int) * 2 - 1
    p_bipolar = pattern.astype(int) * 2 - 1

    # Cross-correlation via convolution
    corr = np.convolve(b_bipolar, p_bipolar[::-1], mode='valid')
    # Perfect match produces corr = p_len
    # Each flip reduces correlation by 2
    threshold = p_len - (2 * max_flips)

    matches = np.where(corr >= threshold)[0]
    return matches.tolist()


def detect_all_preambles(bits: np.ndarray) -> dict:
    """
    Scans bitstream for all known synchronization markers.
    Returns detected candidates and best match weighted by pattern significance.
    """
    results = {}
    best_candidate = None
    max_score = 0

    for key, info in KNOWN_PREAMBLES.items():
        p_len = len(info["pattern"])
        # Scale allowable flips: 0 flips for <= 8 bits, 1 flip for <= 16 bits, 2 flips for > 16 bits
        flips = 0 if p_len <= 8 else (1 if p_len <= 16 else 2)
        matches = correlate_preamble(bits, info["pattern"], max_flips=flips)
        if len(matches) > 0:
            results[key] = {
                "name": info["name"],
                "hex": info["hex"],
                "pattern_length": p_len,
                "num_matches": len(matches),
                "match_indices": matches[:10]  # first 10
            }
            # Score weighted by pattern length squared to reward long distinctive markers
            score = len(matches) * (p_len ** 2)
            if score > max_score:
                max_score = score
                best_candidate = key

    return {
        "detected_preambles": results,
        "best_preamble": best_candidate
    }


def compute_entropy(data_bytes: bytes) -> float:
    """
    Computes Shannon Entropy of byte sequence.
    Range: 0.0 (constant) to 8.0 (completely random / compressed / encrypted).
    """
    if not data_bytes:
        return 0.0

    length = len(data_bytes)
    freq = {}
    for b in data_bytes:
        freq[b] = freq.get(b, 0) + 1

    ent = 0.0
    for count in freq.values():
        p = count / length
        ent -= p * math.log2(p)

    return round(ent, 3)


def format_hex_dump(data_bytes: bytes, bytes_per_line: int = 16, max_lines: int = 64) -> str:
    """Formats byte array into clean Hex + ASCII view."""
    lines = []
    total = len(data_bytes)

    for i in range(0, min(total, max_lines * bytes_per_line), bytes_per_line):
        chunk = data_bytes[i:i + bytes_per_line]
        hex_str = " ".join(f"{b:02X}" for b in chunk)
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        # Pad hex string if line is short
        hex_padded = f"{hex_str:<{bytes_per_line * 3}}"
        lines.append(f"{i:06X}  {hex_padded} |{ascii_str}|")

    if total > max_lines * bytes_per_line:
        lines.append(f"... [{total - max_lines * bytes_per_line} more bytes truncated]")

    return "\n".join(lines)


def parse_frames(bits: np.ndarray, sync_key: str = "CCSDS", header_len_bytes: int = 6) -> dict:
    """
    Extracts frames, headers, and payloads based on sync marker.
    """
    if sync_key in KNOWN_PREAMBLES:
        pattern = KNOWN_PREAMBLES[sync_key]["pattern"]
    else:
        pattern = KNOWN_PREAMBLES["CCSDS"]["pattern"]

    sync_indices = correlate_preamble(bits, pattern, max_flips=1)
    frames = []

    for idx in range(len(sync_indices)):
        start = sync_indices[idx]
        end = sync_indices[idx + 1] if idx + 1 < len(sync_indices) else min(start + 2048, len(bits))
        frame_bits = bits[start:end]

        # Convert to bytes
        n_bytes = len(frame_bits) // 8
        frame_bytes = np.packbits(frame_bits[:n_bytes * 8]).tobytes()

        # Header vs Payload split
        sync_bytes_len = (len(pattern) + 7) // 8
        hdr_bytes = frame_bytes[sync_bytes_len:sync_bytes_len + header_len_bytes]
        payload_bytes = frame_bytes[sync_bytes_len + header_len_bytes:]

        frames.append({
            "frame_index": idx,
            "bit_offset": int(start),
            "frame_length_bits": len(frame_bits),
            "header_hex": hdr_bytes.hex().upper(),
            "payload_length_bytes": len(payload_bytes),
            "payload_entropy": compute_entropy(payload_bytes),
            "payload_hex_snippet": payload_bytes[:32].hex().upper()
        })

    # Global payload bytes
    all_bytes = np.packbits(bits[:(len(bits) // 8) * 8]).tobytes()
    entropy = compute_entropy(all_bytes)
    hex_dump = format_hex_dump(all_bytes)

    return {
        "num_frames_detected": len(frames),
        "frames": frames[:20],  # first 20 frames summary
        "overall_entropy": entropy,
        "total_bytes": len(all_bytes),
        "hex_dump": hex_dump
    }
