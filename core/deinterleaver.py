"""
De-Interleaving Engine
Supports:
1. Block De-Interleaver (Matrix M x N transposition)
2. Convolutional De-Interleaver (Ramsey / Forney B-branch structure)
3. Diagonal De-Interleaver (Diagonal block mapping)
4. Pseudo-Random (PR) De-Interleaver (LFSR / Seed permutation)
Includes blind parameter estimation using autocorrelation analysis.
"""

import numpy as np

# ----------------- 1. Block Interleaver / De-interleaver -----------------

def block_interleave(bits: np.ndarray, rows: int = 16, cols: int = 16) -> np.ndarray:
    """Interleaves bits by writing row-wise and reading column-wise."""
    block_size = rows * cols
    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant')

    num_blocks = len(bits) // block_size
    interleaved = []
    for b in range(num_blocks):
        blk = bits[b * block_size:(b + 1) * block_size].reshape(rows, cols)
        interleaved.append(blk.T.flatten())

    return np.concatenate(interleaved)


def block_deinterleave(bits: np.ndarray, rows: int = 16, cols: int = 16) -> np.ndarray:
    """De-interleaves bits by reversing row/column transposition."""
    block_size = rows * cols
    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant')

    num_blocks = len(bits) // block_size
    deinterleaved = []
    for b in range(num_blocks):
        # Transpose back: read column-wise into (cols, rows) and transpose to (rows, cols)
        blk = bits[b * block_size:(b + 1) * block_size].reshape(cols, rows)
        deinterleaved.append(blk.T.flatten())

    return np.concatenate(deinterleaved)


# ------------- 2. Convolutional Interleaver / De-interleaver -------------

def convolutional_interleave(bits: np.ndarray, branches: int = 6, delay_step: int = 2) -> np.ndarray:
    """
    Forney / Ramsey Convolutional Interleaver.
    Branch i has delay i * delay_step.
    """
    delays = [list(np.zeros(i * delay_step, dtype=np.uint8)) for i in range(branches)]
    interleaved = []

    for idx, bit in enumerate(bits):
        b = idx % branches
        delays[b].append(bit)
        out_bit = delays[b].pop(0)
        interleaved.append(out_bit)

    return np.array(interleaved, dtype=np.uint8)


def convolutional_deinterleave(bits: np.ndarray, branches: int = 6, delay_step: int = 2) -> np.ndarray:
    """
    Forney Convolutional De-interleaver.
    Branch i has complementary delay (branches - 1 - i) * delay_step.
    """
    delays = [list(np.zeros((branches - 1 - i) * delay_step, dtype=np.uint8)) for i in range(branches)]
    deinterleaved = []

    for idx, bit in enumerate(bits):
        b = idx % branches
        delays[b].append(bit)
        out_bit = delays[b].pop(0)
        deinterleaved.append(out_bit)

    # Flush end delay
    total_delay = branches * (branches - 1) * delay_step
    if len(deinterleaved) > total_delay:
        return np.array(deinterleaved[total_delay:], dtype=np.uint8)
    return np.array(deinterleaved, dtype=np.uint8)


# ---------------- 3. Diagonal Interleaver / De-interleaver ---------------

def diagonal_interleave(bits: np.ndarray, size: int = 16) -> np.ndarray:
    """
    Diagonal Interleaver: Maps block matrix (size x size) along wrapped diagonals.
    """
    block_size = size * size
    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant')

    num_blocks = len(bits) // block_size
    output = []

    for b in range(num_blocks):
        blk = bits[b * block_size:(b + 1) * block_size].reshape(size, size)
        diag_blk = np.zeros_like(blk)
        for r in range(size):
            for c in range(size):
                diag_blk[(r + c) % size, c] = blk[r, c]
        output.append(diag_blk.flatten())

    return np.concatenate(output)


def diagonal_deinterleave(bits: np.ndarray, size: int = 16) -> np.ndarray:
    """Reverses diagonal interleaver mapping."""
    block_size = size * size
    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant')

    num_blocks = len(bits) // block_size
    output = []

    for b in range(num_blocks):
        diag_blk = bits[b * block_size:(b + 1) * block_size].reshape(size, size)
        blk = np.zeros_like(diag_blk)
        for r in range(size):
            for c in range(size):
                blk[r, c] = diag_blk[(r + c) % size, c]
        output.append(blk.flatten())

    return np.concatenate(output)


# -------------- 4. Pseudo-Random (PR) Interleaver / De-interleaver -------

def generate_pr_permutation(length: int, seed: int = 1337) -> np.ndarray:
    """Generates a deterministic pseudo-random permutation."""
    rng = np.random.default_rng(seed)
    return rng.permutation(length)


def pr_interleave(bits: np.ndarray, block_size: int = 256, seed: int = 1337) -> np.ndarray:
    """Applies pseudo-random permutation block-by-block."""
    perm = generate_pr_permutation(block_size, seed)
    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant')

    num_blocks = len(bits) // block_size
    interleaved = []
    for b in range(num_blocks):
        blk = bits[b * block_size:(b + 1) * block_size]
        interleaved.append(blk[perm])

    return np.concatenate(interleaved)


def pr_deinterleave(bits: np.ndarray, block_size: int = 256, seed: int = 1337) -> np.ndarray:
    """Inverts pseudo-random permutation."""
    perm = generate_pr_permutation(block_size, seed)
    inv_perm = np.zeros(block_size, dtype=int)
    inv_perm[perm] = np.arange(block_size)

    pad_len = (block_size - (len(bits) % block_size)) % block_size
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant')

    num_blocks = len(bits) // block_size
    deinterleaved = []
    for b in range(num_blocks):
        blk = bits[b * block_size:(b + 1) * block_size]
        deinterleaved.append(blk[inv_perm])

    return np.concatenate(deinterleaved)


# ---------------- 5. Blind Interleaver Parameter Detection ----------------

def estimate_interleaver_period(bits: np.ndarray, max_period: int = 1024) -> tuple[int, float]:
    """
    Blindly detects periodic interleaver depth/period using bit autocorrelation.
    Returns: (detected_period, confidence)
    """
    if len(bits) < 256:
        return 0, 0.0

    b = bits.astype(float) * 2.0 - 1.0
    corr = np.correlate(b[:4096], b[:4096], mode='full')
    half_corr = corr[len(corr) // 2 + 1:len(corr) // 2 + max_period]

    if len(half_corr) == 0:
        return 0, 0.0

    peak_idx = int(np.argmax(half_corr)) + 1
    peak_val = half_corr[peak_idx - 1] / half_corr[0]

    return peak_idx, float(np.clip(peak_val, 0.0, 1.0))


def apply_deinterleaving(bits: np.ndarray, scheme: str = "none",
                         param1: int = 16, param2: int = 16) -> np.ndarray:
    """Universal dispatcher for de-interleaving."""
    s = scheme.lower()
    if s == "block":
        return block_deinterleave(bits, rows=param1, cols=param2)
    elif s in ("convolutional", "convolution"):
        return convolutional_deinterleave(bits, branches=param1, delay_step=param2)
    elif s == "diagonal":
        return diagonal_deinterleave(bits, size=param1)
    elif s in ("pseudorandom", "pseudo_random", "pr"):
        return pr_deinterleave(bits, block_size=param1, seed=param2)
    else:
        # Pass through unchanged
        return bits
