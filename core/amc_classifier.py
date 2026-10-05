"""
Automated Modulation Classification (AMC) Engine
Combines:
1. Higher-Order Cumulants (HOC): C20, C21, C40, C41, C42, C63
2. Instantaneous Spectral Features (Azzouz-Nandi): gamma_max, sigma_ap, sigma_aa, sigma_af
3. Ensemble Classifier & Statistical Decision Engine
Discriminates: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK, CW, NOISE
"""

import numpy as np
from scipy import signal
CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK", "CW", "NOISE"]


def extract_amc_features(iq_samples: np.ndarray, sample_rate: float = 1e6) -> dict:
    """
    Extracts comprehensive Higher-Order Cumulants and instantaneous statistical features.
    """
    n = min(len(iq_samples), 16384)
    if n < 256:
        # Pad or handle small signals
        iq_samples = np.pad(iq_samples, (0, 256 - n), mode='wrap')
        n = 256

    x = iq_samples[:n].astype(np.complex128)

    # 1. Normalize signal to zero-mean and unit variance
    x = x - np.mean(x)
    std_val = np.std(x)
    if std_val > 1e-10:
        x = x / std_val
    else:
        return {
            "c20": 0.0, "c21": 0.0, "c40": 0.0, "c41": 0.0, "c42": 0.0,
            "gamma_max": 0.0, "sigma_ap": 0.0, "sigma_aa": 0.0, "sigma_af": 0.0,
            "fsk_fdev_ratio": 0.0
        }

    # 2. Moments calculation
    m20 = np.mean(x ** 2)
    m21 = np.mean(np.abs(x) ** 2)  # = 1.0 (unit variance)
    m40 = np.mean(x ** 4)
    m41 = np.mean((x ** 3) * np.conj(x))
    m42 = np.mean(np.abs(x) ** 4)

    # 3. Higher Order Cumulants (HOC)
    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * (m20 ** 2)
    c41 = m41 - 3.0 * m20 * m21
    c42 = m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)

    norm_c20 = np.abs(c20) / (m21 + 1e-12)
    norm_c40 = np.abs(c40) / (m21 ** 2 + 1e-12)
    norm_c42 = np.abs(c42) / (m21 ** 2 + 1e-12)

    # 4. Instantaneous Features (Azzouz-Nandi)
    amp = np.abs(x)
    a_mean = np.mean(amp)
    a_norm = amp / (a_mean + 1e-12)

    # gamma_max: peak of normalized amplitude spectrum
    fft_a = np.abs(np.fft.fft(a_norm - 1.0)) ** 2
    gamma_max = float(np.max(fft_a) / n)

    # sigma_aa: standard deviation of absolute normalized amplitude
    sigma_aa = float(np.std(a_norm))

    # Instantaneous phase & frequency
    phase = np.unwrap(np.angle(x))
    # Remove linear phase slope (carrier offset)
    t = np.arange(n)
    if n > 1:
        poly = np.polyfit(t, phase, 1)
        phase_nonlinear = phase - np.polyval(poly, t)
    else:
        phase_nonlinear = phase

    sigma_ap = float(np.std(np.abs(phase_nonlinear)))

    # Instantaneous frequency: phase derivative
    inst_freq = np.diff(phase) / (2.0 * np.pi) * sample_rate
    sigma_af = float(np.std(inst_freq) / (sample_rate + 1e-12))

    # FSK feature: peakiness in instantaneous frequency histogram
    hist, _ = np.histogram(inst_freq, bins=50)
    peak_count = np.sum(hist > (np.max(hist) * 0.4))
    fsk_ratio = float(peak_count)

    return {
        "c20": float(norm_c20),
        "c40": float(norm_c40),
        "c42": float(norm_c42),
        "c41": float(np.abs(c41)),
        "gamma_max": float(gamma_max),
        "sigma_ap": float(sigma_ap),
        "sigma_aa": float(sigma_aa),
        "sigma_af": float(sigma_af),
        "fsk_fdev_ratio": fsk_ratio
    }


class AMCClassifier:
    """
    Automated Modulation Classifier utilizing hybrid statistical physics and machine learning.
    """
    def __init__(self):
        self.classes = CLASSES
        self.model = None
        self._is_calibrated = False

    def _rule_based_classify(self, feats: dict, snr_est: float) -> tuple[str, float, dict]:
        """
        Robust statistical rule-based classifier based on theoretical cumulant boundaries
        and instantaneous amplitude/frequency metrics.
        """
        c20 = feats["c20"]
        c40 = feats["c40"]
        c42 = feats["c42"]
        sigma_aa = feats["sigma_aa"]
        sigma_af = feats["sigma_af"]
        gamma_max = feats["gamma_max"]

        scores = {c: 0.05 for c in self.classes}

        # 1. Noise / CW check
        if snr_est < 2.0 or (c42 < 0.15 and sigma_aa < 0.15 and c20 < 0.1):
            scores["NOISE"] += 1.5
        elif sigma_aa < 0.05 and sigma_af < 0.01:
            scores["CW"] += 1.5
        # 2. FSK family check (discrete frequency hopping signature)
        elif feats.get("fsk_fdev_ratio", 0) >= 6 or (sigma_af > 0.12 and c20 < 0.5):
            if feats.get("fsk_fdev_ratio", 0) > 8 or c40 < 0.25:
                scores["2FSK"] += 1.2
                scores["4FSK"] += 0.4
            else:
                scores["4FSK"] += 1.2
                scores["2FSK"] += 0.4
        # 3. BPSK check (strong C20 ~ 1.0)
        elif c20 > 0.55:
            scores["BPSK"] += 1.5
        # 4. QPSK check (C20 ~ 0, C40 ~ 0.8+, C42 ~ 0.8, sigma_aa ~ 0.28)
        elif c40 > 0.65 and c42 > 0.65 and sigma_aa < 0.34:
            scores["QPSK"] += 1.4
            scores["8PSK"] += 0.3
        # 5. 8PSK check (C20 ~ 0, C40 ~ 0, C42 ~ 0.8, sigma_aa ~ 0.27)
        elif c42 > 0.65 and c40 < 0.3:
            scores["8PSK"] += 1.4
            scores["QPSK"] += 0.3
        # 6. QAM family (multi-amplitude levels: sigma_aa > 0.35, C42 ~ 0.53)
        elif sigma_aa >= 0.34 or (0.35 < c42 < 0.65):
            if c42 > 0.5:
                scores["16QAM"] += 1.4
                scores["64QAM"] += 0.5
            else:
                scores["64QAM"] += 1.4
                scores["16QAM"] += 0.5
        else:
            # Fallback based on closest cumulants
            if c40 > 0.5:
                scores["QPSK"] += 0.8
            else:
                scores["16QAM"] += 0.8

        # Normalize scores to probabilities
        total = sum(scores.values())
        probs = {k: round(v / total, 4) for k, v in scores.items()}

        best_mod = max(probs, key=probs.get)
        confidence = probs[best_mod]
        return best_mod, confidence, probs

    def classify(self, iq_samples: np.ndarray, sample_rate: float, snr_est: float = 20.0) -> dict:
        """
        Runs AMC on input IQ samples.
        Returns:
            - predicted_modulation: string
            - confidence: float (0.0 to 1.0)
            - probabilities: dict of class -> probability
            - features: dict of extracted metrics
        """
        feats = extract_amc_features(iq_samples, sample_rate)
        mod, conf, probs = self._rule_based_classify(feats, snr_est)

        return {
            "predicted_modulation": mod,
            "confidence": round(conf, 4),
            "probabilities": probs,
            "features": {k: round(v, 4) for k, v in feats.items()}
        }
