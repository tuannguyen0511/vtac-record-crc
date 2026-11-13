"""P05-T02b pure signal/annotation functions. No filesystem access.

Filtering follows the Director's D1-D6 order and the audited T01b slot rules.
All samples beyond t0+delta are sliced away BEFORE any interpolation/filtering.
"""

from functools import lru_cache

import numpy as np
from scipy import signal


SLOTS = ("ECG1", "ECG2", "PPG", "ABP")
ECG_LIMB = ("II", "I", "III", "AVR", "AVL", "AVF")
ECG_CHEST = ("V", "MCL", "V1", "V2", "V3", "V4", "V5", "V6")


def normalized_channel(name):
    text = "".join(ch for ch in str(name).upper() if ch.isalnum())
    return text[3:] if text.startswith("ECG") else text


def select_slots(channel_names):
    """Return source channel index for ECG1, ECG2, PPG, ABP."""
    normalized = [normalized_channel(name) for name in channel_names]

    def first(candidates, excluded=None):
        for candidate in candidates:
            for index, name in enumerate(normalized):
                if name == candidate and index != excluded:
                    return index
        return None

    ecg1 = first(ECG_LIMB)
    ecg2 = first(ECG_CHEST)
    if ecg2 is None:
        ecg2 = first(ECG_LIMB, excluded=ecg1)
    return {
        "ECG1": ecg1,
        "ECG2": ecg2,
        "PPG": first(("PLETH",)),
        "ABP": first(("ABP",)),
    }


@lru_cache(maxsize=4)
def _filters(fs):
    notch_b, notch_a = signal.iirnotch(60, Q=30, fs=fs)
    notch_sos = signal.tf2sos(notch_b, notch_a)
    filters = {
        "ECG1": signal.butter(4, (1, 30), btype="bandpass", fs=fs, output="sos"),
        "ECG2": signal.butter(4, (1, 30), btype="bandpass", fs=fs, output="sos"),
        "PPG": signal.butter(4, (0.5, 5), btype="bandpass", fs=fs, output="sos"),
        "ABP": signal.butter(4, 16, btype="lowpass", fs=fs, output="sos"),
    }
    return notch_sos, filters


def filter_slot_segment(raw_segment, slot, fs=250):
    """Interpolate nonfinite samples and apply notch, then slot filter."""
    raw_segment = np.asarray(raw_segment, dtype=np.float64).reshape(-1)
    if raw_segment.size == 0 or slot not in SLOTS:
        raise ValueError("Nonempty segment and known slot required")
    finite = np.isfinite(raw_segment)
    if finite.any():
        samples = np.arange(raw_segment.size)
        cleaned = np.interp(samples, samples[finite], raw_segment[finite])
    else:
        cleaned = np.zeros(raw_segment.size, dtype=np.float64)
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    notch, filters = _filters(float(fs))
    filtered = signal.sosfiltfilt(notch, cleaned)
    return signal.sosfiltfilt(filters[slot], filtered)


def preprocess_event(samples, channel_names, W, D, *,
                     fs=250, t0=75000, start_sample=0,
                     context_s=60, nan_limit=0.20,
                     min_std=1e-6, clip=10.0,
                     return_diagnostics=False):
    """Return float32 [4, (W+D)*fs] and bool [4] from one event.

    samples starts at absolute sample index start_sample, possibly at
    t0-context_s*fs. It may include later samples, which are NEVER consulted.
    With return_diagnostics=True, append one tuple per valid slot:
    (preclip_mean, preclip_std, postclip_mean, postclip_std, n_clipped).
    The diagnostic switch never changes x or mask, and reads no extra samples.
    """
    fs = int(fs)
    W = int(W)
    D = int(D)
    if fs <= 120 or W < 1 or D < 0 or W > context_s:
        raise ValueError("Invalid sampling/window configuration")
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != len(channel_names):
        raise ValueError("Expected samples [time, n_channels]")
    cut_from = int(t0 - context_s * fs - start_sample)
    cut_to = int(t0 + D * fs - start_sample)
    if cut_from < 0 or cut_to > samples.shape[0] or cut_to <= cut_from:
        raise ValueError("Input does not cover the required causal context")
    # Mandatory causal cut. Post-cut samples have no influence on any step.
    segment = np.asarray(samples[cut_from:cut_to, :], dtype=np.float64)
    n_window = (W + D) * fs
    x = np.zeros((4, n_window), dtype=np.float32)
    mask = np.zeros(4, dtype=np.bool_)
    diagnostics = [None] * 4 if return_diagnostics else None
    selection = select_slots(channel_names)
    for slot_id, slot in enumerate(SLOTS):
        channel_index = selection[slot]
        if channel_index is None:
            continue
        original = segment[:, channel_index]
        original_window = original[-n_window:]
        if np.mean(~np.isfinite(original_window)) > nan_limit:
            continue
        filtered = filter_slot_segment(original, slot, fs=fs)
        cropped = filtered[-n_window:]
        sigma = float(np.std(cropped, ddof=0))
        if not np.isfinite(sigma) or sigma < min_std:
            continue
        # Preserve D5 exactly: one window-local z-score, then one clip.
        z = (cropped - np.mean(cropped)) / sigma
        normalized = np.clip(z, -clip, clip)
        if not np.all(np.isfinite(normalized)):
            continue
        x[slot_id] = normalized.astype(np.float32)
        mask[slot_id] = True
        if return_diagnostics:
            # Check the actual float32 cache, not only its float64 precursor.
            cached = x[slot_id].astype(np.float64)
            diagnostics[slot_id] = (
                float(np.mean(z)), float(np.std(z, ddof=0)),
                float(np.mean(cached)), float(np.std(cached, ddof=0)),
                int(np.count_nonzero((z < -clip) | (z > clip))),
            )
    if return_diagnostics:
        return x, mask, diagnostics
    return x, mask


def soft_label(y, annotation_rows):
    """Director E: (q_mean, q_adj, contested, n_votes, votes_missing).

    annotation_rows is a list of (decision, is_adjudication) values for ONE
    event. No input file access and no cross-event statistical calculation.
    """
    if y not in (0, 1):
        raise ValueError("y must be binary")
    retained = []
    binary = set()
    adjudicated = False
    has_nonadjudication_row = False
    for decision, is_adjudication in annotation_rows:
        is_adj = str(is_adjudication).strip().lower() in ("true", "1")
        adjudicated = adjudicated or is_adj
        if is_adj:
            continue
        has_nonadjudication_row = True
        decision = str(decision).strip().lower()
        if decision == "true":
            retained.append(1.0)
            binary.add(1)
        elif decision == "false":
            retained.append(0.0)
            binary.add(0)
        elif decision == "uncertain":
            retained.append(0.5)
        elif decision in ("reject", "save for later"):
            continue
        else:
            raise ValueError("Unexpected annotator decision")
    if not has_nonadjudication_row:
        # HF1 K5=NO: missing means no non-adjudication annotation ROW,
        # not zero retained scores after discarding Reject/Save for Later.
        return float(y), float(y), 0, 0, 1
    # Rows may exist but all be Reject/Save for Later. Their vote scores
    # are discarded; they do not count as a missing KEY in HF1.
    q_mean = float(np.mean(retained)) if retained else float(y)
    q_adj = ((sum(retained) + y) / (len(retained) + 1)
             if adjudicated else q_mean)
    contested = int((len(binary) == 2) or adjudicated)
    return q_mean, float(q_adj), contested, len(retained), 0
