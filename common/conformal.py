"""CRC and Learn-then-Test threshold selection for alarm suppression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

from .metrics import _binary_inputs


def crc_fnr_threshold(scores, y, groups, alpha, weights=None):
    """Largest eligible strict-suppression threshold under record-level CRC.

    A suppression occurs iff score < lambda. Only records containing at least
    one y=1 event contribute to the calibration risk. Each such record has
    total weight 1, regardless of its number of true events.
    """
    scores, y = _binary_inputs(scores, y)
    groups = np.asarray(groups).reshape(-1)
    if groups.size != y.size or pd.isna(groups).any():
        raise ValueError("groups must match scores and contain no missing IDs")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    if weights is None:
        weights = np.ones(scores.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.size != scores.size:
            raise ValueError("weights must match scores")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("weights must be finite and strictly positive")

    group_codes, _ = pd.factorize(groups, sort=False)
    positive = y == 1
    n_groups = int(group_codes.max()) + 1
    positive_count = np.bincount(
        group_codes, weights=positive.astype(np.float64), minlength=n_groups
    )
    n = int(np.count_nonzero(positive_count))
    if n == 0:
        raise ValueError("CRC needs at least one record with a positive event")
    if 1.0 / (n + 1) > alpha:
        raise ValueError("Too few positive records for requested CRC alpha")

    denominators = np.bincount(
        group_codes, weights=weights * positive, minlength=n_groups
    )
    pos_idx = np.flatnonzero(positive)
    pos_scores = scores[pos_idx]
    # This is each positive event's contribution to its RECORD's loss.
    contributions = weights[pos_idx] / denominators[group_codes[pos_idx]]
    order = np.argsort(pos_scores, kind="mergesort")
    sorted_scores = pos_scores[order]
    prefix = np.concatenate(([0.0], np.cumsum(contributions[order])))
    unique_scores, first_indices = np.unique(
        sorted_scores, return_index=True
    )

    # Strict < means tied positive scores are NOT lost at lambda == score.
    empirical_total = prefix[first_indices]
    bounds = (empirical_total + 1.0) / (n + 1)
    eligible = (unique_scores > 0.0) & (bounds <= alpha)
    best = float(unique_scores[eligible][-1]) if eligible.any() else 0.0

    # Explicitly inspect the boundary immediately after the final positive
    # score. For alpha < 1 it cannot be eligible: every positive is suppressed.
    next_boundary = np.nextafter(unique_scores[-1], np.inf)
    if np.isfinite(next_boundary) and next_boundary > 0:
        boundary_bound = (float(n) + 1.0) / (n + 1)
        if boundary_bound <= alpha:
            best = max(best, float(next_boundary))

    return best


def ltt_cp_threshold(scores, y, alpha, delta=0.05, grid=None):
    """LTT with one-sided Clopper-Pearson and Bonferroni correction.

    For each candidate tau, the risk is P(y=1 | score <= tau). Select the
    candidate passing its corrected upper binomial bound that suppresses the
    most y=0 events. Ties favor the smaller threshold.
    """
    scores, y = _binary_inputs(scores, y)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if grid is None:
        raise ValueError("grid is required")
    grid = np.asarray(list(grid), dtype=np.float64).reshape(-1)
    if grid.size == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("grid must contain finite candidate thresholds")

    tested_tail = delta / grid.size
    best = None
    best_suppressed_negatives = -1
    for tau in grid:
        selected = scores <= tau
        count = int(np.sum(selected))
        if count == 0:
            # No observations provide no binomial evidence of low risk.
            continue
        positives = int(np.sum(y[selected]))
        upper = (
            1.0 if positives == count
            else float(beta.ppf(1 - tested_tail, positives + 1,
                                count - positives))
        )
        if upper <= alpha:
            suppressed_negatives = count - positives
            if (suppressed_negatives > best_suppressed_negatives
                    or (suppressed_negatives == best_suppressed_negatives
                        and (best is None or tau < best))):
                best = float(tau)
                best_suppressed_negatives = suppressed_negatives
    return best
