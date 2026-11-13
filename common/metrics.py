"""P05 common metrics. Input arrays are synthetic or caller-provided in memory."""

import numpy as np
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score


def _binary_inputs(scores, y):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if scores.size == 0 or scores.size != y.size:
        raise ValueError("Nonempty scores and labels must have identical lengths")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be finite")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("Labels must be binary")
    return scores, y.astype(np.int8)


def auroc(scores, y):
    """ROC AUC for continuous scores; positive class is y=1."""
    scores, y = _binary_inputs(scores, y)
    if np.unique(y).size != 2:
        raise ValueError("AUROC requires both classes")
    return float(roc_auc_score(y, scores))


def auprc(scores, y):
    """Noninterpolated PR area (sklearn average precision)."""
    scores, y = _binary_inputs(scores, y)
    if np.unique(y).size != 2:
        raise ValueError("AUPRC requires both classes")
    return float(average_precision_score(y, scores))


def ece(p, y, bins=15):
    """Expected calibration error with equal-width bins on [0, 1]."""
    p, y = _binary_inputs(p, y)
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    indices = np.minimum((p * bins).astype(np.int64), bins - 1)
    totals = np.bincount(indices, minlength=bins)
    positive = np.bincount(indices, weights=y, minlength=bins)
    confidence = np.bincount(indices, weights=p, minlength=bins)
    active = totals > 0
    error = np.sum(np.abs(positive[active] / totals[active] -
                          confidence[active] / totals[active]) * totals[active])
    return float(error / p.size)


def challenge_score(tp, tn, fp, fn):
    """PhysioNet 2015-style challenge score from confusion-matrix counts."""
    counts = np.asarray([tp, tn, fp, fn], dtype=np.float64)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Confusion-matrix counts must be finite and nonnegative")
    if not np.all(counts == np.floor(counts)):
        raise ValueError("Confusion-matrix counts must be integers")
    denominator = counts[0] + counts[1] + counts[2] + 5 * counts[3]
    if denominator == 0:
        raise ValueError("Challenge score requires at least one observation")
    return float(100 * (counts[0] + counts[1]) / denominator)


def fa_suppression(scores, y, lam):
    """Fraction of false alarms (y=0) suppressed by score < lam."""
    scores, y = _binary_inputs(scores, y)
    if not np.isfinite(lam):
        raise ValueError("Threshold must be finite")
    negatives = y == 0
    if not negatives.any():
        raise ValueError("False-alarm suppression requires y=0 examples")
    return float(np.mean(scores[negatives] < lam))


def fnr(scores, y, lam):
    """Fraction of true alarms (y=1) suppressed by score < lam."""
    scores, y = _binary_inputs(scores, y)
    if not np.isfinite(lam):
        raise ValueError("Threshold must be finite")
    positives = y == 1
    if not positives.any():
        raise ValueError("FNR requires y=1 examples")
    return float(np.mean(scores[positives] < lam))


def clopper_pearson(k, n, conf=0.95):
    """Exact TWO-SIDED Clopper-Pearson binomial interval."""
    if (not isinstance(n, (int, np.integer)) or isinstance(n, bool)
            or not isinstance(k, (int, np.integer)) or isinstance(k, bool)
            or n < 1 or k < 0 or k > n):
        raise ValueError("Require integer counts 0 <= k <= n and n >= 1")
    if not np.isfinite(conf) or not 0 < conf < 1:
        raise ValueError("Confidence level must lie in (0, 1)")
    tail = (1 - conf) / 2
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
    return lower, upper
