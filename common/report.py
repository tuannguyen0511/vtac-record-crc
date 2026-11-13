"""Small-count masking and compact scientific run-check reporting."""

from __future__ import annotations

import math
from numbers import Integral


def mask_small(n):
    """Hide positive integer counts smaller than 10; retain 0 and >=10."""
    if isinstance(n, bool) or not isinstance(n, Integral):
        raise TypeError("n must be an integer count")
    if n < 0:
        raise ValueError("n must be nonnegative")
    return "<10" if 0 < n < 10 else str(n)


def print_report(title, fields):
    """Print deterministic report lines from an ordered mapping of scalars."""
    if not isinstance(fields, dict):
        raise TypeError("fields must be a dictionary")
    print(f"=== REPORT {title} ===", flush=True)
    for key, value in fields.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("REPORT keys must be nonempty strings")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite REPORT value: {key}")
        print(f"{key}: {value}", flush=True)
    print("=== END REPORT ===", flush=True)
