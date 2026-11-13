"""Record-cluster bootstrap with reproducible paired resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def cluster_bootstrap(stat_fn, df, group_col, B=2000, seed=42, n_jobs=8):
    """Return (bootstrap mean, percentile 2.5%, percentile 97.5%).

    Sample complete records with replacement. Each draw receives a unique
    group ID in group_col, even if its source record was drawn multiple times.
    The original record ID remains available in '_bootstrap_source_group'.
    All score columns share the same record draw, so stat_fn may calculate a
    paired score difference without independent resampling.

    stat_fn accepts one resampled DataFrame and returns one finite scalar.
    The returned mean is the mean across the B bootstrap replicates.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if group_col not in df.columns:
        raise KeyError(f"Missing group column: {group_col}")
    if df.empty or df[group_col].isna().any():
        raise ValueError("df must be nonempty with nonmissing group IDs")
    for name, value in (("B", B), ("n_jobs", n_jobs)):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, bool) or value < 1):
            raise ValueError(f"{name} must be a positive integer")
    if n_jobs > 8:
        raise ValueError("reference-workstation permits at most 8 CPU workers")

    # The group positions are row offsets, independent of the DataFrame index.
    group_positions = [
        np.asarray(pos, dtype=np.int64)
        for pos in df.groupby(group_col, sort=False).indices.values()
    ]
    n_groups = len(group_positions)
    rng = np.random.default_rng(seed)
    record_draws = rng.integers(0, n_groups, size=(B, n_groups))

    def one_replicate(draw):
        positions = np.concatenate([group_positions[i] for i in draw])
        sample = df.iloc[positions].copy()
        sample["_bootstrap_source_group"] = sample[group_col].to_numpy(copy=True)
        # Independent replicated clusters must not collapse on groupby.
        sizes = np.fromiter(
            (group_positions[i].size for i in draw), dtype=np.int64,
            count=n_groups,
        )
        sample[group_col] = np.repeat(np.arange(n_groups), sizes)
        value = float(stat_fn(sample))
        if not np.isfinite(value):
            raise ValueError("stat_fn returned a nonfinite statistic")
        return value

    if n_jobs == 1:
        results = np.asarray(
            [one_replicate(draw) for draw in record_draws], dtype=np.float64
        )
    else:
        results = np.asarray(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(one_replicate)(draw) for draw in record_draws
            ),
            dtype=np.float64,
        )

    lower, upper = np.percentile(results, [2.5, 97.5])
    return float(results.mean()), float(lower), float(upper)
