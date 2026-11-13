"""P05-T02b: causal waveform cache, sealed test targets, preregistration.

NO test decisions are read by the training/validation/statistics process.
The separate sealing process reads test decisions only to write its parquet.
All aggregate checks are performed before that process is launched.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import argparse
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import platform
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from numpy.lib.format import open_memmap
from sklearn.model_selection import train_test_split

from preprocess_lib import SLOTS, filter_slot_segment, preprocess_event, soft_label


EXPECTED_COMMON = {
    "metrics.py": "f0e07d1edcd7e4d1fd9bceb1b397af4392f73854e582e3257f7790a05fdf902e",
    "bootstrap.py": "ceda94c2422f31a1932068c0864a10831b7f6abbedbaea5af3625dc98d134f48",
    "conformal.py": "a7020f93ee7dbee9887662663b9c460b6ee92bd39202b0f16526046d2790fab1",
    "report.py": "759ba2ecea23ea2b9ea050b71370676f0a3307c946421b9de88f8e282f40e7a3",
    "tests/test_common.py": "0831056ca116af0c3979a2a26feb97e888aefce2dceefb92bb5346b7b3a78c27",
}
EXPECTED_HF1_SHA = "e27f3b6005da21a4b3384bbffe9edf251e31ff87de3149f5cec88f82b7354d85"
TRAIN_VAL = ("train", "val")
ALL_SPLITS = ("train", "val", "test")
META_COLS = ("idx", "record", "event", "y", "q_mean",
             "q_adj", "contested", "n_votes", "votes_missing")


def _set_below_normal():
    """Set BelowNormal on Win64 using correctly typed HANDLE arguments."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32
        ]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        below_normal = 0x00004000
        if not kernel32.SetPriorityClass(
                kernel32.GetCurrentProcess(), below_normal):
            raise RuntimeError(
                "Cannot set BelowNormal CPU priority; "
                f"winerror={ctypes.get_last_error()}"
            )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_small(value):
    value = int(value)
    return "<10" if 0 < value < 10 else str(value)


def mask_below_ten(value):
    """Director 5d: censor every integer count smaller than ten."""
    return "<10" if int(value) < 10 else str(int(value))


def _clip_quality_groups(split_names, combinations):
    """Signal-only summaries; never read event labels or decisions."""
    return {
        (name, W, D, slot): {
            "stds": [], "clipped_windows": 0, "clipped_samples": 0,
        }
        for name in split_names
        for W, D in combinations
        for slot in SLOTS
    }


def _clip_quality_report(groups, combinations, split_names, fs,
                         low_std_threshold, max_low_fraction, checks):
    """Run-check 5c per (W,D,slot), and report-only 5d per split."""
    max_low_pct = 100.0 * max_low_fraction
    for W, D in combinations:
        for slot in SLOTS:
            valid = sum(len(groups[(name, W, D, slot)]["stds"])
                        for name in split_names)
            low = sum(sum(v < low_std_threshold for v in
                          groups[(name, W, D, slot)]["stds"])
                      for name in split_names)
            low_pct = 100.0 * low / valid if valid else float("nan")
            check(f"5c_low_std_W{W}_D{D}_{slot}",
                  valid > 0 and low_pct <= max_low_pct,
                  f"{low_pct:.6f}% (low={low}, valid={valid})",
                  f"<={max_low_pct:.6f}%", checks)

    print("=== REPORT P05-T02b-5d SIGNAL QUALITY ===", flush=True)
    print("scope: valid masked windows only; test SIGNALS only; "
          "NO test decisions/labels", flush=True)
    for name in split_names:
        for W, D in combinations:
            length = (W + D) * fs
            for slot in SLOTS:
                group = groups[(name, W, D, slot)]
                stds = group["stds"]
                n = len(stds)
                clip_window_pct = (100 * group["clipped_windows"] / n
                                   if n else float("nan"))
                clip_sample_pct = (100 * group["clipped_samples"] /
                                   (n * length) if n else float("nan"))
                quantiles = (np.percentile(stds, [0.1, 1, 5, 50])
                             if n else [float("nan")] * 4)
                low_half = sum(value < 0.5 for value in stds)
                print(
                    f"REPORT_5d: split={name}; W={W}; D={D}; slot={slot}; "
                    f"valid_windows={n}; "
                    f"clip_windows_pct={clip_window_pct:.6f}; "
                    f"clipped_samples_pct={clip_sample_pct:.6f}; "
                    f"std_p0.1={quantiles[0]:.6f}; "
                    f"std_p1={quantiles[1]:.6f}; "
                    f"std_p5={quantiles[2]:.6f}; "
                    f"std_p50={quantiles[3]:.6f}; "
                    f"std_lt_0.5={mask_below_ten(low_half)}",
                    flush=True,
                )
    print("=== END REPORT P05-T02b-5d ===", flush=True)


def load_cfg(path):
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    required = ("root", "raw_dir", "proc_dir", "project_dir",
                "t01b_counts_json", "common_dir", "seed", "n_jobs",
                "configs", "expected_events", "expected_records")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError("Missing config fields: " + ", ".join(missing))
    if int(cfg["seed"]) != 42 or not 1 <= int(cfg["n_jobs"]) <= 8:
        raise ValueError("Director requires seed 42 and <=8 workers")
    combinations = [(int(c["W"]), int(c["D"])) for c in cfg["configs"]]
    if combinations != [(10, 0), (20, 0), (10, 5), (10, 10)]:
        raise ValueError("Director-locked cache window configurations differ")
    locked_qc = {
        "preclip_max_abs_mean": 1e-4,
        "preclip_max_abs_std_error": 1e-4,
        "postclip_max_abs_mean": 0.05,
        "postclip_max_std": 1.0 + 1e-6,
        "postclip_low_std_threshold": 0.95,
        "postclip_low_std_max_fraction": 0.05,
    }
    for name, expected in locked_qc.items():
        if name not in cfg or not np.isclose(
                float(cfg[name]), expected, atol=1e-12, rtol=0.0):
            raise ValueError(f"Director-locked normalization QC differs: {name}")
    return cfg


def check(flag, condition, actual, expected, checks):
    ok = bool(condition)
    checks.append((flag, ok))
    print(f"CHECK {flag}: actual={actual}; expected={expected}; "
          f"{'PASS' if ok else 'FAIL'}", flush=True)


def report_failed_checks(checks, reason):
    failed = [name for name, ok in checks if not ok]
    print("=== REPORT P05-T02b ===", flush=True)
    print("status: FAIL", flush=True)
    print(f"hard_checks_passed: {sum(ok for _, ok in checks)}/{len(checks)}",
          flush=True)
    print("failed_checks: " + (", ".join(failed) or reason), flush=True)
    print("outputs: not published; sealed test labels not read",
          flush=True)
    print("=== END REPORT ===", flush=True)
    return 1


def read_keys(path, usecols):
    return pd.read_csv(
        path, usecols=list(usecols), dtype=str, keep_default_na=False
    )


def gated_decisions(path, metadata_columns, allowed_keys):
    """Two-pass physical row gate. No disallowed decision field is parsed."""
    keys = read_keys(path, metadata_columns)
    allowed = set(allowed_keys)
    keep = np.fromiter(
        (key in allowed for key in zip(keys["record"], keys["event"])),
        dtype=np.bool_, count=len(keys),
    )
    row_ids = set(np.flatnonzero(keep).tolist())
    decisions = pd.read_csv(
        path, usecols=["decision"], dtype=str, keep_default_na=False,
        skiprows=lambda index: index > 0 and (index - 1) not in row_ids,
    )
    if len(decisions) != len(row_ids):
        raise AssertionError("Gated decision-row count mismatch")
    selected = keys.loc[keep].reset_index(drop=True)
    selected["decision"] = decisions["decision"].to_numpy()
    return selected


def _vote_rows(vote_frame):
    grouped = defaultdict(list)
    for row in vote_frame.itertuples(index=False):
        grouped[(row.record, row.event)].append(
            (row.decision, row.is_adjudication)
        )
    return grouped


def _meta_from_allowed(split_pairs, label_csv, vote_csv):
    """Build train/val metadata. Test keys NEVER enter this function."""
    allowed_keys = list(zip(split_pairs["record"], split_pairs["event"]))
    label_rows = gated_decisions(
        label_csv, ("record", "event"), allowed_keys
    )
    votes = gated_decisions(
        vote_csv, ("record", "event", "is_adjudication"), allowed_keys
    )
    labels = {
        (r.record, r.event): r.decision.strip().lower()
        for r in label_rows.itertuples(index=False)
    }
    if len(labels) != len(allowed_keys):
        raise AssertionError("Train/val label keys are missing or repeated")
    grouped = _vote_rows(votes)
    entries = []
    for r in split_pairs.itertuples(index=False):
        key = (r.record, r.event)
        decision = labels[key]
        if decision not in ("true", "false"):
            raise ValueError("Unexpected train/val event label")
        y = int(decision == "true")
        q_mean, q_adj, contested, n_votes, votes_missing = soft_label(
            y, grouped.get(key, ())
        )
        entries.append((
            r.split, r.record, r.event, y, q_mean, q_adj,
            contested, n_votes, votes_missing,
        ))
    return pd.DataFrame(
        entries, columns=("split",) + META_COLS[1:]
    )


def _finalize_meta(frame):
    result = frame.drop(columns=["split"], errors="ignore").copy()
    result.insert(0, "idx", np.arange(len(result), dtype=np.int64))
    return result[list(META_COLS)]


def _seal_test_labels(keys, label_csv, vote_csv, output_file):
    """Process entry point. Suppress sensitive exception details."""
    try:
        _set_below_normal()
        _seal_test_labels_inner(keys, label_csv, vote_csv, output_file)
    except BaseException as exc:
        print(f"ISOLATED_SEAL_FAIL: {type(exc).__name__}; "
              "per-event details withheld", flush=True)
        os._exit(1)


def _seal_test_labels_inner(keys, label_csv, vote_csv, output_file):
    """Isolated worker: use test decisions ONLY for writing sealed parquet.

    This routine prints nothing and returns nothing about test labels.
    """
    wanted = list(zip(keys["record"], keys["event"]))
    labels = gated_decisions(label_csv, ("record", "event"), wanted)
    votes = gated_decisions(
        vote_csv, ("record", "event", "is_adjudication"), wanted
    )
    final = {
        (r.record, r.event): r.decision.strip().lower()
        for r in labels.itertuples(index=False)
    }
    if len(final) != len(wanted):
        raise ValueError("Test-label sealing key mismatch")
    grouped = _vote_rows(votes)
    result = []
    for idx, (record, event) in enumerate(wanted):
        decision = final[(record, event)]
        if decision not in ("true", "false"):
            raise ValueError("Invalid test event decision")
        y = int(decision == "true")
        q_mean, q_adj, contested, n_votes, missing = soft_label(
            y, grouped.get((record, event), ())
        )
        result.append((idx, y, q_mean, q_adj, contested, n_votes, missing))
    pd.DataFrame(
        result, columns=("idx", "y", "q_mean", "q_adj", "contested",
                         "n_votes", "votes_missing")
    ).to_parquet(output_file, index=False)
    # No further reads, calculations, summaries, or prints of test decisions.


def _process_event(task):
    """One worker event, three causal WFDB reads, four cached configurations."""
    _set_below_normal()
    import wfdb

    split, index, record, event, raw_dir, cfg_values = task
    fs, t0, context_s, nan_limit, min_std, clip, combinations = cfg_values
    base = str(Path(raw_dir) / "annotated_waveforms" / record / event)
    header = wfdb.rdheader(base)
    if float(header.fs) != fs or int(header.sig_len) < t0 + 10 * fs:
        raise ValueError("Waveform header violates fixed fs/t0/length")
    start = t0 - context_s * fs
    if start < 0:
        raise ValueError("Invalid causal context start")
    output = []
    for delta in sorted({d for _, d in combinations}):
        # sampto is exclusive; never read beyond this configuration's endpoint.
        waveform = wfdb.rdrecord(
            base, sampfrom=start, sampto=t0 + delta * fs, physical=True
        )
        samples = waveform.p_signal
        for W, D in combinations:
            if D != delta:
                continue
            x, mask, diag = preprocess_event(
                samples, header.sig_name, W, D, fs=fs, t0=t0,
                start_sample=start, context_s=context_s,
                nan_limit=nan_limit, min_std=min_std, clip=clip,
                return_diagnostics=True,
            )
            output.append((W, D, x, mask, diag))
        del waveform
    return split, index, output


def _synthetic_checks(checks, fs, t0):
    """Science checks 1 and 2, no real dataset access."""
    rng = np.random.default_rng(42)
    times = np.arange(90000, dtype=np.float64) / fs
    base = np.stack((
        np.sin(2 * np.pi * 5 * times),
        np.sin(2 * np.pi * 1.3 * times),
        np.sin(2 * np.pi * 1.7 * times),
        np.sin(2 * np.pi * 1.1 * times),
    ), axis=1)
    names = ["II", "V", "PLETH", "ABP"]
    unchanged = True
    for W, D in ((10, 0), (20, 0), (10, 5), (10, 10)):
        expected = preprocess_event(base, names, W, D, fs=fs, t0=t0)
        modified = base.copy()
        modified[t0 + D * fs:] = rng.normal(
            0, 100, size=modified[t0 + D * fs:].shape
        )
        observed = preprocess_event(modified, names, W, D, fs=fs, t0=t0)
        unchanged &= (
            np.array_equal(expected[0], observed[0])
            and np.array_equal(expected[1], observed[1])
        )
    check("1_future_leakage", unchanged, "array_equal", "array_equal", checks)

    fs = int(fs)
    t = np.arange(60 * fs, dtype=float) / fs
    mid = slice(20 * fs, 40 * fs)

    def gain(freq):
        s = np.sin(2 * np.pi * freq * t)
        filtered = filter_slot_segment(s, "ECG1", fs=fs)
        return float(2 * abs(np.mean(filtered[mid] * s[mid])))

    gain5, gain60 = gain(5), gain(60)
    attenuation_db = float(-20 * np.log10(max(gain60, 1e-30)))
    check("2_filter_5hz", gain5 >= 0.90,
          f"{gain5:.6f}", ">=0.90", checks)
    check("2_filter_60hz", attenuation_db >= 30.0,
          f"{attenuation_db:.2f} dB", ">=30 dB", checks)
    # HF1 K5 uses the presence of NON-ADJUDICATION ROWS, independent of
    # whether their decisions contribute numerical q_mean votes.
    invalid_only = soft_label(
        1, (("Reject", "False"), ("Save for Later", "False"))
    )
    no_rows = soft_label(1, ())
    label_semantics = (
        invalid_only == (1.0, 1.0, 0, 0, 0)
        and no_rows == (1.0, 1.0, 0, 0, 1)
    )
    check("8_synthetic_missing_key_semantics",
          label_semantics, label_semantics, True, checks)

    # Decision A: diagnostics expose statistics WITHOUT modifying D5 output.
    impulse = base.copy()
    impulse[t0 - fs, 0] += 1e5
    plain_x, plain_mask = preprocess_event(
        impulse, names, 10, 0, fs=fs, t0=t0
    )
    diag_x, diag_mask, diag = preprocess_event(
        impulse, names, 10, 0, fs=fs, t0=t0,
        return_diagnostics=True,
    )
    same_cache = (np.array_equal(plain_x, diag_x) and
                  np.array_equal(plain_mask, diag_mask))
    diagnostics_valid = all(
        entry is None if not plain_mask[idx]
        else (abs(entry[0]) <= 1e-4 and
              abs(entry[1] - 1.0) <= 1e-4 and
              abs(entry[2]) <= 0.05 and
              entry[3] <= 1.0 + 1e-6)
        for idx, entry in enumerate(diag)
    )
    check("5_director_A_synthetic_diagnostics",
          same_cache and diagnostics_valid,
          same_cache and diagnostics_valid, True, checks)


def _common_hashes(common_dir):
    common_dir = Path(common_dir)
    file_hashes = {}
    for filename, expected in EXPECTED_COMMON.items():
        source = common_dir / filename
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"Frozen common source mismatch: {filename}")
        file_hashes[filename] = actual
    files = sorted(
        (p for p in common_dir.rglob("*.py")
         if "__pycache__" not in p.parts),
        key=lambda p: p.relative_to(common_dir).as_posix(),
    )
    tree_hash = hashlib.sha256()
    for path in files:
        relative = path.relative_to(common_dir).as_posix().encode("utf-8")
        tree_hash.update(relative + b"\x00")
        tree_hash.update(bytes.fromhex(sha256(path)))
    return tree_hash.hexdigest(), file_hashes, len(files)


def _append_manifest(path, outputs):
    manifest = Path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = manifest.exists()
    now = datetime.now(timezone.utc).isoformat()
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not existing:
            writer.writerow(("path", "sha256", "n_samples", "timestamp_utc",
                             "step"))
        for file, n in outputs:
            writer.writerow((str(file), sha256(file), n, now, "P05-T02b"))


def _io_error_code(exc):
    """Return only OS error numbers; never reveal clinical file paths."""
    return (
        f"errno={getattr(exc, 'errno', None)}; "
        f"winerror={getattr(exc, 'winerror', None)}"
    )


def _cleanup_staging(staging, caches, extra_maps=()):
    """Close Windows memory mappings before discarding private cache files.

    On Windows, unlinking a mapped .npy can raise PermissionError, which used
    to mask the original waveform/cache exception in the enclosing finally.
    All maps here belong exclusively to this run's unpublished staging area.
    """
    maps = [mapped for pair in caches.values() for mapped in pair]
    maps.extend(mapped for mapped in extra_maps if mapped is not None)
    for mapped in maps:
        mapping = getattr(mapped, "_mmap", None)
        if mapping is not None and not mapping.closed:
            try:
                mapped.flush()
            except OSError as exc:
                print("CACHE_CLEANUP_FLUSH_WARNING: " +
                      _io_error_code(exc), flush=True)
            mapping.close()
    caches.clear()
    maps.clear()
    gc.collect()
    if staging.is_dir():
        shutil.rmtree(staging)


def _cleanup_self_test():
    """Verify an open memory-map cannot mask a deliberate synthetic error."""
    staging = Path(tempfile.mkdtemp(prefix="T02b_HF03_synthetic_"))
    maps = {}
    observed_type = None
    try:
        mapped = open_memmap(
            staging / "synthetic.npy", mode="w+", dtype=np.float32,
            shape=(1, 4, 8),
        )
        maps[("synthetic",)] = (mapped,)
        try:
            raise RuntimeError("synthetic original error")
        finally:
            _cleanup_staging(staging, maps)
    except Exception as exc:
        observed_type = type(exc).__name__
    finally:
        if staging.exists():
            try:
                _cleanup_staging(staging, maps)
            except OSError:
                pass
    return observed_type == "RuntimeError" and not staging.exists()


def run(config_path, self_test_only=False):
    started = time.perf_counter()
    _set_below_normal()
    cfg = load_cfg(config_path)
    raw = Path(cfg["raw_dir"])
    proc = Path(cfg["proc_dir"])
    project = Path(cfg["project_dir"])
    fs = int(cfg["fs_hz"])
    t0 = int(cfg["t0_sample"])
    seed = int(cfg["seed"])
    jobs = int(cfg["n_jobs"])
    combos = [(int(row["W"]), int(row["D"])) for row in cfg["configs"]]
    np.random.seed(seed)
    checks = []
    print("=== P05-T02b PRECHECKS ===", flush=True)
    _synthetic_checks(checks, fs, t0)
    if self_test_only:
        cleanup_ok = _cleanup_self_test()
        check("HF03_windows_memmap_cleanup", cleanup_ok,
              cleanup_ok, True, checks)
        passed = all(ok for _, ok in checks)
        print("SELF_TEST=" + ("PASS" if passed else "FAIL"), flush=True)
        print("=== REPORT P05-T02b-SELFTEST ===", flush=True)
        print("status: " + ("PASS" if passed else "FAIL"), flush=True)
        print("data: synthetic only; no RAW/PROC accessed", flush=True)
        print(f"checks: {sum(ok for _, ok in checks)}/{len(checks)}",
              flush=True)
        print(f"runtime_s: {time.perf_counter() - started:.2f}",
              flush=True)
        print("=== END REPORT ===", flush=True)
        return 0 if passed else 1
    if not all(ok for _, ok in checks):
        return report_failed_checks(checks, "synthetic checks failed")

    if sha256(cfg["t01b_hf1_script"]) != EXPECTED_HF1_SHA:
        raise ValueError("T01b-HF1 source SHA differs from approved code")
    tree_sha, common_shas, common_file_count = _common_hashes(
        cfg["common_dir"]
    )
    print(f"FROZEN_COMMON_SHA256={tree_sha}; files={common_file_count}", flush=True)
    with Path(cfg["t01b_counts_json"]).open(encoding="utf-8") as handle:
        prior = json.load(handle)
    if prior.get("status") != "PASS":
        raise ValueError("T01b-HF1 report not PASS")

    split = read_keys(raw / "benchmark_data_split.csv",
                      ("split", "record", "event"))
    split["split"] = split["split"].str.strip().str.lower().replace({
        "validation": "val", "valid": "val"
    })
    split = split.sort_values(
        ["split", "record", "event"], kind="stable"
    ).reset_index(drop=True)
    if not split["split"].isin(ALL_SPLITS).all():
        raise ValueError("Unknown split")
    for name in ALL_SPLITS:
        subset = split.loc[split["split"] == name]
        check("3_event_count_" + name,
              len(subset) == int(cfg["expected_events"][name]),
              len(subset), cfg["expected_events"][name], checks)
        check("3_record_count_" + name,
              subset["record"].nunique() ==
              int(cfg["expected_records"][name]),
              subset["record"].nunique(),
              cfg["expected_records"][name], checks)
    if len(split) != len(split.drop_duplicates(["record", "event"])):
        raise ValueError("Duplicate split event keys")
    split_keys = {
        name: set(split.loc[split["split"] == name, "record"])
        for name in ALL_SPLITS
    }
    distinct = not (
        split_keys["train"] & split_keys["val"]
        or split_keys["train"] & split_keys["test"]
        or split_keys["val"] & split_keys["test"]
    )
    check("3_official_record_disjoint", distinct, distinct, True, checks)

    keys_only = read_keys(raw / "event_labels.csv",
                          ("record", "event"))
    vote_keys_only = read_keys(
        raw / "event_label_per_annotator.csv",
        ("record", "event", "is_adjudication")
    )
    split_event_keys = set(zip(split["record"], split["event"]))
    labels_event_keys = set(zip(keys_only["record"], keys_only["event"]))
    check("3_keys_match", split_event_keys == labels_event_keys,
          len(split_event_keys & labels_event_keys),
          len(split_event_keys), checks)

    trainval_pairs = split.loc[split["split"].isin(TRAIN_VAL)]
    allowed_meta = _meta_from_allowed(
        trainval_pairs, raw / "event_labels.csv",
        raw / "event_label_per_annotator.csv"
    )
    train = allowed_meta.loc[allowed_meta["split"] == "train"].copy()
    val = allowed_meta.loc[allowed_meta["split"] == "val"].copy()

    record_y = train.groupby("record", sort=True)["y"].max()
    fit_records, calib_records = train_test_split(
        record_y.index.to_numpy(), test_size=float(cfg["calib_record_fraction"]),
        random_state=seed, stratify=record_y.to_numpy(dtype=int),
    )
    calib_set = set(calib_records)
    fit_set = set(fit_records)
    n_calib = len(calib_set)
    n_pos_calib = int(record_y.loc[calib_records].sum())
    primary_alpha = 0.01 if n_pos_calib >= 300 else 0.02
    check("7_calib_records",
          n_calib == int(cfg["expected_records"]["calib"]),
          n_calib, cfg["expected_records"]["calib"], checks)
    check("7_calib_positive_records",
          n_pos_calib == int(cfg["expected_calib_positive_records"]),
          n_pos_calib, cfg["expected_calib_positive_records"], checks)
    check("7_primary_alpha", primary_alpha == 0.02,
          primary_alpha, 0.02, checks)
    fit = _finalize_meta(
        train.loc[train["record"].isin(fit_set)]
    )
    calib = _finalize_meta(
        train.loc[train["record"].isin(calib_set)]
    )
    val_meta = _finalize_meta(val)
    test_pairs = split.loc[split["split"] == "test", ["record", "event"]]
    test_meta = test_pairs.copy().reset_index(drop=True)
    test_meta.insert(0, "idx", np.arange(len(test_meta), dtype=np.int64))
    metadata = {
        "fit": fit, "calib": calib, "val": val_meta, "test": test_meta
    }
    actual_train = len(fit) + len(calib)
    check("3_fit_plus_calib", actual_train == len(train),
          actual_train, len(train), checks)
    check("3_fit_calib_disjoint", not (fit_set & calib_set),
          len(fit_set & calib_set), 0, checks)

    difference_pp = 100 * abs(float(fit["y"].mean()) -
                              float(calib["y"].mean()))
    check("6_positive_prevalence_fit_calib",
          difference_pp <= float(cfg["max_event_y_prevalence_diff_pp"]),
          f"{difference_pp:.4f} pp",
          f"<={cfg['max_event_y_prevalence_diff_pp']} pp", checks)

    available_nonadj = set(
        (r.record, r.event)
        for r in vote_keys_only.itertuples(index=False)
        if str(r.is_adjudication).strip().lower() not in ("true", "1")
    )
    trainval_event_keys = set(zip(
        trainval_pairs["record"], trainval_pairs["event"]
    ))
    missing_tv_from_keys = len(trainval_event_keys - available_nonadj)
    missing_tv = int(fit["votes_missing"].sum() +
                     calib["votes_missing"].sum() +
                     val_meta["votes_missing"].sum())
    check("8_votes_missing_train_val",
          missing_tv == int(cfg["expected_votes_missing_train_val"]),
          missing_tv, cfg["expected_votes_missing_train_val"], checks)
    check("8_missing_flags_match_HF1_keys",
          missing_tv == missing_tv_from_keys,
          missing_tv, missing_tv_from_keys, checks)
    all_safe = pd.concat((fit, calib, val_meta), ignore_index=True)
    missing_part = all_safe.loc[all_safe["votes_missing"] == 1]
    fallback_ok = (
        (missing_part["q_mean"] == missing_part["y"]).all()
        and (missing_part["q_adj"] == missing_part["y"]).all()
        and (missing_part["contested"] == 0).all()
    )
    check("8_missing_vote_fallback", fallback_ok, fallback_ok, True, checks)

    test_event_keys = set(zip(test_meta["record"], test_meta["event"]))
    test_missing_keys = len(test_event_keys - available_nonadj)
    check("4_test_meta_unlabeled",
          list(test_meta.columns) == ["idx", "record", "event"],
          list(test_meta.columns), ["idx", "record", "event"], checks)
    if not all(ok for _, ok in checks):
        print("T02B PRE-CACHE HARD CHECKS FAIL; outputs NOT saved", flush=True)
        return report_failed_checks(checks, "pre-cache checks failed")

    proc.mkdir(parents=True, exist_ok=True)
    final_names = [
        f"{name}_W{W}_D{D}_{tail}.npy"
        for name in metadata for W, D in combos for tail in ("x", "mask")
    ] + [f"{name}_meta.parquet" for name in metadata] + [
        "test_labels.parquet"
    ]
    if any((proc / name).exists() for name in final_names):
        raise FileExistsError(
            "T02b outputs already exist; refusing to overwrite sealed files"
        )
    locked_file = project / "configs" / "locked_design.yaml"
    if locked_file.exists():
        raise FileExistsError("locked_design.yaml already exists")

    staging = Path(tempfile.mkdtemp(prefix="T02b_staging_", dir=proc))
    print("CACHE_STAGING=CREATED; exact private path withheld", flush=True)
    stage = "cache_allocation"
    caches = {}
    try:
        for name, frame in metadata.items():
            count = len(frame)
            for W, D in combos:
                stage = f"memmap_{name}_W{W}_D{D}"
                key = (name, W, D)
                caches[key] = (
                    open_memmap(
                        staging / f"{name}_W{W}_D{D}_x.npy",
                        mode="w+", dtype=np.float32,
                        shape=(count, 4, (W + D) * fs),
                    ),
                    open_memmap(
                        staging / f"{name}_W{W}_D{D}_mask.npy",
                        mode="w+", dtype=np.bool_, shape=(count, 4),
                    ),
                )
            stage = f"metadata_write_{name}"
            frame.to_parquet(staging / f"{name}_meta.parquet", index=False)
            print(f"CACHE_STAGE=prepared_{name}; "
                  "patient identifiers withheld", flush=True)

        cfg_values = (
            fs, t0, int(cfg["filter_context_s"]),
            float(cfg["nan_missing_fraction"]),
            float(cfg["min_filtered_std"]),
            float(cfg["clip_z"]), combos,
        )
        items = (
            (name, int(row.idx), row.record, row.event, str(raw), cfg_values)
            for name, frame in metadata.items()
            for row in frame[["idx", "record", "event"]].itertuples(index=False)
        )
        slot_counts = {
            name: np.zeros(4, dtype=np.int64) for name in ALL_SPLITS
        }
        processed = 0
        quality_bad = 0
        preclip_max_abs_mean = 0.0
        preclip_max_abs_std_error = 0.0
        postclip_max_abs_mean = 0.0
        postclip_max_std = 0.0
        quality_groups = _clip_quality_groups(metadata, combos)
        largest_abs = 0.0
        read_errors = 0
        stage = "waveform_worker_start"
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            iterator = iter(items)
            pending = {}

            def enqueue():
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                future = executor.submit(_process_event, item)
                pending[future] = True

            for _ in range(2 * jobs):
                enqueue()
            while pending:
                future = next(as_completed(pending))
                pending.pop(future)
                try:
                    name, index, outputs = future.result()
                except Exception as exc:
                    # Privacy-preserving: never print patient/file identifiers.
                    read_errors += 1
                    print("WAVEFORM_WORKER_ERROR: "
                          f"type={type(exc).__name__}; "
                          f"{_io_error_code(exc)}; "
                          "patient identifiers withheld", flush=True)
                    raise RuntimeError(
                        "Waveform worker failure (one event); "
                        "no patient identifiers disclosed"
                    ) from None
                for W, D, x, mask, diag in outputs:
                    xmap, mmap = caches[(name, W, D)]
                    xmap[index] = x
                    mmap[index] = mask
                    if (W, D) == (10, 0):
                        old_split = "train" if name in ("fit", "calib") else name
                        slot_counts[old_split] += mask.astype(np.int64)
                    if not np.all(np.isfinite(x)) or np.any(np.isnan(x)):
                        quality_bad += 1
                    largest_abs = max(largest_abs, float(np.max(np.abs(x))))
                    if largest_abs > 10.0:
                        quality_bad += 1
                    if np.any(x[~mask] != 0.0):
                        quality_bad += 1
                    for slot_id, slot in enumerate(SLOTS):
                        if not mask[slot_id]:
                            if diag[slot_id] is not None:
                                quality_bad += 1
                            continue
                        if diag[slot_id] is None:
                            quality_bad += 1
                            continue
                        pre_mean, pre_std, post_mean, post_std, n_clipped = (
                            diag[slot_id]
                        )
                        if not np.all(np.isfinite((pre_mean, pre_std,
                                                post_mean, post_std))):
                            quality_bad += 1
                            continue
                        preclip_max_abs_mean = max(
                            preclip_max_abs_mean, abs(pre_mean)
                        )
                        preclip_max_abs_std_error = max(
                            preclip_max_abs_std_error, abs(pre_std - 1.0)
                        )
                        postclip_max_abs_mean = max(
                            postclip_max_abs_mean, abs(post_mean)
                        )
                        postclip_max_std = max(postclip_max_std, post_std)
                        if not 0 <= n_clipped <= x.shape[1]:
                            quality_bad += 1
                        group = quality_groups[(name, W, D, slot)]
                        group["stds"].append(post_std)
                        group["clipped_windows"] += int(n_clipped > 0)
                        group["clipped_samples"] += n_clipped
                processed += 1
                if processed % 500 == 0:
                    print(f"PROGRESS={processed}/{len(split)}", flush=True)
                enqueue()

        stage = "memmap_flush_and_quality_checks"
        for xmap, mmap in caches.values():
            xmap.flush()
            mmap.flush()
        del xmap, mmap
        caches.clear()
        gc.collect()
        check("5_array_finite_bounded",
              quality_bad == 0 and largest_abs <= 10.0,
              f"invalid={quality_bad}, maxabs={largest_abs:.6f}",
              "invalid=0,maxabs<=10", checks)
        check("5a_preclip_mean",
              preclip_max_abs_mean <= cfg["preclip_max_abs_mean"],
              f"{preclip_max_abs_mean:.9f}",
              f"<={cfg['preclip_max_abs_mean']}", checks)
        check("5a_preclip_std",
              preclip_max_abs_std_error <= cfg["preclip_max_abs_std_error"],
              f"{preclip_max_abs_std_error:.9f}",
              f"<={cfg['preclip_max_abs_std_error']}", checks)
        check("5b_postclip_mean",
              postclip_max_abs_mean <= cfg["postclip_max_abs_mean"],
              f"{postclip_max_abs_mean:.9f}",
              f"<={cfg['postclip_max_abs_mean']}", checks)
        check("5b_postclip_std",
              postclip_max_std <= cfg["postclip_max_std"],
              f"{postclip_max_std:.9f}",
              f"<={cfg['postclip_max_std']}", checks)
        check("3_events_processed", processed == len(split),
              processed, len(split), checks)
        check("3_waveform_errors", read_errors == 0,
              read_errors, 0, checks)

        # Director decision A (10 Nov 2026): clipped standard deviation may
        # decrease. Never normalize again or mask an otherwise valid slot.
        _clip_quality_report(
            quality_groups, combos, tuple(metadata), fs,
            cfg["postclip_low_std_threshold"],
            cfg["postclip_low_std_max_fraction"], checks,
        )

        present_pct = {}
        for name in ALL_SPLITS:
            counts = slot_counts[name]
            denominator = int(cfg["expected_events"][name])
            got = {
                slot: round(100 * int(counts[index]) / denominator, 4)
                for index, slot in enumerate(SLOTS)
            }
            expected = prior["E_slots"][name]["present_pct"]
            present_pct[name] = got
            check("7_slots_match_hf1_" + name,
                  got == expected, got, expected, checks)

        if not all(ok for _, ok in checks):
            print("T02B SCIENTIFIC CHECK FAIL; results NOT published", flush=True)
            return report_failed_checks(checks, "scientific checks failed")

        # All statistics are complete. Only NOW may a separate process read
        # the test decisions, solely to seal their file without reporting them.
        stage = "isolated_test_label_seal"
        ctx = mp.get_context("spawn")
        seal_proc = ctx.Process(
            target=_seal_test_labels,
            args=(test_meta[["record", "event"]],
                  str(raw / "event_labels.csv"),
                  str(raw / "event_label_per_annotator.csv"),
                  str(staging / "test_labels.parquet")),
        )
        seal_proc.start()
        seal_proc.join()
        if seal_proc.exitcode != 0:
            raise RuntimeError("Isolated test-label sealing failed")
        if not (staging / "test_labels.parquet").is_file():
            raise RuntimeError("Isolated label seal file is absent")
        sealed_sha = sha256(staging / "test_labels.parquet")
        print(f"CHECK 9_test_label_seal: sha256={sealed_sha}; "
              "test decision counts NOT computed | PASS", flush=True)
        checks.append(("9_test_label_seal", True))

        # Preregistration includes SHA256 of every metadata file and the
        # sealed label file. The label parquet is never read after sealing.
        stage = "locked_design_prepare"
        meta_sha = {
            f"{name}_meta.parquet": sha256(staging / f"{name}_meta.parquet")
            for name in metadata
        }
        meta_sha["test_labels.parquet"] = sealed_sha
        feasible_grid = [a for a in (0.005, 0.01, 0.02, 0.05)
                         if a >= 1.0 / (n_pos_calib + 1)]
        locked = {
            "step": "P05-T02b",
            "preregistration_status": "LOCKED_AFTER_DIRECTOR_ACCEPTANCE",
            "G0": "CLOSED",
            "seed": seed,
            "calibration": {
                "unit": "record",
                "algorithm": "sklearn.train_test_split",
                "test_size": float(cfg["calib_record_fraction"]),
                "stratify": "record contains >=1 y=1",
                "n_calib_records": n_calib,
                "n_positive_calib_records": n_pos_calib,
                "crc_minimum_alpha": 1.0 / (n_pos_calib + 1),
                "primary_alpha": primary_alpha,
                "alpha_grid": feasible_grid,
                "alpha_0_005": ("feasible" if 0.005 in feasible_grid
                                else "infeasible: 1/(n+1)>0.005"),
                "alpha_rule": "n>=300 => 0.01; otherwise 0.02",
            },
            "cache": {
                "fs_hz": fs,
                "t0_sample": t0,
                "context_s": cfg["filter_context_s"],
                "configurations": [{"W": W, "D": D} for W, D in combos],
                "slots": list(SLOTS),
                "dtype_x": "float32",
                "dtype_mask": "bool",
                "clip_z": 10.0,
                "nan_threshold": cfg["nan_missing_fraction"],
                "min_filtered_std": cfg["min_filtered_std"],
                "causal_cut_before_interpolation_filter": True,
            },
            "normalization_qc_director_decision_2026_11_10": {
                "option": "A",
                "method": "D5 unchanged: per-window z-score, then single clip [-10,10]",
                "slot_presence_rule": "D6 unchanged; no exclusion using postclip std",
                "5a_hard_before_clip": {
                    "max_abs_mean": cfg["preclip_max_abs_mean"],
                    "max_abs_std_minus_one": cfg["preclip_max_abs_std_error"],
                    "scope": "every mask=1 window",
                },
                "5b_hard_after_clip": {
                    "max_std": cfg["postclip_max_std"],
                    "max_abs_mean": cfg["postclip_max_abs_mean"],
                    "scope": "every mask=1 window",
                },
                "5c_hard_after_clip": {
                    "std_below": cfg["postclip_low_std_threshold"],
                    "max_fraction": cfg["postclip_low_std_max_fraction"],
                    "grouping": "each configuration (W,D) x slot; pooled fit/calib/val/test",
                    "denominator": "mask=1 windows only",
                },
                "5d_report_only": {
                    "grouping": "configuration x slot x fit/calib/val/test",
                    "clip_windows_fraction": True,
                    "clipped_samples_fraction": True,
                    "postclip_std_percentiles": [0.1, 1, 5, 50],
                    "postclip_std_below_0_5_count": True,
                    "redact_counts_below": 10,
                    "test_statistics": "waveforms only; no test labels",
                },
                "observed_max_preclip_abs_mean": preclip_max_abs_mean,
                "observed_max_preclip_abs_std_error": preclip_max_abs_std_error,
                "observed_max_postclip_abs_mean": postclip_max_abs_mean,
                "observed_max_postclip_std": postclip_max_std,
            },
            "labels": {
                "y": "event_labels.csv decision True=1 False=0",
                "valid_nonadjudicator_votes":
                    {"True": 1.0, "False": 0.0, "Uncertain": 0.5},
                "ignored_votes": ["Reject", "Save for Later"],
                "q_mean":
                    "mean valid non-adjudication votes; fallback y if none retained",
                "q_adj": "(sum valid votes+y)/(n_votes+1) if adjudicated; else q_mean",
                "contested": "disagreement in True/False votes OR adjudication",
                "HF1_K5": "NO; do not infer any key mapping",
                "votes_missing":
                    "no non-adjudication annotation rows after exact HF1 key matching",
                "n_votes": "count of retained True/False/Uncertain votes",
                "ignored_ballots":
                    "Reject/Save for Later excluded from q; existing rows do not imply missing key",
                "missing_fallback": "q_mean=q_adj=y; contested=0",
                "votes_missing_train_val": missing_tv,
                "missing_use": "retain fit/calib/eval; exclude H2",
                "test_seal": "test labels never opened before T05b",
            },
            "stage_gates": {
                "G1": {"val_AUROC_min": 0.92, "val_Challenge_Score_min": 75},
                "G2": {"val_B_minus_A_min": 0.02,
                       "B_better_than_A_seeds_min": 4,
                       "n_seeds": 5},
            },
            "provenance_sha256": {
                "t01b_hf1_script": EXPECTED_HF1_SHA,
                "t02a_test_script": common_shas["tests/test_common.py"],
                "common_source_tree": tree_sha,
                "common_tree_hash_method":
                    "sorted relative *.py paths, each path+NUL+32-byte file sha",
                "common_files": common_shas,
                "meta_files": meta_sha,
                "t02b_script": sha256(__file__),
                "preprocess_lib_script": sha256(
                    Path(__file__).with_name("preprocess_lib.py")
                ),
                "t02b_config": sha256(config_path),
            },
            "test_label_policy": {
                "allowed_read": "T05b ONLY",
                "t02b_exception": "isolated creation of sealed file",
                "no_test_label_statistics": True,
                "no_test_decision_print": True,
            },
        }
        project.joinpath("configs").mkdir(parents=True, exist_ok=True)
        pending_lock = staging / "locked_design.yaml"
        with pending_lock.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(locked, handle, sort_keys=False, allow_unicode=True)

        size_cache_bytes = sum(
            path.stat().st_size for path in staging.glob("*.npy")
        )
        # Publish only after every scientific check, then register outputs.
        stage = "cache_publish"
        published = []
        for name in final_names:
            src = staging / name
            dst = proc / name
            if not src.is_file() or dst.exists():
                raise RuntimeError("Refusing unsafe output publication")
            os.replace(src, dst)
            population = next(
                (len(frame) for split_name, frame in metadata.items()
                 if name.startswith(split_name + "_")), 0
            )
            published.append((dst, population))
        if locked_file.exists():
            raise FileExistsError("Refusing to replace locked_design.yaml")
        os.replace(pending_lock, locked_file)
        stage = "manifest_append"
        _append_manifest(project / "manifest.csv",
                         published + [(locked_file, 0)])
        elapsed = time.perf_counter() - started
        print("=== REPORT P05-T02b ===", flush=True)
        print("status: PASS", flush=True)
        print(f"pc: reference-workstation | python: {platform.python_version()} | gpu: not used",
              flush=True)
        print(f"cpu: workers={jobs}; priority=BelowNormal (runner)",
              flush=True)
        print(f"runtime_s: {elapsed:.1f}", flush=True)
        print(f"script: t02b_cache.py | sha256_8={sha256(__file__)[:8]}",
              flush=True)
        for name, frame in metadata.items():
            print(f"{name}: events={len(frame)}; records={frame['record'].nunique()}",
                  flush=True)
        print(f"calib: n_positive_records={n_pos_calib}; "
              f"alpha_primary={primary_alpha}", flush=True)
        for name in ALL_SPLITS:
            print(f"slots_{name}_W10D0: {present_pct[name]}", flush=True)
        print(f"votes_missing_train_val: {missing_tv}", flush=True)
        print(f"votes_missing_test_keys_only: {mask_small(test_missing_keys)}",
              flush=True)
        print(f"cache_gib: {size_cache_bytes / (1024 ** 3):.4f}",
              flush=True)
        print(f"common_tree_sha256: {tree_sha}", flush=True)
        print(f"test_labels_sha256: {sealed_sha}", flush=True)
        print(f"locked_design_sha256: {sha256(locked_file)}", flush=True)
        print(f"checks: {sum(ok for _, ok in checks)}/{len(checks)} PASS",
              flush=True)
        print(f"outputs: PROC cache/meta/sealed labels; "
              "P/configs/locked_design.yaml; P/manifest.csv", flush=True)
        print("warnings: 0; test decision statistics: NOT computed", flush=True)
        print("=== END REPORT ===", flush=True)
        return 0
    except Exception as exc:
        print(f"CACHE_FAILURE_STAGE={stage}; "
              f"cause_type={type(exc).__name__}; "
              f"{_io_error_code(exc)}; "
              "patient identifiers withheld", flush=True)
        raise
    finally:
        # Always close mappings BEFORE removing staging on Windows.
        # Keep the original error if staging cleanup has a second failure.
        # Never touch published, project, or raw files.
        try:
            _cleanup_staging(staging, caches)
        except Exception as exc:
            print("CACHE_CLEANUP_WARNING: "
                  f"type={type(exc).__name__}; "
                  f"{_io_error_code(exc)}; "
                  "private staging may need later cleanup", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        return run(args.config, self_test_only=args.self_test)
    except Exception as exc:
        # No traceback: WFDB path exceptions could disclose record IDs.
        print("=== REPORT P05-T02b ===", flush=True)
        print("status: FAIL", flush=True)
        print(f"error_type: {type(exc).__name__}", flush=True)
        print("detail: execution stopped; see hard-check aggregate lines",
              flush=True)
        print("test labels: no statistics disclosed", flush=True)
        print("=== END REPORT ===", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
