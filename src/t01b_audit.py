"""P05-T01b: aggregate-only, read-only audit of VTaC 1.1."""

import os

# Set limits before importing numerical libraries.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import argparse
import csv
import gc
import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml
from scipy import signal
from sklearn.model_selection import train_test_split


EXPECTED = {
    "labeled_events": 5037,
    "labeled_records": 2260,
    "split_events_train": 4060,
    "split_events_val": 495,
    "split_events_test": 482,
    "split_records_train": 1808,
    "split_records_val": 226,
    "split_records_test": 226,
    "annotated_pairs": 5037,
    "waveform_pairs": 17131,
}
SPLITS = ("train", "val", "test")
SLOTS = ("ECG1", "ECG2", "PPG", "ABP")
ECG_LIMB = ("II", "I", "III", "AVR", "AVL", "AVF")
ECG_CHEST = ("V", "MCL", "V1", "V2", "V3", "V4", "V5", "V6")


def small(n):
    n = int(n)
    return "<10" if 0 < n < 10 else str(n)


def counted(counter):
    return {str(k): small(v) for k, v in sorted(counter.items(), key=lambda x: str(x[0]))}


def read_csv(path, columns):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if not set(columns).issubset(frame.columns):
        raise ValueError("Required CSV columns are absent")
    return frame


def read_allowed_decisions(path, key_columns, allowed_keys):
    """Read decision ONLY from allowed physical CSV rows, preserving row order.

    The first pass loads key metadata, not decisions. The second pass skips
    every disallowed CSV row at parse time. Never load test vote decisions.
    """
    keys_only = pd.read_csv(
        path, usecols=list(key_columns), dtype=str, keep_default_na=False,
    )
    allowed = set(allowed_keys)
    mask = [pair in allowed for pair in zip(keys_only["record"], keys_only["event"])]
    row_indices = {index for index, selected in enumerate(mask) if selected}
    expected = sum(mask)
    # skiprows uses 0 for the header, 1 for the first data row, etc.
    decisions = pd.read_csv(
        path, usecols=["decision"], dtype=str, keep_default_na=False,
        skiprows=lambda index: index > 0 and (index - 1) not in row_indices,
    )
    actual = len(decisions)
    if actual != expected:
        raise AssertionError(
            f"HF1 decision-row gate FAIL: actual={small(actual)}, "
            f"expected={small(expected)}"
        )
    selected = keys_only.loc[mask].copy().reset_index(drop=True)
    selected["decision"] = decisions["decision"].to_numpy()
    print(f"HF1 {Path(path).name} decision-row gate PASS: "
          f"actual={small(actual)}, expected={small(expected)}")
    del decisions, keys_only
    return selected


def keys(frame):
    return set(zip(frame["record"], frame["event"]))


def normalize_split(value):
    value = str(value).strip().lower()
    return {"validation": "val", "valid": "val"}.get(value, value)


def scan_pairs(folder):
    headers = {
        (p.relative_to(folder).parent.as_posix(), p.stem)
        for p in folder.rglob("*.hea")
    }
    data = {
        (p.relative_to(folder).parent.as_posix(), p.stem)
        for p in folder.rglob("*.dat")
    }
    return headers, data, headers & data


def check(name, actual, expected, hard_checks):
    passed = actual == expected
    hard_checks.append((name, passed))
    print(f"CHECK {name}: actual={small(actual)} | expected={expected} | {'PASS' if passed else 'FAIL'}")


def normalized_channel(name):
    text = "".join(ch for ch in str(name).upper() if ch.isalnum())
    return text[3:] if text.startswith("ECG") else text


def select_slots(channel_names):
    normalized = [normalized_channel(name) for name in channel_names]

    def first(candidates, excluded=None):
        for candidate in candidates:
            for idx, name in enumerate(normalized):
                if name == candidate and idx != excluded:
                    return idx
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


def preprocess_presence(raw_60s, channel_names, fs, window_s):
    """Pure D1-D6 pipeline for W=10, delta=0; returns only four mask bits."""
    fs = float(fs)
    raw_60s = np.asarray(raw_60s, dtype=np.float64)
    needed = int(round(window_s * fs))
    if raw_60s.ndim != 2 or raw_60s.shape[0] != int(round(60 * fs)):
        raise ValueError("Unexpected input segment length")
    if needed < 1:
        raise ValueError("Invalid window duration")

    notch_b, notch_a = signal.iirnotch(60, Q=30, fs=fs)
    notch_sos = signal.tf2sos(notch_b, notch_a)
    filters = {
        "ECG1": signal.butter(4, (1, 30), btype="bandpass", fs=fs, output="sos"),
        "ECG2": signal.butter(4, (1, 30), btype="bandpass", fs=fs, output="sos"),
        "PPG": signal.butter(4, (0.5, 5), btype="bandpass", fs=fs, output="sos"),
        "ABP": signal.butter(4, 16, btype="lowpass", fs=fs, output="sos"),
    }
    selected = select_slots(channel_names)
    presence = []
    for slot in SLOTS:
        index = selected[slot]
        if index is None:
            presence.append(0)
            continue
        original = raw_60s[:, index]
        original_window = original[-needed:]
        if np.mean(~np.isfinite(original_window)) > 0.20:
            presence.append(0)
            continue

        finite = np.isfinite(original)
        if finite.any():
            samples = np.arange(original.size)
            cleaned = np.interp(samples, samples[finite], original[finite])
        else:
            cleaned = np.zeros(original.size, dtype=np.float64)
        cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
        filtered = signal.sosfiltfilt(notch_sos, cleaned)
        filtered = signal.sosfiltfilt(filters[slot], filtered)
        cropped = filtered[-needed:]
        sigma = float(np.std(cropped, ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-6:
            presence.append(0)
            continue
        normalized = np.clip((cropped - np.mean(cropped)) / sigma, -10, 10)
        presence.append(int(np.all(np.isfinite(normalized))))
    return tuple(presence)


def count_headers(root, pairs, record_split):
    by_split = defaultdict(lambda: {
        "fs": Counter(), "sig_len": Counter(), "channels_n": Counter(),
        "channel_names": Counter(), "read_errors": 0,
    })
    for record, event in sorted(pairs):
        category = record_split.get(record, "unassigned")
        entry = by_split[category]
        try:
            header = wfdb.rdheader(str(root / record / event))
            entry["fs"][f"{float(header.fs):g}"] += 1
            entry["sig_len"][str(int(header.sig_len))] += 1
            entry["channels_n"][str(int(header.n_sig))] += 1
            entry["channel_names"].update(map(str, header.sig_name))
        except Exception:
            # Exceptions may contain pseudonymous record/event names: never print them.
            entry["read_errors"] += 1
    return {
        group: {k: counted(v) if isinstance(v, Counter) else small(v)
                for k, v in entry.items()}
        for group, entry in sorted(by_split.items())
    }


def count_slot_presence(annotated_dir, split_frame, t0, window_s):
    per_split = defaultdict(lambda: {
        "n": 0, "slot_present": Counter(), "combinations": Counter(), "read_errors": 0,
    })
    # One event at a time: do not retain or save any waveform data.
    for row in split_frame.itertuples(index=False):
        group = row.split
        entry = per_split[group]
        entry["n"] += 1
        try:
            base = str(annotated_dir / row.record / row.event)
            header = wfdb.rdheader(base)
            fs = float(header.fs)
            if fs != 250:
                raise ValueError("The locked 250 Hz sampling rate is not satisfied")
            start = t0 - 60 * 250
            if start < 0 or int(header.sig_len) < t0:
                raise ValueError("Segment is shorter than required")
            waveform = wfdb.rdrecord(base, sampfrom=start, sampto=t0, physical=True)
            mask = preprocess_presence(waveform.p_signal, header.sig_name, fs, window_s)
            del waveform
            entry["slot_present"].update(slot for slot, bit in zip(SLOTS, mask) if bit)
            entry["combinations"]["".join(map(str, mask))] += 1
        except Exception:
            entry["read_errors"] += 1
            entry["combinations"]["ERROR"] += 1

    result = {}
    for group in SPLITS:
        entry = per_split[group]
        n = entry["n"]
        popular = entry["combinations"].most_common(6)
        result[group] = {
            "events": n,
            "present_pct": {
                slot: round(100 * entry["slot_present"][slot] / n, 4) if n else None
                for slot in SLOTS
            },
            "top6_combinations": [
                {"mask": mask, "n": small(count),
                 "pct": round(100 * count / n, 4) if n else None}
                for mask, count in popular
            ],
            "read_errors": small(entry["read_errors"]),
        }
    return result


def summarize_votes(safe_labels, votes):
    # Keep only train/val votes; no test vote is evaluated or reported.
    safe_keys = safe_labels[["record", "event"]].drop_duplicates()
    votes = votes.merge(safe_keys, on=["record", "event"], how="inner", validate="many_to_one")
    grouped = {key: group for key, group in votes.groupby(["record", "event"], sort=False)}
    result = {}
    record_flags = {}
    for split in ("train", "val"):
        subset = safe_labels.loc[safe_labels["split"] == split]
        decisions = Counter()
        records_positive = set()
        n_votes = Counter()
        qm = Counter({"0": 0, "0.25": 0, "0.5": 0, "0.75": 0, "1": 0, "other": 0})
        other = Counter({"Uncertain": 0, "Reject": 0, "Save for Later": 0})
        adjudications = 0
        contested_total = 0
        for row in subset.itertuples(index=False):
            key = (row.record, row.event)
            y = int(row.decision == "True")
            decisions[row.decision] += 1
            if y:
                records_positive.add(row.record)
            group = grouped.get(key)
            if group is None:
                vote_rows = []
            else:
                vote_rows = list(group[["decision", "is_adjudication"]].itertuples(index=False, name=None))
            n_votes[len(vote_rows)] += 1
            retained = []
            binary = set()
            has_adjudication = False
            for decision, is_adjudication in vote_rows:
                is_adj = str(is_adjudication).strip().lower() in ("true", "1")
                has_adjudication |= is_adj
                decision = str(decision).strip().lower()
                if is_adj:
                    continue
                if decision == "true":
                    binary.add(1)
                    retained.append(1.0)
                elif decision == "false":
                    binary.add(0)
                    retained.append(0.0)
                elif decision == "uncertain":
                    retained.append(0.5)
                    other["Uncertain"] += 1
                elif decision in ("reject", "save for later"):
                    other["Reject" if decision == "reject" else "Save for Later"] += 1
                else:
                    other["other"] += 1
            q_mean = float(np.mean(retained)) if retained else float(y)
            q_adj = ((sum(retained) + y) / (len(retained) + 1)) if has_adjudication else q_mean
            contested = bool(len(binary) == 2 or has_adjudication)
            if has_adjudication:
                adjudications += 1
            contested_total += contested
            known = (0.0, 0.25, 0.5, 0.75, 1.0)
            bucket = next((f"{v:g}" for v in known if abs(q_mean - v) < 1e-10), "other")
            qm[bucket] += 1
            # q_adj is deliberately computed, but no per-event score or label is saved.
            if not 0.0 <= q_adj <= 1.0:
                raise ValueError("Invalid adjusted target")
        record_flags[split] = records_positive
        result[split] = {
            "events_true_false": counted(decisions),
            "records_with_true": small(len(records_positive)),
            "vote_count_distribution": counted(n_votes),
            "contested_pct": round(100 * contested_total / len(subset), 4) if len(subset) else None,
            "q_mean_distribution": counted(qm),
            "other_vote_counts": counted(other),
            "adjudicated_events": small(adjudications),
        }
    return result, record_flags


def crc_probe(train_labels, seed):
    flags = train_labels.assign(y=train_labels["decision"].eq("True"))
    record_positive = flags.groupby("record", sort=True)["y"].any()
    record_names = record_positive.index.to_numpy()
    strata = record_positive.to_numpy(dtype=int)
    _, calib = train_test_split(
        record_names, test_size=0.30, random_state=seed, stratify=strata,
    )
    n = int(record_positive.loc[calib].sum())
    return {
        "calib_records": len(calib),
        "calib_positive_records": n,
        "minimum_alpha": round(1 / (n + 1), 9),
        "primary_alpha": 0.01 if n >= 300 else 0.02,
    }


def write_outputs(out_json, report_md, summary, hard_checks, code_path, started):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    # Director-approved HF1: replace only AFTER full comparison with old A-G.
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows = [
        "# P05-T01b audit (HF1, aggregate only)",
        "",
        "## Scientific run-checks",
        "",
        *[f"- {name}: {'PASS' if passed else 'FAIL'}" for name, passed in hard_checks],
    ]
    rows += ["", "## C_waveforms", "", "```json",
             json.dumps(summary["C_waveforms"], indent=2, ensure_ascii=False), "```"]
    for section in ("D_headers_annotated", "D_headers_waveforms"):
        rows += ["", f"## {section}", "",
                 "| Split | Header field | Value | Count |", "|---|---|---|---:|"]
        for group, data in summary[section].items():
            for metric in ("fs", "sig_len", "channels_n", "channel_names"):
                for value, count in data[metric].items():
                    safe_value = str(value).replace("|", "\\|")
                    rows.append(f"| {group} | {metric} | {safe_value} | {count} |")
            rows.append(f"| {group} | read_errors | n/a | {data['read_errors']} |")
    for section in ("E_slots", "F_train_val_only", "G_crc"):
        rows += ["", f"## {section}", "", "```json",
                 json.dumps(summary[section], indent=2, ensure_ascii=False), "```"]
    report_md.write_text("\n".join(rows) + "\n", encoding="utf-8")

    manifest = code_path.parents[1] / "manifest.csv"
    header_needed = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header_needed:
            writer.writerow(["path", "sha256", "n_samples", "timestamp_utc", "step"])
        for path in (out_json, report_md):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow([str(path), digest, summary["A_labeled_event_count"],
                             datetime.now(timezone.utc).isoformat(), "P05-T01b-HF1"])
    return round(time.perf_counter() - started, 3)


def compare_old_ag(old, new):
    """Compare every scientific A-G field; never expose event identifiers."""
    sections = (
        "status", "A_labeled_event_count", "AB_checks", "C_waveforms",
        "D_headers_annotated", "D_headers_waveforms", "E_slots",
        "F_train_val_only", "G_crc",
    )
    differences = []

    def compare(path, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    differences.append((f"{path}.{key}", left.get(key), right.get(key)))
                else:
                    compare(f"{path}.{key}", left[key], right[key])
        elif left != right:
            differences.append((path, left, right))

    for section in sections:
        compare(section, old.get(section), new.get(section))
    if differences:
        print("HF1 A-G COMPARISON: FAIL; existing outputs preserved")
        for field, before, after in differences:
            print(f"DIFF {field}: old={before!r} | hf1={after!r}")
        raise ValueError(f"HF1 scientific comparison differs in {len(differences)} fields")
    print(f"HF1 A-G COMPARISON: PASS | sections={len(sections)}/{len(sections)}")


def audit(config):
    started = time.perf_counter()
    np.random.seed(int(config["seed"]))
    raw = Path(config["raw_dir"])
    ann_dir = raw / "annotated_waveforms"
    waves_dir = raw / "waveforms"
    out_json = Path(config["out_json"])
    report_md = Path(config["report_md"])
    if not out_json.is_file() or not report_md.is_file():
        raise FileNotFoundError("HF1 requires both original T01b outputs for comparison")
    with out_json.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    hard = []

    split_frame = read_csv(raw / "benchmark_data_split.csv", ("split", "record", "event"))
    split_frame = split_frame[["split", "record", "event"]].copy()
    split_frame["split"] = split_frame["split"].map(normalize_split)
    if not split_frame["split"].isin(SPLITS).all():
        raise ValueError("Unknown split value")
    split_frame = split_frame.sort_values(["split", "record", "event"], kind="stable")
    all_split_keys = keys(split_frame)
    trainval_frame = split_frame.loc[split_frame["split"].isin(("train", "val"))]

    label_path = raw / "event_labels.csv"
    # Approved exception: use ALL final decisions for these two global totals
    # only. Release the entire decision DataFrame immediately afterwards.
    global_decisions = pd.read_csv(
        label_path, usecols=["decision"], dtype=str, keep_default_na=False,
    )
    decision_counts = global_decisions["decision"].str.strip().str.lower().value_counts()
    global_true = int(decision_counts.get("true", 0))
    global_false = int(decision_counts.get("false", 0))
    check("A_global_true", global_true, 1441, hard)
    check("A_global_false", global_false, 3596, hard)
    del global_decisions, decision_counts, global_true, global_false
    gc.collect()

    label_keys_only = pd.read_csv(
        label_path, usecols=["record", "event"],
        dtype=str, keep_default_na=False,
    )
    total_label_rows = len(label_keys_only)
    label_keys = keys(label_keys_only)
    del label_keys_only
    labels_trainval = read_allowed_decisions(
        label_path, ("record", "event"), keys(trainval_frame),
    )
    safe_labels = labels_trainval.drop_duplicates(["record", "event"]).merge(
        trainval_frame.drop_duplicates(["record", "event"]),
        on=["record", "event"], how="inner", validate="one_to_one",
    )
    safe_labels["decision"] = safe_labels["decision"].str.strip().str.title()
    del labels_trainval
    gc.collect()

    label_count = total_label_rows
    check("A_labeled_events", label_count, EXPECTED["labeled_events"], hard)
    check("A_labeled_records", len({r for r, _ in label_keys}), EXPECTED["labeled_records"], hard)
    for split in SPLITS:
        subset = split_frame.loc[split_frame["split"] == split]
        check(f"A_split_events_{split}", len(subset), EXPECTED[f"split_events_{split}"], hard)
        check(f"A_split_records_{split}", subset["record"].nunique(),
              EXPECTED[f"split_records_{split}"], hard)

    ann_hea, ann_dat, ann_pairs = scan_pairs(ann_dir)
    w_hea, w_dat, wave_pairs = scan_pairs(waves_dir)
    check("A_annotated_pairs", len(ann_pairs), EXPECTED["annotated_pairs"], hard)
    check("A_waveform_pairs", len(wave_pairs), EXPECTED["waveform_pairs"], hard)
    check("B_annotated_orphan_header", len(ann_hea - ann_dat), 0, hard)
    check("B_annotated_orphan_data", len(ann_dat - ann_hea), 0, hard)
    check("B_waveform_orphan_header", len(w_hea - w_dat), 0, hard)
    check("B_waveform_orphan_data", len(w_dat - w_hea), 0, hard)

    rec_sets = {name: set(split_frame.loc[split_frame["split"] == name, "record"])
                for name in SPLITS}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        check(f"B_record_overlap_{first}_{second}", len(rec_sets[first] & rec_sets[second]), 0, hard)
    check("B_duplicate_split_keys", len(split_frame) - len(all_split_keys), 0, hard)
    check("B_duplicate_label_keys", total_label_rows - len(label_keys), 0, hard)
    check("B_split_without_label", len(all_split_keys - label_keys), 0, hard)
    check("B_label_without_split", len(label_keys - all_split_keys), 0, hard)
    check("B_split_missing_waveform", len(all_split_keys - ann_pairs), 0, hard)

    wave_rec = Counter(record for record, _ in wave_pairs)
    record_split = {record: name for name in SPLITS for record in rec_sets[name]}
    counts = list(wave_rec.values())
    c_data = {
        "records": len(wave_rec),
        "record_overlap_by_split": {name: len(set(wave_rec) & rec_sets[name]) for name in SPLITS},
        "events_also_annotated": len(wave_pairs & ann_pairs),
        "events_per_record": {
            "min": int(np.min(counts)) if counts else None,
            "median": float(np.median(counts)) if counts else None,
            "p99": float(np.percentile(counts, 99)) if counts else None,
            "max": int(np.max(counts)) if counts else None,
        },
        "records_with_more_than_10_events": len([x for x in counts if x > 10]),
    }

    print("Scanning all WFDB headers (no waveform samples)...")
    headers_a = count_headers(ann_dir, ann_pairs, record_split)
    headers_w = count_headers(waves_dir, wave_pairs, record_split)
    print("Scanning 60 s pre-alarm windows (aggregate masks only)...")
    slots = count_slot_presence(ann_dir, split_frame, int(config["t0"]), int(config["window_s"]))

    votes = read_allowed_decisions(
        raw / "event_label_per_annotator.csv",
        ("record", "event", "is_adjudication"), keys(trainval_frame),
    )
    vote_data, _ = summarize_votes(safe_labels, votes)
    del votes
    crc = crc_probe(safe_labels.loc[safe_labels["split"] == "train"], int(config["seed"]))

    print("=== REPORT P05-T01b ===")
    status = "PASS" if all(passed for _, passed in hard) else "FAIL"
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:8]
    print(f"status={status} | pc=reference-workstation | gpu=not used | Python={platform.python_version()}")
    print(f"script=t01b_audit.py | sha256_8={script_sha} | runtime_s={time.perf_counter()-started:.1f}")
    print(f"A: labeled={label_count} expected=5037; annotated={len(ann_pairs)} expected=5037; "
          f"waveforms={len(wave_pairs)} expected=17131")
    print(f"A: global True expected=1441, False expected=3596 | "
          f"{'PASS' if dict(hard).get('A_global_true') and dict(hard).get('A_global_false') else 'FAIL'}")
    print(f"B: hard_checks_passed={sum(p for _,p in hard)}/{len(hard)}")
    overlaps_safe = {k: small(v) for k, v in c_data["record_overlap_by_split"].items()}
    print(f"C: wave_records={c_data['records']}; overlaps={overlaps_safe}; "
          f"shared_events={small(c_data['events_also_annotated'])}; "
          f"events_per_record={c_data['events_per_record']}; "
          f">10={small(c_data['records_with_more_than_10_events'])}")
    print(f"D: annotated_headers={len(ann_pairs)}; waveform_headers={len(wave_pairs)}")
    for corpus, details in (("annotated", headers_a), ("waveforms", headers_w)):
        for split, info in details.items():
            print(f"D_{corpus}_{split}: fs={info['fs']}; sig_len={info['sig_len']}; "
                  f"n_sig={info['channels_n']}; channels={info['channel_names']}; errors={info['read_errors']}")
    for split in SPLITS:
        e = slots[split]
        print(f"E_{split}: present_pct={e['present_pct']}; top6={e['top6_combinations']}; "
              f"read_errors={e['read_errors']}")
    for split in ("train", "val"):
        print(f"F_{split}: {vote_data[split]}")
    print(f"G: calib_records={small(crc['calib_records'])}; "
          f"positive_records={small(crc['calib_positive_records'])}; "
          f"1/(n+1)={crc['minimum_alpha']}; alpha={crc['primary_alpha']}")
    error_groups = sum(entry["read_errors"] != "0"
                       for corpus in (headers_a, headers_w) for entry in corpus.values())
    error_splits = sum(entry["read_errors"] != "0" for entry in slots.values())
    print(f"warnings: header_groups_with_errors={small(error_groups)}; "
          f"slot_splits_with_errors={small(error_splits)}")

    summary = {
        "step": "P05-T01b", "status": status,
        "A_labeled_event_count": label_count,
        "AB_checks": {name: "PASS" if passed else "FAIL" for name, passed in hard},
        "C_waveforms": {
            **c_data,
            "record_overlap_by_split": overlaps_safe,
            "events_also_annotated": small(c_data["events_also_annotated"]),
            "records_with_more_than_10_events": small(c_data["records_with_more_than_10_events"]),
        },
        "D_headers_annotated": headers_a,
        "D_headers_waveforms": headers_w,
        "E_slots": slots,
        "F_train_val_only": vote_data,
        "G_crc": crc,
    }
    if status == "PASS":
        compare_old_ag(baseline, summary)
        elapsed = write_outputs(out_json, report_md, summary, hard, Path(__file__), started)
        print(f"outputs: {out_json.name}, {report_md.name}, manifest.csv | saved=PASS | total_s={elapsed}")
    else:
        print("outputs: not saved because a scientific hard check failed")
    print("=== END ===")
    return 0 if status == "PASS" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return audit(config)


if __name__ == "__main__":
    raise SystemExit(main())
