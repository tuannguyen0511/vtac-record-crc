"""P05-T01b-HF1: key-only diagnostic. Never read any decision column."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


SPLITS = ("train", "val", "test")
CLASS_NAMES = ("a_case", "b_integer", "c_separator_suffix_prefix",
               "d_record_shared", "e_event_shared", "f_no_match")


def safe(value):
    value = int(value)
    return "<10" if 0 < value < 10 else str(value)


def read_keys(path, columns):
    """Pandas usecols prevents exposing decisions to Python code."""
    return pd.read_csv(
        path, usecols=list(columns), dtype=str, keep_default_na=False,
    )


def pair_set(frame):
    return set(zip(frame["record"], frame["event"]))


def normalize_split(value):
    value = str(value).strip().lower()
    return {"validation": "val", "valid": "val"}.get(value, value)


def norm_case(pair):
    return tuple(part.strip().casefold() for part in pair)


def strip_zeroes(part):
    # Never convert scientific-looking IDs such as "0e1234" into a number.
    return str(int(part)) if re.fullmatch(r"[0-9]+", part) else part


def norm_integer(pair):
    return tuple(strip_zeroes(part) for part in norm_case(pair))


def canonical_part(part, kind):
    part = part.strip().casefold()
    part = re.sub(r"\.(hea|dat|csv|mat|wfdb)$", "", part)
    prefix = r"(?:record|rec)" if kind == "record" else r"(?:event|evt|alarm)"
    part = re.sub(r"^(?:" + prefix + r")[._\- ]+", "", part)
    # Treat underscores, dots, hyphens and spaces as equivalent separators.
    part = re.sub(r"[._\- ]+", "", part)
    return strip_zeroes(part)


def norm_common(pair):
    return (canonical_part(pair[0], "record"),
            canonical_part(pair[1], "event"))


def unique_matches(source_all, vote_all):
    """A/B/C candidates must be 1:1 in BOTH complete key universes."""
    exact = source_all & vote_all
    source_missing = source_all - exact
    vote_extra = vote_all - exact
    recovered = {}
    by_stage = {}
    for stage, normalizer in (
        ("a_case", norm_case),
        ("b_integer", norm_integer),
        ("c_separator_suffix_prefix", norm_common),
    ):
        source_index = defaultdict(list)
        vote_index = defaultdict(list)
        for key in source_all:
            source_index[normalizer(key)].append(key)
        for key in vote_all:
            vote_index[normalizer(key)].append(key)
        pairs = []
        for name in sorted(set(source_index) & set(vote_index)):
            source_matches = source_index[name]
            vote_matches = vote_index[name]
            if len(source_matches) != 1 or len(vote_matches) != 1:
                continue
            first, second = source_matches[0], vote_matches[0]
            if first in source_missing and second in vote_extra:
                pairs.append((first, second))
        for first, second in pairs:
            source_missing.remove(first)
            vote_extra.remove(second)
            recovered[first] = (second, stage)
        by_stage[stage] = len(pairs)
    return recovered, by_stage, source_missing, vote_extra


def classify_missing(missing_keys, other_all, recovered):
    """Exclusive a..f classification, without emitting any actual identifier."""
    categories = Counter({name: 0 for name in CLASS_NAMES})
    for _, stage in recovered.values():
        categories[stage] += 1
    records = {record for record, _ in other_all}
    events = {event for _, event in other_all}
    for record, event in missing_keys:
        if record in records:
            categories["d_record_shared"] += 1
        elif event in events:
            categories["e_event_shared"] += 1
        else:
            categories["f_no_match"] += 1
    return {name: safe(categories[name]) for name in CLASS_NAMES}


def annotated_keys(folder):
    headers = {
        (path.relative_to(folder).parent.as_posix(), path.stem)
        for path in folder.rglob("*.hea")
    }
    data = {
        (path.relative_to(folder).parent.as_posix(), path.stem)
        for path in folder.rglob("*.dat")
    }
    return headers & data, len(headers - data), len(data - headers)


def audit(config):
    raw = Path(config["raw_dir"])
    original_json = Path(config["out_json"])
    original_md = Path(config["report_md"])
    out_json = original_json.with_name("t01b_hf1_keys.json")
    out_md = original_md.with_name("P05-T01b-HF1_keys.md")

    if out_json.exists() or out_md.exists():
        raise FileExistsError("HF1 key reports already exist; preserving previous outputs")

    split = read_keys(raw / "benchmark_data_split.csv", ("split", "record", "event"))
    split["split"] = split["split"].map(normalize_split)
    if not split["split"].isin(SPLITS).all():
        raise ValueError("Unknown split category")
    labels = read_keys(raw / "event_labels.csv", ("record", "event"))
    votes = read_keys(
        raw / "event_label_per_annotator.csv",
        ("record", "event", "is_adjudication"),
    )

    split_all = pair_set(split)
    label_all = pair_set(labels)
    vote_all = pair_set(votes)
    ann_all, missing_dat, missing_hea = annotated_keys(raw / "annotated_waveforms")
    split_by_group = {
        group: pair_set(split.loc[split["split"] == group])
        for group in SPLITS
    }

    # K1: totals include test KEY presence only, never any test decision.
    missing_by_split = {
        group: safe(len(group_keys - vote_all))
        for group, group_keys in split_by_group.items()
    }
    missing_trainval = len((split_by_group["train"] | split_by_group["val"]) - vote_all)
    missing_total = len(split_all - vote_all)
    extra_total = len(vote_all - split_all)
    k1 = {
        "vote_rows": safe(len(votes)),
        "unique_vote_keys": safe(len(vote_all)),
        "split_events_without_votes": missing_by_split,
        "vote_keys_not_in_split": safe(extra_total),
        "total_split_events_without_votes": safe(missing_total),
        "train_val_missing_over_91": missing_trainval > 91,
    }

    # K2: only uniquely invertible matches from the original raw key universes.
    recovered, stage_counts, still_missing, still_extra = unique_matches(split_all, vote_all)
    reverse = {vote: (source, stage) for source, (vote, stage) in recovered.items()}
    k2 = {
        "split_missing_classification": classify_missing(
            still_missing, vote_all, recovered,
        ),
        "vote_extra_classification": classify_missing(
            still_extra, split_all, reverse,
        ),
        "unique_recovered_by_rule": {
            name: safe(stage_counts[name])
            for name in ("a_case", "b_integer", "c_separator_suffix_prefix")
        },
    }

    # K3: intersection matrix, ordered as labels / official split / WFDB pairs.
    sets = {
        "event_labels": label_all,
        "benchmark_split": split_all,
        "annotated_waveforms": ann_all,
    }
    k3 = {
        "set_sizes": {key: safe(len(value)) for key, value in sets.items()},
        "intersections": {
            left: {right: safe(len(x & y)) for right, y in sets.items()}
            for left, x in sets.items()
        },
        "unpaired_annotated_hea": safe(missing_dat),
        "unpaired_annotated_dat": safe(missing_hea),
    }

    # K4: the only per-annotator extra field read is is_adjudication.
    flags = votes["is_adjudication"].str.strip().str.casefold()
    is_adj = flags.isin(("true", "1"))
    is_nonadj = flags.isin(("false", "0"))
    other_flag_rows = int((~(is_adj | is_nonadj)).sum())
    nonadj_votes = Counter(zip(votes.loc[is_nonadj, "record"],
                               votes.loc[is_nonadj, "event"]))
    per_split = {}
    for group, event_keys in split_by_group.items():
        hist = Counter()
        for key in event_keys:
            count = nonadj_votes.get(key, 0)
            bucket = str(count) if count < 4 else ">=4"
            hist[bucket] += 1
        per_split[group] = {
            bucket: safe(hist[bucket]) for bucket in ("0", "1", "2", "3", ">=4")
        }
    k4 = {
        "adjudication_row_counts": {
            "TRUE": safe(is_adj.sum()),
            "FALSE": safe(is_nonadj.sum()),
            "other": safe(other_flag_rows),
        },
        "nonadjudication_vote_rows_per_event_by_split": per_split,
    }

    # K5: the preregistered >=95% denominator is ALL unmatched official keys.
    recovered_count = len(recovered)
    recovered_pct = round(100.0 * recovered_count / missing_total, 4) if missing_total else 100.0
    k5_yes = recovered_pct >= 95.0
    rule_descriptions = {
        "a_case": "lower-case record and event (strip surrounding whitespace)",
        "b_integer": "then remove leading zeros ONLY from all-digit record/event tokens",
        "c_separator_suffix_prefix": (
            "then strip .hea/.dat/.mat/.csv/.wfdb, standard record/event/"
            "evt/alarm prefixes, and normalize -_. separators"
        ),
    }
    active_rules = [rule_descriptions[name] for name, count in stage_counts.items() if count]
    k5 = {
        "deterministic_recovery_at_least_95pct": "YES" if k5_yes else "NO",
        "mismatched_official_keys": safe(missing_total),
        "uniquely_recovered_keys": safe(recovered_count),
        "recovered_pct": recovered_pct,
        "mapping_rules": active_rules if k5_yes else [],
        "train_val_missing_over_91": missing_trainval > 91,
        "status": "STOP_DIRECTOR" if missing_trainval > 91 else "REPORT_ONLY",
    }

    result = {
        "step": "P05-T01b-HF1", "decision_columns_read": 0,
        "K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5,
    }
    print("=== REPORT P05-T01b-HF1 KEYS ===")
    print("decision_columns_read=0 | expected=0 | PASS")
    for name, value in (("K1", k1), ("K2", k2), ("K3", k3),
                        ("K4", k4), ("K5", k5)):
        print(f"{name}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")
    k3_pass = len(split_all) == len(label_all) == len(ann_all) == 5037 and (
        split_all == label_all == ann_all
    ) and missing_dat == 0 and missing_hea == 0
    print(f"K3_all_equal: actual={int(k3_pass)} | expected=1 | "
          f"{'PASS' if k3_pass else 'FAIL'}")
    print(f"K1_train_val_missing_over_91: actual={int(missing_trainval > 91)} "
          f"| expected=0 | {'PASS' if missing_trainval <= 91 else 'FAIL'}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_lines = [
        "# P05-T01b-HF1: key-only audit", "",
        "This report reads only record, event, split, and is_adjudication metadata.",
        "No decision column or waveform samples were read.", "",
        f"- K3 three-way alignment: {'PASS' if k3_pass else 'FAIL'}",
        f"- Train + val mismatch threshold (<=91): "
        f"{'PASS' if missing_trainval <= 91 else 'FAIL'}", "",
    ]
    for name in ("K1", "K2", "K3", "K4", "K5"):
        md_lines += [f"## {name}", "", "```json",
                     json.dumps(result[name], indent=2, ensure_ascii=False), "```", ""]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    manifest = Path(__file__).resolve().parents[1] / "manifest.csv"
    add_header = not manifest.exists()
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if add_header:
            writer.writerow(["path", "sha256", "n_samples", "timestamp_utc", "step"])
        for path in (out_json, out_md):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow([str(path), digest, len(split_all),
                             datetime.now(timezone.utc).isoformat(), "P05-T01b-HF1"])

    status = "PASS" if k3_pass and missing_trainval <= 91 else "FAIL"
    print(f"HF1_KEYS_STATUS={status} | output_saved=PASS")
    print("=== END ===")
    return 0 if status == "PASS" else 2


def main():
    parser = argparse.ArgumentParser(description="Key-only VTaC HF1 diagnostic")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return audit(config)


if __name__ == "__main__":
    raise SystemExit(main())
