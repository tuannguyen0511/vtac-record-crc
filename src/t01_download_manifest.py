"""P05-T01: VTaC v1.1 download, integrity and metadata audit."""
def _t08_public_path(kind, suffix):
    from pathlib import Path
    import yaml
    root=Path(__file__).resolve().parents[1]
    cfile=root/'configs'/'paths.yaml'
    if not cfile.is_file():
        raise FileNotFoundError('Create configs/paths.yaml from paths.example.yaml')
    cfg=yaml.safe_load(cfile.read_text(encoding='utf-8'))
    base=root if kind in ('repo','workspace') else (root/str(cfg[kind])).resolve()
    if kind=='workspace':
        raise RuntimeError('workspace root is not a portable release path')
    return base.joinpath(*[x for x in suffix.split('/') if x])


import argparse
import collections
import csv
import hashlib
import logging
import os
import random
import re
import shutil
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import quote

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "40"

import pandas as pd
import requests
import wfdb
import yaml


ROOT = Path(str(_t08_public_path("repo", "")))
LOG_DIR = ROOT / "P05_VTaC_FalseAlarm" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (
    "P05-T01_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
)

logger = logging.getLogger("p05_t01")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(message)s")

for handler in (
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(LOG_FILE, encoding="utf-8"),
):
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_tls = threading.local()


def session():
    if not hasattr(_tls, "http"):
        _tls.http = requests.Session()
        _tls.http.headers.update({"User-Agent": "P05-VTaC-research/1.0"})
    return _tls.http


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_count(value):
    if value is None:
        return "NA"
    return "<10" if 0 < value < 10 else str(value)


def download_checksums(base_url, root):
    path = root / "SHA256SUMS.txt"
    if path.is_file() and path.stat().st_size > 0:
        return path

    partial = root / "SHA256SUMS.txt.part"
    url = base_url.rstrip("/") + "/SHA256SUMS.txt"

    with session().get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    os.replace(partial, path)
    return path


def load_checksums(path, root):
    expected = {}
    resolved_root = root.resolve()
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+)$")

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            match = pattern.match(line)
            if not match:
                raise ValueError("Invalid SHA256SUMS format")

            digest, name = match.groups()
            name = name.replace("\\", "/")
            rel = PurePosixPath(name)

            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe checksum path")

            target = root.joinpath(*rel.parts).resolve()
            if not target.is_relative_to(resolved_root):
                raise ValueError("Checksum path escapes output directory")

            key = rel.as_posix()
            if key in expected:
                raise ValueError("Duplicate checksum path")
            expected[key] = digest.lower()

    if not expected:
        raise ValueError("Empty SHA256SUMS")

    return expected


def download_one(item, root, base_url):
    rel, expected_hash = item
    target = root.joinpath(*PurePosixPath(rel).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")

    if target.exists():
        if sha256_file(target) == expected_hash:
            return "OK", rel, target.stat().st_size, expected_hash
        return "BAD", rel, 0, ""

    if partial.exists() and sha256_file(partial) == expected_hash:
        os.replace(partial, target)
        return "OK", rel, target.stat().st_size, expected_hash

    url = base_url.rstrip("/") + "/" + quote(rel, safe="/")
    hash_mismatch = False

    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}

            with session().get(
                url,
                headers=headers,
                stream=True,
                timeout=(15, 120),
            ) as response:
                if response.status_code == 416:
                    if partial.exists() and sha256_file(partial) == expected_hash:
                        os.replace(partial, target)
                        return "OK", rel, target.stat().st_size, expected_hash
                    raise IOError("Invalid partial download")

                response.raise_for_status()

                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise IOError("Invalid Content-Range")
                    mode = "ab"
                else:
                    mode = "wb"

                with partial.open(mode) as handle:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            handle.write(chunk)

            if sha256_file(partial) == expected_hash:
                os.replace(partial, target)
                return "OK", rel, target.stat().st_size, expected_hash

            hash_mismatch = True
            partial.unlink()

        except (requests.RequestException, OSError):
            pass

        if attempt < 3:
            time.sleep(2 ** attempt)

    return (
        ("BAD" if hash_mismatch else "MISSING"),
        rel,
        0,
        "",
    )


def verify_and_download(expected, root, base_url, workers):
    counts = collections.Counter()
    manifest_rows = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(download_one, item, root, base_url)
            for item in expected.items()
        ]

        for index, future in enumerate(as_completed(futures), start=1):
            try:
                status, rel, size, digest = future.result()
            except Exception:
                status, rel, size, digest = "MISSING", "", 0, ""

            counts[status] += 1

            if status == "OK":
                manifest_rows.append(
                    {
                        "relative_path": rel,
                        "bytes": size,
                        "sha256": digest,
                    }
                )

            if index % 1000 == 0:
                logger.info(
                    "Download progress: completed=%d/%d; OK=%d; BAD=%d; MISSING=%d",
                    index,
                    len(expected),
                    counts["OK"],
                    counts["BAD"],
                    counts["MISSING"],
                )

    return counts, manifest_rows


def normalize_split(value):
    value = str(value).strip().lower()
    if value in ("train", "training"):
        return "train"
    if value in ("val", "valid", "validation", "dev"):
        return "val"
    if value in ("test", "testing"):
        return "test"
    return "unknown"


def load_splits(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []

        split_col = next(
            (
                col for col in columns
                if col.lower() in (
                    "split", "data_split", "dataset_split",
                    "partition", "subset"
                )
            ),
            None,
        )

        if split_col is None or "record" not in columns:
            raise ValueError("Required split columns unavailable")

        has_event = "event" in columns
        mapping = {}

        for row in reader:
            record = row["record"]
            event = row["event"] if has_event else ""
            key = (record, event)
            split = normalize_split(row[split_col])

            if key in mapping and mapping[key] != split:
                raise ValueError("Conflicting split assignment")
            mapping[key] = split

    return mapping, has_event


def split_of(row, mapping, has_event):
    record = row.get("record", "")
    event = row.get("event", "")
    return mapping.get(
        (record, event if has_event else ""),
        "unknown",
    )


def true_value(value):
    value = str(value).strip().lower()
    if value in ("true", "t", "1", "yes"):
        return True
    if value in ("false", "f", "0", "no"):
        return False
    return None


def csv_type(values):
    values = [v for v in values if v != ""]
    if not values:
        return "unknown"
    if all(v.lower() in ("true", "false") for v in values):
        return "bool"
    try:
        for v in values:
            int(v)
        return "int"
    except ValueError:
        pass
    try:
        for v in values:
            float(v)
        return "float"
    except ValueError:
        return "string"


def audit_csv(root, mapping, has_event):
    all_columns = set()
    labeled_keys = set()
    trainval_true = 0
    trainval_labeled = 0
    unknown_decisions = 0
    labeled_total = 0

    for path in sorted(root.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            all_columns.update(c.lower() for c in columns)
            unique = {c: set() for c in columns}
            rows = 0

            protected = path.name in (
                "event_labels.csv",
                "event_label_per_annotator.csv",
            )

            for row in reader:
                rows += 1
                split = split_of(row, mapping, has_event)

                for col in columns:
                    # Do not inspect annotation-related fields for test
                    # or unknown-split events.
                    if protected and col not in ("record", "event"):
                        if split not in ("train", "val"):
                            continue
                    unique[col].add(row.get(col, ""))

                if path.name == "event_labels.csv":
                    labeled_total += 1
                    key = (row.get("record", ""), row.get("event", ""))
                    labeled_keys.add(key)

                    if split in ("train", "val"):
                        decision = true_value(row.get("decision", ""))
                        if decision is None:
                            unknown_decisions += 1
                        else:
                            trainval_labeled += 1
                            trainval_true += int(decision)

            logger.info(
                "CSV: %s; rows=%s",
                path.name,
                safe_count(rows),
            )

            for col in columns:
                logger.info(
                    "  column=%s; dtype=%s; unique=%s%s",
                    col,
                    csv_type(unique[col]),
                    safe_count(len(unique[col])),
                    "; non-test only" if protected and col not in (
                        "record", "event"
                    ) else "",
                )

    true_rate = (
        trainval_true / trainval_labeled
        if trainval_labeled else None
    )

    return {
        "columns": all_columns,
        "labeled_keys": labeled_keys,
        "labeled_total": labeled_total,
        "true_rate": true_rate,
        "true_n": trainval_true,
        "true_den": trainval_labeled,
        "unknown_decisions": unknown_decisions,
    }


def scan_headers(root):
    annotated = root / "annotated_waveforms"
    waveforms = root / "waveforms"

    all_events = set()
    waveform_events = set()
    waveform_headers = []

    for directory in (annotated, waveforms):
        if not directory.exists():
            raise FileNotFoundError("Waveform directory unavailable")

        for path in directory.rglob("*.hea"):
            key = (path.parent.name, path.stem)
            all_events.add(key)

            if directory == waveforms:
                waveform_events.add(key)
                waveform_headers.append(path)

    patients = collections.Counter()
    for record, event in all_events:
        patients[record] += 1

    return all_events, waveform_events, waveform_headers, patients


def sample_header_metadata(paths, seed):
    rng = random.Random(seed)
    sample = rng.sample(sorted(paths), min(20, len(paths)))

    channel_counts = collections.Counter()
    fs_counts = collections.Counter()
    durations = []
    failures = 0

    for path in sample:
        try:
            header = wfdb.rdheader(str(path.with_suffix("")))
            channels = tuple(sorted(header.sig_name))
            fs = float(header.fs)
            duration = float(header.sig_len) / fs

            channel_counts[channels] += 1
            fs_counts[fs] += 1
            durations.append(duration)
        except Exception:
            failures += 1

    logger.info("WFDB header sample: requested=20; valid=%s; failures=%s",
                safe_count(len(durations)), safe_count(failures))

    for fs, count in sorted(fs_counts.items()):
        logger.info(
            "  sampling_frequency_hz=%.3f; count=%s",
            fs,
            safe_count(count),
        )

    for channels, count in sorted(channel_counts.items()):
        logger.info(
            "  channel_combination=%s; count=%s",
            ",".join(channels),
            safe_count(count),
        )

    if durations:
        logger.info(
            "  duration_s: min=%.2f; median=%.2f; max=%.2f",
            min(durations),
            statistics.median(durations),
            max(durations),
        )

    return len(durations), failures


def main():
    timer = time.perf_counter()
    warnings = []
    status = "FAIL"

    counts = collections.Counter()
    expected_n = 0
    csv_info = {}
    split_counts = collections.Counter()
    unlabeled_n = None
    patient_n = None
    patient_distribution = None
    header_valid = 0
    manifest_path = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    try:
        with open(args.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        base_url = config["url"]
        raw_dir = Path(config["out_dir"])
        manifest_path = Path(config["manifest"])
        workers = int(config["workers"])
        seed = int(config["seed"])
        min_free_gb = float(config["min_free_gb"])

        if not base_url.startswith(
            "https://physionet.org/files/vtac/1.1/"
        ):
            raise ValueError("Unexpected dataset URL")

        if workers != 8 or seed != 0:
            raise ValueError("Invalid workers or seed")

        free_gb = shutil.disk_usage(str(_t08_public_path("t01_external_resource", ""))).free / 1.0e9
        logger.info("Free disk E: %.2f GB", free_gb)

        if free_gb < min_free_gb:
            raise RuntimeError("Free disk space below 40 GB")

        raw_dir.mkdir(parents=True, exist_ok=True)

        checksum_file = download_checksums(base_url, raw_dir)
        expected = load_checksums(checksum_file, raw_dir)
        expected_n = len(expected)

        logger.info(
            "SHA256SUMS loaded; expected files=%s",
            safe_count(expected_n),
        )

        counts, manifest_rows = verify_and_download(
            expected,
            raw_dir,
            base_url,
            workers,
        )

        logger.info(
            "SHA256: OK=%d; BAD=%d; MISSING=%d",
            counts["OK"],
            counts["BAD"],
            counts["MISSING"],
        )

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(
            manifest_path.name + ".tmp.parquet"
        )

        pd.DataFrame(
            manifest_rows,
            columns=["relative_path", "bytes", "sha256"],
        ).to_parquet(temporary, index=False)

        os.replace(temporary, manifest_path)

        if counts["BAD"] or counts["MISSING"]:
            warnings.append("Download or SHA256 verification incomplete")
        else:
            mapping, has_event = load_splits(
                raw_dir / "benchmark_data_split.csv"
            )

            split_counts = collections.Counter(mapping.values())
            csv_info = audit_csv(raw_dir, mapping, has_event)

            (
                all_events,
                waveform_events,
                waveform_headers,
                patients,
            ) = scan_headers(raw_dir)

            unlabeled_n = len(
                waveform_events - csv_info["labeled_keys"]
            )
            patient_n = len(patients)

            if patients:
                values = list(patients.values())
                patient_distribution = (
                    min(values),
                    statistics.median(values),
                    max(values),
                )

            header_valid, header_failures = sample_header_metadata(
                waveform_headers,
                seed,
            )

            if header_valid != 20 or header_failures:
                warnings.append("WFDB header sample incomplete")

            if csv_info["unknown_decisions"]:
                warnings.append("Unknown train/validation decisions")

            if csv_info["true_den"] == 0:
                warnings.append("Train/validation True rate unavailable")

            if "unknown" in split_counts:
                warnings.append("Unknown split assignments")

            columns = csv_info["columns"]

            site = any(
                "hospital" in c or "site" in c for c in columns
            )
            vendor = any(
                "vendor" in c or "manufacturer" in c
                or "monitor" in c for c in columns
            )
            annotator = "annotator" in columns
            patient_id = "record" in columns

            logger.info(
                "QUESTIONS: hospital/site=%s; monitor_vendor=%s; "
                "per_annotator=%s; surrogate_record_id=%s",
                "YES" if site else "NO",
                "YES" if vendor else "NO",
                "YES" if annotator else "NO",
                "YES" if patient_id else "NO",
            )

            if not csv_info["labeled_total"]:
                warnings.append("Labeled event count unavailable")

            if not patient_id or not annotator:
                warnings.append("Required metadata columns unavailable")

            if not warnings:
                status = "PASS"

    except Exception as exc:
        # Do not print exception text or record identifiers.
        warnings.append("Pipeline error: " + type(exc).__name__)

    runtime = time.perf_counter() - timer
    sha8 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()[:8]

    logger.info("=== REPORT P05-T01 ===")
    logger.info("status: %s", status)
    logger.info("pc: reference-workstation")
    logger.info("gpu: not used; CPU metadata/download task")
    logger.info("runtime_s: %.1f", runtime)
    logger.info(
        "script: t01_download_manifest.py; sha256=%s",
        sha8,
    )
    logger.info("data: VTaC v1.1; no patient-level output")
    logger.info(
        "result: expected=%s; sha_ok=%s; sha_bad=%s; sha_missing=%s",
        safe_count(expected_n),
        safe_count(counts["OK"]),
        safe_count(counts["BAD"]),
        safe_count(counts["MISSING"]),
    )

    if csv_info:
        logger.info(
            "labeled_events=%s; true_trainval=%s/%s; true_rate_trainval=%s",
            safe_count(csv_info["labeled_total"]),
            safe_count(csv_info["true_n"]),
            safe_count(csv_info["true_den"]),
            (
                "%.6f" % csv_info["true_rate"]
                if csv_info["true_rate"] is not None else "NA"
            ),
        )

    logger.info(
        "unlabeled_events=%s; surrogate_patient_records=%s",
        safe_count(unlabeled_n),
        safe_count(patient_n),
    )

    logger.info(
        "split_events: train=%s; val=%s; test=%s; unknown=%s",
        safe_count(split_counts["train"]),
        safe_count(split_counts["val"]),
        safe_count(split_counts["test"]),
        safe_count(split_counts["unknown"]),
    )

    if patient_distribution:
        logger.info(
            "events_per_record: min=%s; median=%.2f; max=%s",
            safe_count(patient_distribution[0]),
            patient_distribution[1],
            safe_count(patient_distribution[2]),
        )

    if csv_info:
        columns = csv_info["columns"]
        logger.info(
            "hospital_site=%s; monitor_vendor=%s; "
            "per_annotator=%s; surrogate_patient_id=%s",
            "YES" if any(
                "hospital" in c or "site" in c for c in columns
            ) else "NO",
            "YES" if any(
                "vendor" in c or "manufacturer" in c
                or "monitor" in c for c in columns
            ) else "NO",
            "YES" if "annotator" in columns else "NO",
            "YES" if "record" in columns else "NO",
        )
    else:
        logger.info(
            "hospital_site=NA; monitor_vendor=NA; "
            "per_annotator=NA; surrogate_patient_id=NA"
        )

    logger.info("wfdb_header_sample_valid=%s", safe_count(header_valid))
    logger.info("outputs: %s; %s", manifest_path, LOG_FILE)
    logger.info(
        "warnings: %s",
        "; ".join(warnings) if warnings else "none",
    )
    logger.info("=== END ===")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())