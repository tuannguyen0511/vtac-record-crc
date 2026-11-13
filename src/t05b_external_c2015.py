"""P05-T05b external validation. Called once by t05b_final_test.py.

Challenge 2015 data must be present and pass its official SHA256SUMS.
Never refit models or thresholds. No P05 test labels are read by this module.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for _env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_env] = "8"

import csv
import hashlib
import math
from collections import Counter
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError("T05B_EXTERNAL_STOP: " + message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def external_root(cfg):
    root = Path(cfg["project_dir"]).resolve().parent / "data_raw" / "challenge-2015" / "1.0.0"
    folder = root / "training" if (root / "training" / "SHA256SUMS").is_file() else root
    require((folder / "SHA256SUMS").is_file(), "official training/SHA256SUMS missing")
    return folder


def audit_training(folder):
    """Read only public external data; hash every official manifest entry."""
    folder = Path(folder).resolve()
    manifest = folder / "SHA256SUMS"
    checked = 0
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            parts = line.rstrip("\r\n").split(None, 1)
            require(len(parts) == 2 and len(parts[0]) == 64, "invalid SHA256SUMS entry")
            relative = parts[1].lstrip(" *")
            path = (folder / relative).resolve()
            require(path.is_relative_to(folder) and path.is_file(), "missing or unsafe manifest path")
            require(sha256_file(path) == parts[0].lower(), "external manifest SHA mismatch")
            checked += 1
    require(checked >= 1500, "official training SHA256SUMS truncated")
    with (folder / "RECORDS").open("r", encoding="utf-8") as stream:
        records = [line.strip() for line in stream if line.strip()]
    with (folder / "ALARMS").open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    require(len(records) == 750 and len(set(records)) == 750, "expected 750 public records")
    require(len(rows) == 750 and all(len(x) == 3 for x in rows), "ALARMS 750 x 3 schema")
    require(set(row[0] for row in rows) == set(records), "ALARMS and RECORDS disagree")
    require(all(row[2] in ("0", "1") for row in rows), "unknown public label encoding")
    vt = [(name, int(y)) for name, alarm, y in rows if alarm == "Ventricular_Tachycardia"]
    require(len(vt) > 0 and {y for _, y in vt} == {0, 1}, "VT public labels unavailable")
    print(f"C2015_SHA256SUMS=PASS; entries={checked}; recordings={len(records)}", flush=True)
    print("C2015_LABEL_COLUMNS=record,alarm_type,label; types=Ventricular_Tachycardia", flush=True)
    print(f"C2015_VT_RECORDS={len(vt)}; C2015_VT_POSITIVE={sum(y for _, y in vt)}; "
          f"C2015_VT_NEGATIVE={sum(1-y for _, y in vt)}", flush=True)
    return vt


def fingerprint_ii(raw, names, np, scipy_signal, preprocess):
    idx = next((i for i, s in enumerate(names)
                if preprocess.normalized_channel(s) == "II"), None)
    if idx is None:
        return None
    src = np.asarray(raw[:, idx], dtype=np.float64)
    usable = np.isfinite(src)
    if not usable.any():
        return None
    src = np.interp(np.arange(src.size), np.flatnonzero(usable), src[usable])
    vec = scipy_signal.resample_poly(src, 1, 10)
    require(vec.size == 1500, "II fingerprint must be 60s at 25Hz")
    vec -= vec.mean()
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm < 1e-10:
        return None
    return (vec / norm).astype(np.float32)


def read_pre_alarm(base, wfdb, np):
    header = wfdb.rdheader(str(base))
    require(abs(float(header.fs)-250) < 1e-9 and int(header.sig_len) >= 75000,
            "non-250Hz or truncated WFDB recording")
    samples, meta = wfdb.rdsamp(str(base), sampfrom=60000, sampto=75000)
    require(samples.shape[0] == 15000, "WFDB last 60s missing")
    return np.asarray(samples), list(meta["sig_name"])


def prepare_public_vt(cfg, vt, folder, preprocess, np, wfdb, scipy_signal):
    """Public labels and signals only; no P05 sealed input is read."""
    X, M, labels, fingers = [], [], [], []
    known = set(preprocess.SLOTS) | set(preprocess.ECG_LIMB) | set(preprocess.ECG_CHEST) | {"PLETH", "ABP"}
    unknown = Counter()
    for name, y in vt:
        raw, names = read_pre_alarm(folder / name, wfdb, np)
        for n in names:
            if preprocess.normalized_channel(n) not in known:
                unknown[str(n)] += 1
        x, mask = preprocess.preprocess_event(raw, names, 10, 0,
                                               fs=250, t0=75000, start_sample=60000)
        require(x.shape == (4, 2500) and mask.shape == (4,), "external input shape")
        X.append(x)
        M.append(mask)
        labels.append(y)
        fingers.append(fingerprint_ii(raw, names, np, scipy_signal, preprocess))
    print(f"C2015_UNKNOWN_CHANNEL_NAMES={sorted(unknown)}", flush=True)
    return {"x": np.stack(X), "mask": np.stack(M),
            "y": np.asarray(labels, dtype=np.int8), "fingerprints": fingers}


def read_reference_fingerprints(cfg, preprocess, np, pd, wfdb, scipy_signal):
    """Only signal files and unlabeled test_meta; no P05 test labels."""
    rawroot = Path(cfg["raw_dir"]) / "annotated_waveforms"
    procroot = Path(cfg["proc_dir"])
    refs = {}
    for key, splits in (("train", ("fit", "calib")), ("val", ("val",)),
                        ("test", ("test",))):
        samples = []
        for split in splits:
            meta = pd.read_parquet(procroot / f"{split}_meta.parquet", columns=["record", "event"])
            for rec, ev in meta.itertuples(index=False, name=None):
                signal, channels = read_pre_alarm(rawroot / str(rec) / str(ev), wfdb, np)
                fp = fingerprint_ii(signal, channels, np, scipy_signal, preprocess)
                if fp is not None:
                    samples.append(fp)
        refs[key] = np.stack(samples) if samples else np.empty((0, 1500), dtype=np.float32)
        print(f"C2015_REFERENCE_II={key}; usable={len(samples)}", flush=True)
    return refs


def max_normalized_xcorr(q, refs, np):
    """Absolute full-lag normalized FFT cross-correlation, full-vector L2 norms.

    Zero-mean, unit-norm inputs; unbounded lags automatically penalized by
    missing overlap. Inverted leads also match via absolute correlation.
    """
    if q is None or refs.shape[0] == 0:
        return np.zeros(refs.shape[0], dtype=np.float64)
    fftlen = 4096
    a = np.fft.rfft(q, n=fftlen)
    batch = 64
    found = np.empty(refs.shape[0], dtype=np.float64)
    for start in range(0, len(refs), batch):
        chunk = refs[start:start+batch]
        b = np.fft.rfft(chunk, n=fftlen, axis=-1)
        corr = np.fft.irfft(b * np.conj(a)[None, :], n=fftlen, axis=-1)
        found[start:start+batch] = np.max(np.abs(corr), axis=-1)
    return found


def flag_overlap(payload, refs, np):
    flags = np.zeros(len(payload["fingerprints"]), dtype=np.bool_)
    counts = {}
    for key in ("train", "val", "test"):
        n_pairs = 0
        for i, fp in enumerate(payload["fingerprints"]):
            match = max_normalized_xcorr(fp, refs[key], np) > 0.95
            n_pairs += int(match.sum())
            flags[i] |= bool(match.any())
        counts[key] = n_pairs
    return flags, counts


def score_external(payload, logits, locks, tau, metrics, np, removed, privacy):
    y = payload["y"]
    p = np.exp(-np.logaddexp(0.0, -np.asarray(logits)))
    out = []
    for subset, good in (("all", np.ones(len(y), dtype=np.bool_)),
                         ("exclude_overlap", ~removed)):
        use_y, use_p = y[good], p[good]
        count_left = int(good.sum())
        protect = (subset != "all" and (0 < int(removed.sum()) < 10 or
                   count_left < 10 or min(int((use_y == 0).sum()),
                                          int((use_y == 1).sum())) < 10))
        for alpha in (.01, .02, .05):
            lam = float(locks[alpha])
            if protect or len(set(use_y.tolist())) != 2:
                out.append({"subset": subset, "alpha": alpha,
                            "status": "WITHHELD_SMALL_CELL" if protect else "N/A_ONE_CLASS"})
                continue
            negative = use_y == 0
            positive = use_y == 1
            suppression = use_p < lam
            n_true = int(positive.sum())
            n_fn = int(np.count_nonzero(suppression & positive))
            ci = metrics.clopper_pearson(n_fn, n_true)
            out.append({"subset": subset, "alpha": alpha, "status": "OK",
                        "n_records": privacy(len(use_y)), "true": privacy(n_true),
                        "false": privacy(int(negative.sum())),
                        "auroc": float(metrics.auroc(use_p, use_y)),
                        "auprc": float(metrics.auprc(use_p, use_y)),
                        "suppression": float(suppression[negative].mean()),
                        "fnr": n_fn / n_true, "fnr_ci_low": ci[0],
                        "fnr_ci_high": ci[1], "true_suppressed": privacy(n_fn)})
        if not protect and len(set(use_y.tolist())) == 2:
            ltt_suppressed = use_p <= tau
            neg, pos = use_y == 0, use_y == 1
            out.append({"subset": subset, "alpha": "LTT_tau_0.09", "status": "OK",
                        "suppression": float(ltt_suppressed[neg].mean()),
                        "fnr": float(ltt_suppressed[pos].mean())})
        else:
            out.append({"subset": subset, "alpha": "LTT_tau_0.09",
                        "status": "WITHHELD_SMALL_CELL" if protect else "N/A_ONE_CLASS"})
    return out


def run_external(cfg, lock, preflight_payload, model_logits, metrics,
                 np, pd, wfdb, scipy_signal, preprocess, privacy):
    """Use already-verified and precomputed public payload; no calibration."""
    refs = read_reference_fingerprints(cfg, preprocess, np, pd, wfdb, scipy_signal)
    flags, pairs = flag_overlap(preflight_payload, refs, np)
    result = score_external(preflight_payload, model_logits,
                            lock["lambda"]["A"], float(lock["ltt_comparator"]["tau"]),
                            metrics, np, flags, privacy)
    return {"pairs_redacted": {k: privacy(v) for k, v in pairs.items()},
            "records_flagged_redacted": privacy(int(flags.sum())),
            "results": result}
