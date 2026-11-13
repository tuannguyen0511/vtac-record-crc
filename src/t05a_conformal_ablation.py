"""P05-T05a: frozen T03-score analyses and six approved ResNet1D ablations.

No test or sealed-label paths are accessed. Source files, T02c and T03 outputs
remain immutable. Only new T05a run/checkpoint/report artifacts are written.
"""

# Resource isolation must precede ALL imports capable of initializing Torch.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "8"

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import yaml

FROZEN = {
    "src/models.py": "4c082cbe1f14de69fda9b251e94970099fa900b31f37715c0ce156b45d2b070c",
    "src/train_lib.py": "8015e99d240cbc87198e51c3c04cebc3758899f885be13f3d7f7ba9b08c581bc",
    "src/t02c_pilot.py": "7eeac13d574e37e5c0b4692dc603e580c8bd0c0e770d3cce65790a0245db443b",
    "src/t02c_director_full.py": "5148baa7f7976ba582d946f3beffd6bbcd76d2e4233163fe963fe6bec679a22e",
    "configs/t02c.yaml": "a7c705c370dde2b164088b0fbe6af80512a409a8d075a317b772329dd90fbc33",
    "src/t03_label_variants.py": "d24c14151b49a3499d8264cef4245ee29fca517bc824bf28aa45307c0cdf1d19",
    "configs/t03.yaml": "b67672f305d148e8aece5358315d85a2c82de844d33ddc729081babcd51de4b4",
}
MODELS = ("A", "B", "C", "D", "E")
SEEDS = tuple(range(5))
AB_NAMES = ("no_ppg", "no_abp", "ecg_only", "w20_d0", "w10_d5", "w10_d10")
EXPECTED_EVENTS = {"fit": 2830, "val": 495, "calib": 1230}
EXPECTED_RECORDS = {"fit": 1265, "val": 226, "calib": 543}
FILES_PER_RUN = ("ckpt.pt", "scores_val.parquet", "scores_calib.parquet", "log.csv")
APPROVAL = "T05A_AUTHORIZED"


def hard(condition, message):
    if not condition:
        raise RuntimeError("T05A_HARD_STOP: " + str(message))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def spec_expected():
    return {
        "step": "P05-T05a", "approval": APPROVAL,
        "primary_variant": "A", "secondary_variants": ["B", "C", "D", "E"],
        "backbone": "resnet1d", "seeds": list(SEEDS),
        "alpha_levels": [0.005, 0.01, 0.02, 0.05],
        "alpha_primary": 0.02, "cp_confidence": 0.95,
        "ltt_alpha": 0.05, "ltt_delta": 0.05,
        "ltt_grid_low": [0.0005, 0.01, 20],
        "ltt_grid_high": [0.01, 0.40, 40],
        "ltt_grid_unique_count": 59,
        "bootstrap_replicates": 2000, "bootstrap_seed": 42,
        "ablation": [
            {"name": "no_ppg", "w": 10, "d": 0, "zero_channels": [2]},
            {"name": "no_abp", "w": 10, "d": 0, "zero_channels": [3]},
            {"name": "ecg_only", "w": 10, "d": 0, "zero_channels": [2, 3]},
            {"name": "w20_d0", "w": 20, "d": 0, "zero_channels": []},
            {"name": "w10_d5", "w": 10, "d": 5, "zero_channels": []},
            {"name": "w10_d10", "w": 10, "d": 10, "zero_channels": []},
        ],
    }


def preflight(root):
    """Check all frozen SHA digests before importing external project code."""
    for relative, expected in FROZEN.items():
        path = root / relative
        hard(path.is_file(), f"missing frozen file {relative}")
        actual = digest(path)
        hard(actual == expected, f"SHA mismatch {relative}")
        print(f"FROZEN_SHA_PASS={relative}; sha256={actual}", flush=True)
    import t02c_pilot as core
    cfg = core.validate_frozen(root / "configs" / "t02c.yaml")
    hard(Path(cfg["project_dir"]).resolve() == root.resolve(),
         "project path differs from Director locked configuration")
    with Path(cfg["locked_design"]).open("r", encoding="utf-8") as stream:
        locked = yaml.safe_load(stream)
    meta_hashes = locked.get("provenance_sha256", {}).get("meta_files", {})
    for split in ("fit", "calib", "val"):
        filename = f"{split}_meta.parquet"
        hard(filename in meta_hashes, f"frozen metadata digest absent: {split}")
        hard(digest(Path(cfg["proc_dir"]) / filename) == meta_hashes[filename],
             f"frozen metadata SHA mismatch: {split}")
    print("FROZEN_META_FIT_CALIB_VAL_SHA=PASS; TEST_LABELS_NOT_READ", flush=True)
    full_g1 = root / "outputs" / "t02c" / "REPORT_T02c_DIRECTOR_FULL.md"
    g2 = root / "reports" / "P05-T03.md"
    hard(full_g1.is_file() and g2.is_file(), "accepted T02c/T03 report missing")
    hard("G1: PASS." in full_g1.read_text(encoding="utf-8") and
         "Selected backbone: resnet1d" in full_g1.read_text(encoding="utf-8"),
         "G1 not accepted")
    text = g2.read_text(encoding="utf-8")
    hard("T03_REPORT_COMPLETE=TRUE" in text and "**G2: FAIL**" in text,
         "T03 report does not match Director's fallback decision")
    conf_path = root / "configs" / "t05a.yaml"
    with conf_path.open("r", encoding="utf-8") as stream:
        conf = yaml.safe_load(stream)
    hard(conf == spec_expected(), "T05a configuration differs from authorization")
    for variant in MODELS:
        for seed in SEEDS:
            run = root / "outputs" / "t03" / f"{variant}_s{seed}"
            hard(all((run / filename).is_file() for filename in FILES_PER_RUN),
                 f"T03 frozen run incomplete: {variant}_s{seed}")
    source_hashes = {key.replace("/", "_").replace(".", "_"): value
                     for key, value in FROZEN.items()}
    source_hashes["t05a_yaml"] = digest(conf_path)
    source_hashes["t05a_script"] = digest(Path(__file__).resolve())
    source_hashes["locked_design"] = cfg["expected_locked_sha256"]
    source_hashes["common_tree"] = cfg["expected_common_tree_sha256"]
    print("G1=PASS; G2=FAIL; PRIMARY_VARIANT=A; T03_ALL_25=PASS", flush=True)
    print("T05A_PROTOCOL=PASS; SIX_ABLATIONS; FIVE_SEEDS; TEST=SEALED", flush=True)
    return cfg, conf, source_hashes, core


def read_metadata(proc_dir, split, pd, np):
    hard(split in EXPECTED_EVENTS, "metadata split outside fit/val/calib")
    names = ["idx", "record", "y", "contested", "votes_missing"]
    data = pd.read_parquet(Path(proc_dir) / f"{split}_meta.parquet", columns=names)
    hard(list(data.columns) == names, "metadata column mismatch")
    hard(len(data) == EXPECTED_EVENTS[split] and data["idx"].is_unique,
         "metadata event count/idx mismatch")
    hard(data["record"].nunique() == EXPECTED_RECORDS[split],
         "metadata record count mismatch")
    for name in ("y", "contested", "votes_missing"):
        hard(np.isin(data[name].to_numpy(), (0, 1)).all(),
             f"{split} {name} invalid")
    return data


def probabilities(logits, np):
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    hard(len(z) > 0 and np.isfinite(z).all(), "invalid saved logits")
    return np.exp(-np.logaddexp(0.0, -z))


def redact_count(value):
    k = int(value)
    hard(k >= 0, "negative count")
    return "<10" if k < 10 else str(k)


def score_aggregate(logits, y, metrics, np):
    y = np.asarray(y, dtype=np.int8)
    p = probabilities(logits, np)
    return {
        "auroc": float(metrics.auroc(logits, y)),
        "auprc": float(metrics.auprc(logits, y)),
        "ece": float(metrics.ece(p, y, bins=15)),
        "nll": float(np.mean(np.logaddexp(0.0, logits) - y * logits)),
    }


def suppression_stats(prob, y, lam, metrics, np, strict=True):
    if lam is None:
        return None
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    suppressed = p < lam if strict else p <= lam
    pos, neg = y == 1, y == 0
    hard(pos.any() and neg.any(), "val must contain both label classes")
    k = int(np.count_nonzero(suppressed & pos))
    n = int(pos.sum())
    ci = metrics.clopper_pearson(k, n, conf=0.95)
    return {"lambda": float(lam), "fa_suppression": float(np.mean(suppressed[neg])),
            "fnr": float(k / n), "fnr_ci_low": float(ci[0]),
            "fnr_ci_high": float(ci[1]), "true_suppressed": k,
            "true_total": n, "suppressed_total": int(suppressed.sum()),
            "suppressed_positive": k, "mask": suppressed}


def crc_metrics(calib_prob, calib, val_prob, val, alpha,
                conformal, metrics, np, weights=None):
    y_calib = calib["y"].to_numpy(dtype=np.int8)
    records = calib["record"].to_numpy()
    eligible_records = int(calib.loc[calib["y"] == 1, "record"].nunique())
    if 1.0 / (eligible_records + 1) > alpha:
        return None
    lam = float(conformal.crc_fnr_threshold(calib_prob, y_calib, records,
                                             alpha, weights=weights))
    return suppression_stats(val_prob, val["y"].to_numpy(dtype=np.int8),
                             lam, metrics, np)


def ltt_metrics(calib_prob, calib, val_prob, val, conformal, metrics, np):
    grid = np.unique(np.concatenate((
        np.linspace(0.0005, 0.01, 20),
        np.linspace(0.01, 0.40, 40))))
    hard(len(grid) == 59, "LTT candidate grid must have exactly 59 values")
    lam = conformal.ltt_cp_threshold(
        calib_prob, calib["y"].to_numpy(dtype=np.int8),
        alpha=0.05, delta=0.05, grid=grid)
    if lam is None:
        return {"status": "NO_ELIGIBLE_THRESHOLD", "grid_size": 59}
    result = suppression_stats(val_prob, val["y"].to_numpy(dtype=np.int8),
                               lam, metrics, np, strict=False)
    count = result["suppressed_total"]
    result["risk_y1_given_suppressed"] = (
        result["suppressed_positive"] / count if count else None)
    result["status"] = "OK"
    result["grid_size"] = 59
    return result


def h2_counts(val_prob, val, outcome, np):
    """H2 excludes votes_missing=1 and reports counts only (no record IDs)."""
    if outcome is None:
        return None
    valid = (val["votes_missing"].to_numpy(dtype=np.int8) == 0) & (
        val["y"].to_numpy(dtype=np.int8) == 1)
    contested = val["contested"].to_numpy(dtype=np.int8)
    suppressed = val_prob < outcome["lambda"]
    result = {}
    for status in (0, 1):
        selected = valid & (contested == status)
        result[str(status)] = {
            "total": int(selected.sum()),
            "suppressed": int(np.count_nonzero(selected & suppressed)),
        }
    return result


def bootstrap_crc(calib_prob, calib, val_prob, val,
                  conformal, metrics, np):
    """2000 calibration RECORD resamples, unique ID for each drawn record."""
    ids = calib["record"].to_numpy()
    unique, inverse = np.unique(ids, return_inverse=True)
    index_by_record = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(42)
    labels = calib["y"].to_numpy(dtype=np.int8)
    rows = []
    for rep in range(2000):
        sampled = rng.integers(0, len(unique), size=len(unique))
        row_ids = np.concatenate([index_by_record[i] for i in sampled])
        # Repeated sampled records count as different resampled records.
        drawn_groups = np.repeat(np.arange(len(sampled)),
                                 [len(index_by_record[i]) for i in sampled])
        ys = labels[row_ids]
        scores = calib_prob[row_ids]
        npos_records = int(np.unique(drawn_groups[ys == 1]).size)
        hard(1 / (npos_records + 1) <= 0.02,
             "record bootstrap makes primary CRC infeasible")
        lam = float(conformal.crc_fnr_threshold(
            scores, ys, drawn_groups, alpha=0.02))
        ev = suppression_stats(val_prob, val["y"].to_numpy(dtype=np.int8),
                               lam, metrics, np)
        rows.append({"replicate": rep, "lambda": lam,
                     "fa_suppression_val": ev["fa_suppression"]})
    return rows


def five_stats(values, np):
    a = np.asarray(values, dtype=np.float64)
    hard(a.shape == (5,) and np.isfinite(a).all(), "incomplete five-seed statistics")
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "sd": float(a.std(ddof=1))}


def analyse_frozen_t03(root, cfg, np, pd, metrics, conformal, t03):
    """No Torch/GPU/training. Read only T03 val/calib score files."""
    val = read_metadata(cfg["proc_dir"], "val", pd, np)
    calib = read_metadata(cfg["proc_dir"], "calib", pd, np)
    yi = val["idx"].to_numpy()
    ci = calib["idx"].to_numpy()
    t03_logits = {}
    for name in MODELS:
        t03_logits[name] = {}
        for seed in SEEDS:
            run = root / "outputs" / "t03" / f"{name}_s{seed}"
            t03_logits[name][seed] = {
                "val": t03.align_scores(run / "scores_val.parquet", yi,
                                        pd, np, f"T03/{name}_s{seed}/val"),
                "calib": t03.align_scores(run / "scores_calib.parquet", ci,
                                          pd, np, f"T03/{name}_s{seed}/calib"),
            }

    ensemble = {}
    for name in MODELS:
        ensemble[name] = {
            split: np.mean(np.stack([t03_logits[name][seed][split]
                                    for seed in SEEDS]), axis=0)
            for split in ("val", "calib")
        }

    ya = val["y"].to_numpy(dtype=np.int8)
    pc = probabilities(ensemble["A"]["calib"], np)
    pv = probabilities(ensemble["A"]["val"], np)
    alphas = (0.005, 0.01, 0.02, 0.05)
    primary = {a: crc_metrics(pc, calib, pv, val, a,
                              conformal, metrics, np) for a in alphas}
    hard(primary[0.02] is not None, "primary CRC at alpha=0.02 infeasible")
    ltt = ltt_metrics(pc, calib, pv, val, conformal, metrics, np)

    weighted = {}
    weights = np.where(calib["contested"].to_numpy(dtype=np.int8) == 1,
                       0.5, 1.0).astype(np.float64)
    for alpha in alphas:
        weighted[alpha] = crc_metrics(pc, calib, pv, val, alpha,
                                      conformal, metrics, np, weights=weights)

    h2 = []
    for alpha in alphas:
        for method, obj in (("unweighted", primary[alpha]),
                            ("weighted", weighted[alpha])):
            group = h2_counts(pv, val, obj, np)
            for c in (0, 1):
                h2.append({
                    "alpha": alpha, "method": method, "contested": c,
                    "positive_count": None if group is None else group[str(c)]["total"],
                    "suppressed_count": None if group is None else group[str(c)]["suppressed"],
                })

    stability = []
    for alpha in alphas:
        single = []
        for seed in SEEDS:
            p_val = probabilities(t03_logits["A"][seed]["val"], np)
            p_calib = probabilities(t03_logits["A"][seed]["calib"], np)
            single.append(crc_metrics(p_calib, calib, p_val, val, alpha,
                                      conformal, metrics, np))
        if any(v is None for v in single):
            hard(all(v is None for v in single) and primary[alpha] is None,
                 "inconsistent CRC alpha feasibility")
            stability.append({"alpha": alpha, "status": "INFEASIBLE"})
            continue
        item = {"alpha": alpha, "status": "OK"}
        for key in ("lambda", "fa_suppression", "fnr"):
            item.update({key + "_" + name: value for name, value in
                         five_stats([v[key] for v in single], np).items()})
            item[key + "_ensemble"] = primary[alpha][key]
        stability.append(item)

    secondary = []
    for variant in MODELS:
        pcv = probabilities(ensemble[variant]["calib"], np)
        pvv = probabilities(ensemble[variant]["val"], np)
        row = score_aggregate(ensemble[variant]["val"], ya, metrics, np)
        summary = crc_metrics(pcv, calib, pvv, val, 0.02,
                              conformal, metrics, np)
        hard(summary is not None, "secondary CRC alpha infeasible")
        secondary.append({"variant": variant, **row,
                          "lambda": summary["lambda"],
                          "fa_suppression": summary["fa_suppression"],
                          "fnr": summary["fnr"]})

    bootstrap = bootstrap_crc(pc, calib, pv, val, conformal, metrics, np)
    print("T05A_T03_FROZEN_SCORES=PASS; ENSEMBLES=5; TEST=SEALED", flush=True)
    print("T05A_CRC_ALPHA_PRIMARY=0.02; LTT_GRID=59; BOOTSTRAP=2000", flush=True)
    return {"val": val, "calib": calib, "primary": primary, "weighted": weighted,
            "ltt": ltt, "h2": h2, "stability": stability,
            "secondary": secondary, "bootstrap": bootstrap,
            "ensemble": ensemble, "t03_logits": t03_logits}


def cache_data(proc_dir, split, w, d, labels, train_lib, torch, np, pd):
    """Load only explicitly approved, preprocessed fit/val/calib caches."""
    hard(split in EXPECTED_EVENTS and (w, d) in
         ((10, 0), (20, 0), (10, 5), (10, 10)),
         "cache request outside approved splits/windows")
    if (w, d) == (10, 0):
        return train_lib.load_split(proc_dir, split, labels=labels)
    root = Path(proc_dir)
    n = EXPECTED_EVENTS[split]
    length = (w + d) * 250
    x = np.load(root / f"{split}_W{w}_D{d}_x.npy", allow_pickle=False)
    mask = np.load(root / f"{split}_W{w}_D{d}_mask.npy", allow_pickle=False)
    columns = ["idx", "record", "y"] if labels else ["idx", "record"]
    meta = pd.read_parquet(root / f"{split}_meta.parquet", columns=columns)
    hard(list(meta.columns) == columns and meta["idx"].is_unique,
         f"W{w}D{d} {split} cache/meta schema")
    hard(meta["record"].nunique() == EXPECTED_RECORDS[split],
         f"W{w}D{d} {split} record count")
    hard(x.shape == (n, 4, length) and mask.shape == (n, 4) and
         len(meta) == n and x.dtype == np.float32 and mask.dtype == np.bool_,
         f"W{w}D{d} {split} cached shape/dtype")
    hard(np.isfinite(x).all() and np.max(np.abs(x)) <= 10.000001,
         f"W{w}D{d} {split} invalid signal values")
    if labels:
        y = meta["y"].to_numpy(dtype=np.float32)
        hard(np.isin(y, (0, 1)).all(), f"W{w}D{d} {split} labels")
    else:
        y = None
    return {
        "idx": meta["idx"].to_numpy(copy=True),
        "x": torch.from_numpy(np.array(x, copy=True, order="C")),
        "mask": torch.from_numpy(np.array(mask, copy=True, order="C")),
        "y": torch.from_numpy(y.copy()) if labels else None,
    }


def remove_channels(data, zero_channels):
    """Set both waveform and availability mask to zero without mutating cache."""
    if not zero_channels:
        return data
    out = dict(data)
    out["x"] = data["x"].clone()
    out["mask"] = data["mask"].clone()
    out["x"][:, zero_channels, :] = 0.0
    out["mask"][:, zero_channels] = False
    return out


def manifest_once(cfg, entries, stage="P05-T05a"):
    path = Path(cfg["project_dir"]) / "manifest.csv"
    prior = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                hard(row.get("path") and row.get("sha256"), "manifest row invalid")
                prior.setdefault(str(Path(row["path"]).resolve()).lower(),
                                 set()).add(row["sha256"])
    new = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for artifact, sample_count in entries:
        artifact = Path(artifact).resolve()
        hard(artifact.is_file(), f"manifest missing artifact {artifact.name}")
        sha = digest(artifact)
        key = str(artifact).lower()
        old = prior.get(key, set())
        hard(not old or old == {sha},
             f"manifest existing artifact hash changed: {artifact.name}")
        if not old:
            new.append((str(artifact), sha, int(sample_count), timestamp, stage))
            prior[key] = {sha}
    if new:
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if path.stat().st_size == 0:
                writer.writerow(["path", "sha256", "n_samples", "timestamp_utc", "step"])
            writer.writerows(new)
    print(f"T05A_MANIFEST_NEW_ROWS={len(new)}; PASS", flush=True)


def saved_run(run_dir, run_sha, seed, cfg, fit, val, calib,
              train_lib, metrics, pd, np):
    hard(train_lib.completed_run(run_dir), f"ablation run incomplete: {run_dir.name}")
    result = train_lib.load_completed_result(run_dir, "resnet1d", seed, run_sha,
                                             cfg["expected_locked_sha256"])
    log = pd.read_csv(run_dir / "log.csv")
    hard(1 <= result["best_epoch"] <= result["epochs_run"] <= 150 and
         len(log) == result["epochs_run"], "ablation epoch audit failed")
    hard(np.isfinite(log["train_loss"].to_numpy(dtype=float)).all() and
         np.isfinite(log["val_auprc"].to_numpy(dtype=float)).all(),
         "ablation NaN/Inf training values")
    val_z = align_run(run_dir / "scores_val.parquet", val["idx"], pd, np)
    calib_z = align_run(run_dir / "scores_calib.parquet", calib["idx"], pd, np)
    new_metrics = score_aggregate(val_z, val["y"].numpy(), metrics, np)
    for key in ("auroc", "auprc", "ece"):
        hard(abs(new_metrics[key] - result["metrics"][key]) < 1e-7,
             "ablation saved/independent metric mismatch")
    return {"val_z": val_z, "calib_z": calib_z,
            "train": result, "metrics": new_metrics}


def align_run(path, idx, pd, np):
    df = pd.read_parquet(path, columns=["idx", "logit"])
    hard(list(df.columns) == ["idx", "logit"] and len(df) == len(idx) and
         df["idx"].is_unique, "ablation scores invalid")
    index = df.set_index("idx")["logit"]
    hard(pd.Index(idx).isin(index.index).all(), "ablation scores idx mismatch")
    z = index.reindex(pd.Index(idx)).to_numpy(dtype=np.float64)
    hard(np.isfinite(z).all(), "ablation nonfinite scores")
    return z


def run_ablations(root, cfg, conf, sha, core,
                  metrics, conformal, np, pd, torch, train_lib):
    """30 distinct runs. Temporary in-memory adapter supplies matching calib.

    The frozen train_lib.train_model is called verbatim. Its internal
    load_split('calib') is redirected to this ablation's own preloaded,
    label-free calibration cache. The train_lib.py file is never edited.
    """
    device = core.gpu_smoke_test(torch, __import__("models"), train_lib, cfg)
    output = root / "outputs" / "t05a" / "ablation"
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    run_started = time.perf_counter()
    original_load = train_lib.load_split
    for ab in conf["ablation"]:
        name, w, d, zero = (ab["name"], ab["w"], ab["d"], ab["zero_channels"])
        fit = remove_channels(cache_data(cfg["proc_dir"], "fit", w, d,
                                         True, train_lib, torch, np, pd), zero)
        val = remove_channels(cache_data(cfg["proc_dir"], "val", w, d,
                                         True, train_lib, torch, np, pd), zero)
        calib = remove_channels(cache_data(cfg["proc_dir"], "calib", w, d,
                                           False, train_lib, torch, np, pd), zero)
        fit_target = fit["y"].numpy().astype(np.float32, copy=False)
        run_sha = {
            **sha, "t05a_ablation": name, "t05a_w": w, "t05a_d": d,
            "t05a_zero_channels": list(zero),
            "t05a_fit_y_sha256": hashlib.sha256(fit_target.tobytes()).hexdigest(),
            "t05a_fit_len": int(len(fit["idx"])),
            "t05a_sequence_length": int(fit["x"].shape[-1]),
        }

        def matching_calib(proc_dir, split, *, w=10, d=0, labels=True):
            hard(split == "calib" and not labels and (w, d) == (10, 0) and
                 Path(proc_dir).resolve() == Path(cfg["proc_dir"]).resolve(),
                 "train_lib requested unexpected split while ablation adapter active")
            return calib

        for seed in SEEDS:
            run_dir = output / f"{name}_s{seed}"
            if train_lib.completed_run(run_dir):
                print(f"T05A_ABLATION_SKIP_COMPLETE={name}_s{seed}", flush=True)
            else:
                print(f"T05A_ABLATION_RUN_BEGIN={name}_s{seed}; W={w}; D={d}",
                      flush=True)
                hard(train_lib.load_split is original_load,
                     "training adapter was not restored")
                train_lib.load_split = matching_calib
                try:
                    train_lib.train_model(
                        "resnet1d", seed, fit, val, cfg, device, metrics,
                        output_dir=run_dir, sha=run_sha)
                finally:
                    train_lib.load_split = original_load
            result = saved_run(run_dir, run_sha, seed, cfg, fit, val, calib,
                               train_lib, metrics, pd, np)
            manifest_once(cfg, [
                (run_dir / "ckpt.pt", 0),
                (run_dir / "scores_val.parquet", EXPECTED_EVENTS["val"]),
                (run_dir / "scores_calib.parquet", EXPECTED_EVENTS["calib"]),
                (run_dir / "log.csv", result["train"]["epochs_run"]),
            ])
            runs[(name, seed)] = result
            print(f"T05A_ABLATION_VERIFIED={name}_s{seed}; "
                  f"best_epoch={result['train']['best_epoch']}; "
                  f"epochs={result['train']['epochs_run']}; PASS", flush=True)
        del fit, val, calib
        print(f"T05A_ABLATION_GROUP={name}; SEEDS=5/5; PASS", flush=True)
    hard(train_lib.load_split is original_load, "calibration adapter not restored")
    hard(len(runs) == 30, "expected precisely 30 approved ablation runs")
    print(f"T05A_ABLATION_COMPLETED=30/30; "
          f"ORCHESTRATOR_WALL_H={(time.perf_counter()-run_started)/3600:.5f}",
          flush=True)

    val_meta = read_metadata(cfg["proc_dir"], "val", pd, np)
    calib_meta = read_metadata(cfg["proc_dir"], "calib", pd, np)
    val_y = val_meta["y"].to_numpy(dtype=np.int8)
    summaries = []
    for name in AB_NAMES:
        m = [runs[(name, s)] for s in SEEDS]
        mval = np.mean(np.stack([x["val_z"] for x in m]), axis=0)
        mcalib = np.mean(np.stack([x["calib_z"] for x in m]), axis=0)
        e_metrics = score_aggregate(mval, val_y, metrics, np)
        ens = crc_metrics(probabilities(mcalib, np), calib_meta,
                          probabilities(mval, np), val_meta, 0.02,
                          conformal, metrics, np)
        hard(ens is not None, "ablation primary alpha infeasible")
        row = {"name": name, "window": "W" + str(next(a["w"] for a in conf["ablation"] if a["name"] == name)) +
               "D" + str(next(a["d"] for a in conf["ablation"] if a["name"] == name))}
        for key in ("auroc", "auprc", "ece", "nll"):
            row[key + "_mean"] = float(np.mean([x["metrics"][key] for x in m]))
            row[key + "_sd"] = float(np.std([x["metrics"][key] for x in m], ddof=1))
            row[key + "_ensemble"] = e_metrics[key]
        sevals = [crc_metrics(probabilities(x["calib_z"], np), calib_meta,
                              probabilities(x["val_z"], np), val_meta, 0.02,
                              conformal, metrics, np) for x in m]
        for key in ("fa_suppression", "fnr", "lambda"):
            row[key + "_mean"] = float(np.mean([o[key] for o in sevals]))
            row[key + "_sd"] = float(np.std([o[key] for o in sevals], ddof=1))
            row[key + "_ensemble"] = ens[key]
        row["true_suppressed_ensemble"] = redact_count(ens["true_suppressed"])
        row["gpu_train_wall_h"] = sum(
            x["train"]["wall_seconds"] for x in m) / 3600.0
        row["max_epochs_reached"] = redact_count(sum(
            x["train"]["epochs_run"] == 150 for x in m))
        summaries.append(row)
    return runs, summaries


def public_primary_rows(data):
    rows = []
    for alpha in (0.005, 0.01, 0.02, 0.05):
        o = data["primary"][alpha]
        rows.append({
            "alpha": alpha,
            "status": "INFEASIBLE" if o is None else "OK",
            "lambda": None if o is None else o["lambda"],
            "fa_suppression_val": None if o is None else o["fa_suppression"],
            "fnr_val": None if o is None else o["fnr"],
            "fnr_ci_low": None if o is None else o["fnr_ci_low"],
            "fnr_ci_high": None if o is None else o["fnr_ci_high"],
            "true_suppressed_val": None if o is None else redact_count(o["true_suppressed"]),
        })
    return rows


def public_weighted_rows(data):
    rows = []
    for alpha in (0.005, 0.01, 0.02, 0.05):
        o = data["weighted"][alpha]
        rows.append({"alpha": alpha,
                     "status": "INFEASIBLE" if o is None else "OK",
                     "lambda": None if o is None else o["lambda"],
                     "fa_suppression_val": None if o is None else o["fa_suppression"],
                     "fnr_val": None if o is None else o["fnr"],
                     "true_suppressed_val": None if o is None else redact_count(o["true_suppressed"])})
    return rows


def public_h2_rows(data):
    rows = []
    for row in data["h2"]:
        rows.append({**row,
                     "positive_count": None if row["positive_count"] is None
                     else redact_count(row["positive_count"]),
                     "suppressed_count": None if row["suppressed_count"] is None
                     else redact_count(row["suppressed_count"])})
    return rows


def report_text(cfg, conf, hashes, data, ablations, runs, np):
    def f(value, n=6):
        return "N/A" if value is None else f"{float(value):.{n}f}"

    primary = public_primary_rows(data)
    weighted = public_weighted_rows(data)
    h2 = public_h2_rows(data)
    boot = data["bootstrap"]
    b_lam = np.asarray([r["lambda"] for r in boot], dtype=np.float64)
    b_sup = np.asarray([r["fa_suppression_val"] for r in boot], dtype=np.float64)
    lines = [
        "# REPORT P05-T05a: Frozen-Score Conformal Analysis and Ablations", "",
        "Status: COMPLETE; AWAIT DIRECTOR APPROVAL BEFORE T05b.",
        "Director authorization: T05A_AUTHORIZED (2026-11-11).",
        "G1=PASS (resnet1d); G2=FAIL; primary_variant=A unchanged.",
        "Secondary variants B,C,D,E are preserved without retraining or re-selection.",
        "All 25 T03 runs, 9 T02c runs and frozen scripts remain unchanged.",
        "Test data and labels: SEALED; NOT READ.",
        "Only fit/val/calib used. All val risks are empirical estimates.",
        "CRC calibration uses positive-record risk; suppression strictly score < lambda.",
        "LTT suppression uses score <= tau and a 59-threshold grid; risks differ.",
        "All probabilities are sigmoid(logit), ensembles average logits before sigmoid.",
        "CI: exact two-sided Clopper-Pearson 95% for validation FNR.",
        "Small aggregate counts <10 are redacted in this report and summary Parquet.",
        "H2 excludes events with votes_missing=1; weighted CRC uses w=1/0.5 by contested.",
        "Frozen train_lib source is never modified. For each ablation an in-memory,",
        "label-free calibration-data adapter supplies the matching W,D and channel mask",
        "to train_lib.train_model; the original loader is restored after each run.",
        "No architecture, learning-rate, patience, or early-stopping change.",
        "Code SHA256: `" + json.dumps(hashes, sort_keys=True) + "`", "",
        "## 1. Primary A, record-level CRC across alpha", "",
        "Feasibility for alpha=0.005 requires 1/(n_positive_calib_records+1) <=0.005.",
        "| alpha | lambda | FA suppression val | FNR val | FNR 95% CI | True alarms suppressed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in primary:
        ci = ("N/A" if r["status"] != "OK" else
              f"[{f(r['fnr_ci_low'])}, {f(r['fnr_ci_high'])}]")
        lines.append(f"| {r['alpha']:.3f} | {f(r['lambda'],8)} | "
                     f"{f(r['fa_suppression_val'])} | {f(r['fnr_val'])} | {ci} | "
                     f"{r['true_suppressed_val'] or 'N/A'} |")

    lines += ["", "## 2. LTT at alpha=0.05, delta=0.05, 59 threshold candidates", "",
              "LTT estimates the conditional risk P(y=1 | suppressed); CRC controls",
              "record-level FNR risk. These two risks are not interchangeable.",
              "| Method | Threshold | FA suppression val | FNR val | P(y=1 | suppressed) val | True suppressed |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    ltt = data["ltt"]
    if ltt["status"] == "OK":
        lines.append(f"| LTT | {f(ltt['lambda'],8)} | {f(ltt['fa_suppression'])} | "
                     f"{f(ltt['fnr'])} | {f(ltt['risk_y1_given_suppressed'])} | "
                     f"{redact_count(ltt['true_suppressed'])} |")
    else:
        lines.append("| LTT (no eligible threshold) | N/A | N/A | N/A | N/A | N/A |")
    c05 = data["primary"][0.05]
    cp = ("N/A" if c05["suppressed_total"] == 0 else
          f(c05["suppressed_positive"] / c05["suppressed_total"]))
    lines.append(f"| CRC at alpha=0.05 | {f(c05['lambda'],8)} | "
                 f"{f(c05['fa_suppression'])} | {f(c05['fnr'])} | {cp} | "
                 f"{redact_count(c05['true_suppressed'])} |")

    lines += ["", "## 3. H2 counts by contested status, unweighted vs weighted CRC", "",
              "H2 reports COUNTS ONLY; votes_missing=1 excluded. `<10` means suppressed",
              "or total count is smaller than 10; it does not mean zero.",
              "| alpha | CRC | contested | True count | Suppressed true count |",
              "| ---: | --- | ---: | ---: | ---: |"]
    for r in h2:
        lines.append(f"| {r['alpha']:.3f} | {r['method']} | {r['contested']} | "
                     f"{r['positive_count'] if r['positive_count'] is not None else 'N/A'} | "
                     f"{r['suppressed_count'] if r['suppressed_count'] is not None else 'N/A'} |")
    lines += ["", "Weighted CRC overall validation outcomes (descriptive):", "",
              "| alpha | lambda weighted | FA suppression val | FNR val | True suppressed |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for r in weighted:
        lines.append(f"| {r['alpha']:.3f} | {f(r['lambda'],8)} | "
                     f"{f(r['fa_suppression_val'])} | {f(r['fnr_val'])} | "
                     f"{r['true_suppressed_val'] or 'N/A'} |")

    lines += ["", "## 4. Pre-registered six ablations, primary A only", "",
              "Cells show five-seed mean +/- sample SD followed by ensemble value.",
              "| Ablation | Window | AUROC | AUPRC | ECE | NLL | FA suppression at alpha=0.02 | FNR at alpha=0.02 | GPU training h |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in ablations:
        vals = []
        for key in ("auroc", "auprc", "ece", "nll", "fa_suppression", "fnr"):
            vals.append(f"{r[key+'_mean']:.4f} +/- {r[key+'_sd']:.4f} / "
                        f"{r[key+'_ensemble']:.4f}")
        lines.append(f"| {r['name']} | {r['window']} | " +
                     " | ".join(vals) + f" | {r['gpu_train_wall_h']:.4f} |")
    total_gpu_h = sum(r["gpu_train_wall_h"] for r in ablations)
    lines += ["", f"30 ablation runs verified: 30/30.",
              f"Total ablation GPU training wall hours: {total_gpu_h:.5f}.",
              "No T02c/T03 run was overwritten; every new ablation uses frozen training code.", "",
              "## 5. Primary A CRC stability: five individual seeds vs ensemble", "",
              "Min/max/SD are from individual seed results; ensemble is independent.",
              "| alpha | Metric | Individual min | Individual max | Individual SD | Ensemble |",
              "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in data["stability"]:
        for key in ("lambda", "fa_suppression", "fnr"):
            lines.append(f"| {r['alpha']:.3f} | {key} | "
                         f"{f(r.get(key+'_min'),8)} | {f(r.get(key+'_max'),8)} | "
                         f"{f(r.get(key+'_sd'),8)} | {f(r.get(key+'_ensemble'),8)} |")

    lines += ["", "Record-bootstrap on primary A ensemble calibration scores:",
              "2000 replicates, seed 42; calibration records sampled with replacement.",
              "Each duplicate draw receives a new record group ID before CRC.",
              "| Statistic | Mean | SD | 2.5% | Median | 97.5% |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key, vals in (("lambda", b_lam), ("FA suppression val", b_sup)):
        q = np.quantile(vals, [0.025, 0.5, 0.975])
        lines.append(f"| {key} | {vals.mean():.8f} | {vals.std(ddof=1):.8f} | "
                     f"{q[0]:.8f} | {q[1]:.8f} | {q[2]:.8f} |")

    lines += ["", "## 6. Secondary variants retained, NOT eligible for re-selection", "",
              "These rows use the original T03 five-seed ensembles, without retraining.",
              "| Variant | AUROC val | AUPRC val | ECE val | NLL val | FA suppression at alpha=0.02 | FNR val |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in data["secondary"]:
        lines.append(f"| {r['variant']} | {f(r['auroc'])} | {f(r['auprc'])} | "
                     f"{f(r['ece'])} | {f(r['nll'])} | "
                     f"{f(r['fa_suppression'])} | {f(r['fnr'])} |")
    lines += ["", "## 7. Configuration proposal and hold", "",
              "Proposal: retain primary_variant=A, backbone=resnet1d, W=10, D=0,",
              "alpha=0.02 as the locked baseline. Preserve B-E as secondary analyses.",
              "The six ablations are descriptive; no ablation or threshold is selected",
              "automatically on validation after seeing their outcomes.",
              "All decisions to lock a different configuration require Director review.",
              "**STOP: T05b is NOT authorized. Keep all test labels sealed.**",
              "", "## Decisions made only for technical compatibility", "",
              "For non-W10D0 ablations, T05a loads the already-preprocessed cache",
              "and temporarily injects the matching label-free calibration tensor",
              "into the frozen trainer. Source files remain byte-identical.",
              "For channel ablations, waveform and channel-availability mask are zeroed",
              "in independent fit, val and calib data copies.",
              "No raw signal processing or new data split is performed.",
    ]
    return "\n".join(lines) + "\n"


def save_parquet_once(path, data, pd):
    path = Path(path)
    frame = pd.DataFrame(data)
    if path.exists():
        old = pd.read_parquet(path)
        try:
            pd.testing.assert_frame_equal(old, frame, check_dtype=False)
        except AssertionError as exc:
            raise RuntimeError("T05A_HARD_STOP: existing output differs, refusing overwrite: " + path.name) from exc
        print(f"T05A_OUTPUT_REUSED={path.name}", flush=True)
        return
    frame.to_parquet(path, index=False)
    print(f"T05A_OUTPUT_SAVED={path.name}; rows={len(frame)}", flush=True)


def output_all(root, cfg, hashes, conf, analysis, runs, ablations, pd, np):
    output = root / "outputs" / "t05a"
    output.mkdir(parents=True, exist_ok=True)
    basic = {
        "primary_crc.parquet": public_primary_rows(analysis),
        "weighted_crc.parquet": public_weighted_rows(analysis),
        "h2_counts_redacted.parquet": public_h2_rows(analysis),
        "crc_stability.parquet": analysis["stability"],
        "secondary_variants.parquet": analysis["secondary"],
        "bootstrap_primary.parquet": analysis["bootstrap"],
        "ablation_summary.parquet": ablations,
    }
    ltt = analysis["ltt"]
    basic["ltt_comparison.parquet"] = [{
        "status": ltt["status"], "grid_count": 59,
        "lambda": ltt.get("lambda"),
        "fa_suppression": ltt.get("fa_suppression"),
        "fnr": ltt.get("fnr"),
        "risk_y1_given_suppressed": ltt.get("risk_y1_given_suppressed"),
        "true_suppressed": None if ltt["status"] != "OK" else redact_count(ltt["true_suppressed"]),
    }]
    for filename, records in basic.items():
        save_parquet_once(output / filename, records, pd)
    manifest_once(cfg, [(output / filename, len(rows)) for filename, rows in basic.items()])

    report = root / "reports" / "P05-T05a.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    rendered = report_text(cfg, conf, hashes, analysis, ablations, runs, np)
    if report.exists():
        hard(report.read_text(encoding="utf-8") == rendered,
             "existing T05a report differs, preserving original")
        print(f"T05A_REPORT_REUSED={report}", flush=True)
    else:
        with report.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        print(f"T05A_REPORT_SAVED={report}", flush=True)
    manifest_once(cfg, [(report, 0)])
    return report


def full(root, cfg, conf, hashes, core):
    core.below_normal_priority()
    import numpy as np
    import pandas as pd
    import torch
    import models
    import train_lib
    import t03_label_variants as t03

    sys.path.insert(0, str(Path(cfg["common_dir"]).resolve().parent))
    from common import metrics, conformal
    print(f"PYTHON={sys.version.split()[0]}; TORCH={torch.__version__}; "
          "FP32=TRUE; AMP=FALSE; CPU_THREADS<=8", flush=True)
    started = time.perf_counter()
    analysis = analyse_frozen_t03(root, cfg, np, pd, metrics, conformal, t03)
    runs, ablations = run_ablations(root, cfg, conf, hashes, core,
                                    metrics, conformal, np, pd, torch, train_lib)
    report = output_all(root, cfg, hashes, conf, analysis, runs, ablations, pd, np)
    print("=== REPORT P05-T05a ===", flush=True)
    print("status: COMPLETE; WAIT_FOR_DIRECTOR", flush=True)
    print("pc: reference-workstation; physical_gpu: GPU1; CPU_THREADS<=8; priority: BelowNormal", flush=True)
    print(f"runtime_wall_h: {(time.perf_counter()-started)/3600:.5f}", flush=True)
    print(f"script: t05a_conformal_ablation.py; sha256_8={hashes['t05a_script'][:8]}", flush=True)
    print("data: fit/val/calib W10D0 plus approved ablation caches", flush=True)
    print("frozen_sources: PASS; T03_SCORES: PASS; T03_OUTPUTS: UNCHANGED", flush=True)
    print("analysis: CRC, LTT(59), H2, CRC stability, record bootstrap(2000)", flush=True)
    print("ablation: 6 configs x 5 seeds = 30/30; PASS", flush=True)
    print("primary_variant: A; backbone: resnet1d; no variant re-selection", flush=True)
    print(f"outputs: {root / 'outputs' / 't05a'}", flush=True)
    print(f"report: {report}", flush=True)
    print("warnings: T05b NOT AUTHORIZED; TEST SEALED", flush=True)
    print("=== END ===", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "full"), required=True)
    parser.add_argument("--director-approval", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg, conf, hashes, core = preflight(root)
    if args.stage == "inspect":
        print("T05A_INSPECT=PASS; TRAINING=NOT_STARTED; TEST=SEALED", flush=True)
        return 0
    hard(args.director_approval == APPROVAL, "Director approval token absent")
    return full(root, cfg, conf, hashes, core)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("T05A_INTERRUPTED; completed ablation runs reusable; test sealed",
              flush=True)
        sys.exit(130)
    except Exception as exc:
        print("=== REPORT P05-T05a ===", flush=True)
        print("status: FAIL_TECHNICAL; frozen sources and existing outputs preserved", flush=True)
        if str(exc).startswith("T05A_HARD_STOP:"):
            print(str(exc), flush=True)
        else:
            print(f"error_type: {type(exc).__name__}", flush=True)
        traceback.print_exc()
        print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
        print("=== END ===", flush=True)
        sys.exit(1)
