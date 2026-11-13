"""P05-T03: five fixed label variants x five seeds, validation-only G2.

Director approval: T03_AUTHORIZED (2026-11-11). This entry point never
accesses test files and never changes the frozen training library or models.

    python -u P05_VTaC_FalseAlarm/src/t03_label_variants.py --stage inspect
    python -u P05_VTaC_FalseAlarm/src/t03_label_variants.py --stage full \
        --director-approval T03_AUTHORIZED
"""

# Set GPU isolation and resource limits before any torch import.
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[_env_key] = "8"

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


EXPECTED_HASHES = {
    "models.py": "4c082cbe1f14de69fda9b251e94970099fa900b31f37715c0ce156b45d2b070c",
    "train_lib.py": "8015e99d240cbc87198e51c3c04cebc3758899f885be13f3d7f7ba9b08c581bc",
    "t02c_pilot.py": "7eeac13d574e37e5c0b4692dc603e580c8bd0c0e770d3cce65790a0245db443b",
    "t02c_director_full.py": "5148baa7f7976ba582d946f3beffd6bbcd76d2e4233163fe963fe6bec679a22e",
    "t02c.yaml": "a7c705c370dde2b164088b0fbe6af80512a409a8d075a317b772329dd90fbc33",
}
VARIANTS = ("A", "B", "C", "D", "E")
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.01, 0.02, 0.05)
MODEL = "resnet1d"
EXPECTED_EVENTS = {"fit": 2830, "val": 495, "calib": 1230}
EXPECTED_RECORDS = {"fit": 1265, "val": 226, "calib": 543}
SCORE_FILES = ("ckpt.pt", "scores_val.parquet", "scores_calib.parquet", "log.csv")


def hard(ok, reason):
    if not ok:
        raise RuntimeError("T03_HARD_STOP: " + str(reason))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(root):
    """Verify all locks before importing torch or reading any PROC file."""
    paths = {
        "models.py": root / "src" / "models.py",
        "train_lib.py": root / "src" / "train_lib.py",
        "t02c_pilot.py": root / "src" / "t02c_pilot.py",
        "t02c_director_full.py": root / "src" / "t02c_director_full.py",
        "t02c.yaml": root / "configs" / "t02c.yaml",
    }
    for name, path in paths.items():
        hard(path.is_file(), f"frozen source missing: {name}")
        actual = sha256_file(path)
        hard(actual == EXPECTED_HASHES[name], f"frozen SHA mismatch: {name}")
        print(f"FROZEN_SHA256_{name.replace('.', '_').upper()}={actual}; PASS", flush=True)

    # Import only AFTER its exact approved digest has been checked.
    import t02c_pilot as frozen
    cfg = frozen.validate_frozen(paths["t02c.yaml"])
    hard(Path(cfg["project_dir"]).resolve() == root.resolve(),
         "project path does not match frozen T02c config")

    director_report = root / "outputs" / "t02c" / "REPORT_T02c_DIRECTOR_FULL.md"
    hard(director_report.is_file(), "accepted Director G1 report missing")
    director_text = director_report.read_text(encoding="utf-8")
    hard("G1: PASS." in director_text and
         "Selected backbone: resnet1d" in director_text and
         "hold T03 pending a separate Director decision" in director_text,
         "Director T02c report does not confirm accepted resnet1d G1")

    t03_path = root / "configs" / "t03.yaml"
    with t03_path.open("r", encoding="utf-8") as handle:
        t03 = yaml.safe_load(handle)
    hard(isinstance(t03, dict), "invalid T03 YAML")
    expected_t03 = {
        "step": "P05-T03", "backbone": MODEL,
        "variants": list(VARIANTS), "seeds": list(SEEDS),
        "alpha_primary": 0.02, "alpha_sensitivity": [0.01, 0.05],
        "g2_min_mean_delta": 0.02, "g2_min_positive_seeds": 4,
        "label_smoothing": 0.10, "approved_gate": "T03_AUTHORIZED",
    }
    hard(t03 == expected_t03, "T03 protocol differs from Director instruction")
    hard(int(cfg["max_epochs"]) == 150 and int(cfg["patience"]) == 20,
         "training schedule differs from T02c")

    source_hashes = {
        "t03_yaml": sha256_file(t03_path),
        "t03_label_variants_py": sha256_file(Path(__file__).resolve()),
        "models_py": EXPECTED_HASHES["models.py"],
        "train_lib_py": EXPECTED_HASHES["train_lib.py"],
        "t02c_pilot_py": EXPECTED_HASHES["t02c_pilot.py"],
        "t02c_director_full_py": EXPECTED_HASHES["t02c_director_full.py"],
        "t02c_yaml": EXPECTED_HASHES["t02c.yaml"],
    }
    print("G1_DIRECTOR=PASS; BACKBONE=resnet1d", flush=True)
    print("T03_PROTOCOL=PASS; VARIANTS=A,B,C,D,E; SEEDS=0..4; ALPHA=0.02", flush=True)
    print("FROZEN_CODE=PASS; TEST=SEALED", flush=True)
    return cfg, t03, source_hashes, frozen


def metadata(proc_dir, split, columns, pd, np):
    """Explicit fit/val/calib reads only; no folder enumeration."""
    hard(split in EXPECTED_EVENTS, "unapproved metadata split")
    path = Path(proc_dir) / f"{split}_meta.parquet"
    frame = pd.read_parquet(path, columns=list(columns))
    hard(list(frame.columns) == list(columns), f"{split} metadata schema")
    hard(len(frame) == EXPECTED_EVENTS[split], f"{split} metadata count")
    hard(frame["idx"].is_unique and not frame["idx"].isna().any(),
         f"{split} idx invalid")
    if "record" in frame:
        hard(not frame["record"].isna().any() and
             frame["record"].nunique() == EXPECTED_RECORDS[split],
             f"{split} record count")
    if "y" in frame:
        y = frame["y"].to_numpy()
        hard(np.isin(y, (0, 1)).all(), f"{split} hard labels not binary")
    if "contested" in frame:
        hard(np.isin(frame["contested"].to_numpy(), (0, 1)).all(),
             f"{split} contested flags invalid")
    if "votes_missing" in frame:
        hard(np.isin(frame["votes_missing"].to_numpy(), (0, 1)).all(),
             f"{split} missing-vote flags invalid")
    for key in ("q_mean", "q_adj"):
        if key in frame:
            q = frame[key].to_numpy(dtype=np.float64)
            hard(np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all(),
                 f"{split} soft-label range invalid")
    return frame


def fit_variant(original, fit_meta, variant, torch, np):
    """Copy the fit dictionary and replace ONLY the fit target tensor.

    D additionally filters the entire copied fit dictionary on contested=0.
    Original fit data and hard-label validation are never modified.
    """
    hard(variant in VARIANTS, "unknown variant")
    hard(np.array_equal(original["idx"], fit_meta["idx"].to_numpy()),
         "fit cache/meta ordering mismatch")
    out = dict(original)
    y = fit_meta["y"].to_numpy(dtype=np.float32)
    if variant == "B":
        y = fit_meta["q_mean"].to_numpy(dtype=np.float32)
    elif variant == "C":
        y = fit_meta["q_adj"].to_numpy(dtype=np.float32)
    elif variant == "E":
        y = y * np.float32(0.90) + np.float32(0.05)
    if variant == "D":
        keep = fit_meta["contested"].to_numpy() == 0
        hard(keep.any() and (~keep).any(), "D consensus subset invalid")
        missing = fit_meta["votes_missing"].to_numpy() == 1
        hard(not np.any(missing & ~keep),
             "D would remove votes_missing=1 observations")
        indices = torch.from_numpy(np.flatnonzero(keep).astype(np.int64))
        out["x"] = original["x"].index_select(0, indices)
        out["mask"] = original["mask"].index_select(0, indices)
        out["idx"] = original["idx"][keep].copy()
        y = y[keep]
    hard(len(y) == len(out["idx"]) and len(y) > 0, "fit target length")
    hard(np.isfinite(y).all() and ((y >= 0) & (y <= 1)).all(),
         "fit target must be finite in [0, 1]")
    out["y"] = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    hard(out["y"].dtype == torch.float32 and
         out["y"].data_ptr() != original["y"].data_ptr(),
         "fit y must be an independent float32 tensor")
    print(f"FIT_VARIANT={variant}; N={len(y)}; "
          f"N_REMOVED={len(original['idx']) - len(y)}; "
          f"LABEL_MIN={float(y.min()):.4f}; LABEL_MAX={float(y.max()):.4f}",
          flush=True)
    return out


def variant_hashes(source_hashes, variant, fit_copy, np):
    """Bind a checkpoint to its exact label tensor and selected fit indices."""
    indices_text = json.dumps([str(i) for i in fit_copy["idx"]],
                              ensure_ascii=True, separators=(",", ":"))
    targets = np.ascontiguousarray(fit_copy["y"].numpy(), dtype=np.float32)
    return {
        **source_hashes,
        "t03_variant": variant,
        "t03_fit_count": int(len(fit_copy["idx"])),
        "t03_fit_index_sha256": hashlib.sha256(
            indices_text.encode("utf-8")).hexdigest(),
        "t03_fit_target_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
    }


def sigmoid(z, np):
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    hard(len(z) > 0 and np.isfinite(z).all(), "nonfinite logits")
    return np.exp(-np.logaddexp(0.0, -z))


def align_scores(path, reference_idx, pd, np, stage):
    frame = pd.read_parquet(path, columns=["idx", "logit"])
    hard(list(frame.columns) == ["idx", "logit"], f"{stage} scores columns")
    hard(len(frame) == len(reference_idx) and frame["idx"].is_unique,
         f"{stage} scores count/idx")
    hard(not frame["idx"].isna().any(), f"{stage} missing score idx")
    # Reindex to the immutable metadata/cache order, regardless of Parquet row order.
    indexed = frame.set_index("idx")["logit"]
    hard(pd.Index(reference_idx).isin(indexed.index).all(),
         f"{stage} idx does not match frozen metadata")
    z = indexed.reindex(pd.Index(reference_idx)).to_numpy(dtype=np.float64)
    hard(np.isfinite(z).all(), f"{stage} logits NaN/Inf")
    return z


def outcome(logits_val, logits_calib, val_meta, calib_meta,
            metrics, conformal, np):
    """Evaluate hard-label val only; calibrate lambda from hard-label calib."""
    yv = val_meta["y"].to_numpy(dtype=np.int8)
    yc = calib_meta["y"].to_numpy(dtype=np.int8)
    groups = calib_meta["record"].to_numpy()
    contested = val_meta["contested"].to_numpy(dtype=np.int8)
    pv = sigmoid(logits_val, np)
    pc = sigmoid(logits_calib, np)
    basic = {
        "auroc": float(metrics.auroc(logits_val, yv)),
        "auprc": float(metrics.auprc(logits_val, yv)),
        "ece": float(metrics.ece(pv, yv, bins=15)),
        "nll": float(np.mean(np.logaddexp(0.0, logits_val) - yv * logits_val)),
    }
    hard(all(math.isfinite(v) for v in basic.values()), "nonfinite val metric")
    alphas = {}
    for alpha in ALPHAS:
        try:
            lam = float(conformal.crc_fnr_threshold(pc, yc, groups, alpha))
        except ValueError as exc:
            # Only alpha=0.01 is allowed to be unavailable due to CRC sample size.
            hard(alpha == 0.01 and
                 "Too few positive records" in str(exc),
                 f"CRC threshold error at alpha={alpha}: {type(exc).__name__}")
            alphas[f"{alpha:.2f}"] = {"status": "INSUFFICIENT_POSITIVE_RECORDS"}
            continue
        suppressed = pv < lam  # Strict suppression is frozen in common.metrics.
        true_suppressed = suppressed & (yv == 1)
        by_contested = {}
        for group in (0, 1):
            pos = (yv == 1) & (contested == group)
            by_contested[str(group)] = {
                "true_count": int(pos.sum()),
                "true_suppressed": int((suppressed & pos).sum()),
            }
        alphas[f"{alpha:.2f}"] = {
            "status": "OK", "lambda": lam,
            "fa_suppression": float(metrics.fa_suppression(pv, yv, lam)),
            "fnr": float(metrics.fnr(pv, yv, lam)),
            "true_suppressed": int(true_suppressed.sum()),
            "true_contested": by_contested,
        }
    hard(alphas["0.02"]["status"] == "OK", "G2 primary alpha infeasible")
    return {"metrics": basic, "alpha": alphas}


def audit_run(run_dir, result, cfg, val_idx, calib_idx,
              pd, np, train_lib):
    hard(train_lib.completed_run(run_dir), f"missing files: {run_dir.name}")
    log = pd.read_csv(run_dir / "log.csv")
    hard(len(log) == result["epochs_run"] and len(log) > 0,
         f"epoch log mismatch: {run_dir.name}")
    hard(np.array_equal(log["epoch"].to_numpy(),
                        np.arange(1, len(log) + 1)),
         f"nonconsecutive epoch log: {run_dir.name}")
    losses = pd.to_numeric(log["train_loss"], errors="coerce").to_numpy()
    aps = pd.to_numeric(log["val_auprc"], errors="coerce").to_numpy()
    hard(np.isfinite(losses).all() and np.isfinite(aps).all(),
         f"NaN/Inf in epoch log: {run_dir.name}")
    hard(1 <= int(result["best_epoch"]) <= len(log) <= cfg["max_epochs"],
         f"best epoch invalid: {run_dir.name}")
    hard(math.isfinite(result["wall_seconds"]) and result["wall_seconds"] > 0,
         f"run wall time invalid: {run_dir.name}")
    val_z = align_scores(run_dir / "scores_val.parquet", val_idx, pd, np,
                         f"{run_dir.name}/val")
    calib_z = align_scores(run_dir / "scores_calib.parquet", calib_idx, pd, np,
                           f"{run_dir.name}/calib")
    return {
        "train": result, "val_logits": val_z, "calib_logits": calib_z,
        "reached_150": len(log) == 150, "loss_nan_inf": False,
        "gpu_run_wall_hours": float(result["wall_seconds"]) / 3600.0,
    }


def manifest_once(cfg, entries, train_lib):
    """Append an artifact exactly once, including interrupted/restarted runs."""
    path = Path(cfg["project_dir"]) / "manifest.csv"
    existing = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                hard("path" in row and "sha256" in row,
                     "existing manifest schema incorrect")
                name = str(Path(row["path"]).resolve()).lower()
                existing.setdefault(name, set()).add(row["sha256"])
    now = datetime.now(timezone.utc).isoformat()
    additions = []
    for artifact, samples in entries:
        artifact = Path(artifact).resolve()
        hard(artifact.is_file(), f"manifest artifact missing: {artifact.name}")
        digest = train_lib.sha256_file(artifact)
        key = str(artifact).lower()
        prior = existing.get(key, set())
        hard(not prior or prior == {digest},
             f"existing manifest file hash changed: {artifact.name}")
        if not prior:
            additions.append((str(artifact), digest, int(samples), now, "P05-T03"))
            existing[key] = {digest}
    if additions:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if path.stat().st_size == 0:
                writer.writerow(["path", "sha256", "n_samples", "timestamp_utc", "step"])
            writer.writerows(additions)
    print(f"MANIFEST_T03_NEW_ROWS={len(additions)}; PASS", flush=True)


def summarize(values, np):
    a = np.asarray(values, dtype=np.float64)
    hard(len(a) == 5 and np.isfinite(a).all(), "5-seed metric incomplete")
    return float(a.mean()), float(a.std(ddof=1))


def fs(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_metric(items, key, np, digits=4):
    mean, sd = summarize([x[key] for x in items], np)
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def assemble(root, cfg, t03, runs, val_meta, calib_meta, metrics,
             conformal, np, pd):
    """Recompute every metric from saved run logits, never from test."""
    evaluations = {}
    for variant in VARIANTS:
        evaluations[variant] = {}
        for seed in SEEDS:
            one = runs[(variant, seed)]
            ev = outcome(one["val_logits"], one["calib_logits"],
                         val_meta, calib_meta, metrics, conformal, np)
            for key in ("auroc", "auprc", "ece"):
                saved = float(one["train"]["metrics"][key])
                hard(abs(saved - ev["metrics"][key]) < 1e-7,
                     f"saved/recomputed metric mismatch: {variant}_s{seed} {key}")
            evaluations[variant][seed] = ev
        mean_val_logits = np.mean(
            np.stack([runs[(variant, seed)]["val_logits"] for seed in SEEDS]), axis=0)
        mean_calib_logits = np.mean(
            np.stack([runs[(variant, seed)]["calib_logits"] for seed in SEEDS]), axis=0)
        evaluations[variant]["ensemble"] = outcome(
            mean_val_logits, mean_calib_logits, val_meta, calib_meta,
            metrics, conformal, np)

    baseline = []
    for seed in (0, 1, 2):
        previous = root / "outputs" / "t02c" / f"resnet1d_s{seed}"
        hard(all((previous / name).is_file() for name in SCORE_FILES),
             f"T02c reference run missing for A seed {seed}")
        old_logits = align_scores(previous / "scores_val.parquet",
                                  val_meta["idx"].to_numpy(), pd, np,
                                  f"T02c reference seed {seed}")
        new_logits = runs[("A", seed)]["val_logits"]
        maximum_delta = float(np.max(np.abs(old_logits - new_logits)))
        yv = val_meta["y"].to_numpy(dtype=np.int8)
        delta_ap = float(metrics.auprc(new_logits, yv) -
                         metrics.auprc(old_logits, yv))
        baseline.append({"seed": seed, "max_abs_delta_logit": maximum_delta,
                         "delta_auprc": delta_ap,
                         "over_1e-4": maximum_delta > 1e-4})

    deltas = [
        evaluations["B"][seed]["alpha"]["0.02"]["fa_suppression"] -
        evaluations["A"][seed]["alpha"]["0.02"]["fa_suppression"]
        for seed in SEEDS
    ]
    delta_mean = float(np.mean(deltas))
    positive_seeds = sum(delta > 0.0 for delta in deltas)
    passed = delta_mean >= 0.02 and positive_seeds >= 4
    return evaluations, baseline, deltas, delta_mean, positive_seeds, passed


def report_lines(cfg, sha, runs, evaluations, baseline, deltas,
                 delta_mean, positive_seeds, passed, np, frozen):
    lines = [
        "# REPORT P05-T03: Label Variants and G2", "",
        f"Status: {'G2 PASS' if passed else 'G2 FAIL; STOP FOR DIRECTOR'}.",
        "T03_REPORT_COMPLETE=TRUE; 25/25 trained or verified runs.",
        "Director authorization: T03_AUTHORIZED (2026-11-11); G1 accepted.",
        "Data: fit for training, val hard labels for early stopping and metrics,",
        "calib hard labels and record groups solely for CRC threshold estimation.",
        "Test files and test labels: SEALED; NOT READ.",
        "Hardware: physical GPU1 Quadro M6000; FP32; AMP off; CPU <=8; BelowNormal.",
        "Backbone: resnet1d; seeds 0..4; A,B,C,D,E; max_epochs=150; patience=20.",
        "The common, models, train_lib, locked design and T02c artifacts are unchanged.",
        "Soft targets alter only an independent copy of fit_data['y']; D filters contested=0.",
        "Each checkpoint binds its variant, fit count, selected-index SHA256 and target SHA256.",
        "Val early stopping uses hard-label AUPRC for every variant.",
        "ROC/PR/NLL/ECE are evaluated against hard validation labels.",
        "CRC thresholds are calibrated independently on hard calib labels/records;",
        "strict suppression is score < lambda and score is sigmoid(logit).",
        "Calibration is not used for model selection. CRC controls calibration",
        "record-level false-negative risk; validation FNR is descriptive, not a guarantee.",
        "Ensembles average five logits before sigmoid and calibrate their own lambda.",
        "All mean +/- SD entries use sample SD (ddof=1) over five seeds.",
        f"Locked design SHA256: {frozen.EXPECTED_LOCK_SHA}",
        f"Frozen common tree SHA256: {frozen.EXPECTED_COMMON_SHA}",
        "Code SHA256: `" + json.dumps(sha, sort_keys=True) + "`", "",
        "## Five-variant summary at primary alpha=0.02", "",
        "Each cell is five-run mean +/- sample SD / ensemble value.",
        "| Variant | AUROC | AUPRC | ECE | NLL | FA suppression | FNR | True suppressed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        five = [evaluations[variant][s] for s in SEEDS]
        ens = evaluations[variant]["ensemble"]
        cells = []
        for key in ("auroc", "auprc", "ece", "nll"):
            cells.append(fmt_metric([v["metrics"] for v in five], key, np) +
                         f" / {ens['metrics'][key]:.4f}")
        for key in ("fa_suppression", "fnr", "true_suppressed"):
            values = [v["alpha"]["0.02"][key] for v in five]
            mean, sd = summarize(values, np)
            cells.append(f"{mean:.4f} +/- {sd:.4f} / " +
                         f"{ens['alpha']['0.02'][key]:.4f}")
        lines.append("| " + variant + " | " + " | ".join(cells) + " |")

    lines += ["", "## Per-run primary alpha=0.02 and contested false negatives", "",
              "`True suppressed 0/1` is the number of y=1 alarms suppressed for",
              "contested=0 or contested=1, respectively. No identifiers are reported.",
              "| Run | AUROC | AUPRC | ECE | NLL | lambda | FA suppression | FNR | True suppressed | True suppressed 0/1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        for seed in SEEDS:
            ev = evaluations[variant][seed]
            m = ev["metrics"]
            a = ev["alpha"]["0.02"]
            group = a["true_contested"]
            lines.append(
                f"| {variant}_s{seed} | {m['auroc']:.6f} | {m['auprc']:.6f} | "
                f"{m['ece']:.6f} | {m['nll']:.6f} | {a['lambda']:.8f} | "
                f"{a['fa_suppression']:.6f} | {a['fnr']:.6f} | "
                f"{a['true_suppressed']} | {group['0']['true_suppressed']}/"
                f"{group['1']['true_suppressed']} |")

    lines += ["", "## Ensemble thresholds and outcomes", "",
              "| Variant | alpha | lambda | FA suppression | FNR | True suppressed | True suppressed contested=0/1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        ens = evaluations[variant]["ensemble"]
        for alpha in ALPHAS:
            key = f"{alpha:.2f}"
            a = ens["alpha"][key]
            if a["status"] != "OK":
                lines.append(f"| {variant} | {key} | N/A (insufficient positive records) | N/A | N/A | N/A | N/A |")
            else:
                g = a["true_contested"]
                lines.append(
                    f"| {variant} | {key} | {a['lambda']:.8f} | "
                    f"{a['fa_suppression']:.6f} | {a['fnr']:.6f} | "
                    f"{a['true_suppressed']} | "
                    f"{g['0']['true_suppressed']}/{g['1']['true_suppressed']} |")

    lines += ["", "## Sensitivity for alpha=0.01 and alpha=0.05", "",
              "Exploratory reporting only; G2 remains fixed at alpha=0.02.",
              "| Variant | alpha | Mean +/- SD FA suppression | Mean +/- SD FNR |",
              "| --- | ---: | ---: | ---: |"]
    for variant in VARIANTS:
        for alpha in (0.01, 0.05):
            key = f"{alpha:.2f}"
            five = [evaluations[variant][s]["alpha"][key] for s in SEEDS]
            if any(a["status"] != "OK" for a in five):
                lines.append(f"| {variant} | {key} | N/A (CRC infeasible) | N/A |")
            else:
                lines.append(
                    f"| {variant} | {key} | {fmt_metric(five, 'fa_suppression', np, 6)} | "
                    f"{fmt_metric(five, 'fnr', np, 6)} |")

    lines += ["", "## G2: fixed primary alpha=0.02, B minus A (paired seeds)", "",
              "G2 requires mean delta >=0.02 AND delta>0 for at least four of five seeds.",
              "| Seed | Suppression A | Suppression B | B - A | B > A? |",
              "| --- | ---: | ---: | ---: | :---: |"]
    for seed, delta in zip(SEEDS, deltas):
        a = evaluations["A"][seed]["alpha"]["0.02"]["fa_suppression"]
        b = evaluations["B"][seed]["alpha"]["0.02"]["fa_suppression"]
        lines.append(f"| {seed} | {a:.6f} | {b:.6f} | {delta:+.6f} | "
                     f"{'YES' if delta > 0 else 'NO'} |")
    lines += ["", f"G2 mean B-A: {delta_mean:+.6f}; required >=+0.020000.",
              f"G2 positive seeds: {positive_seeds}/5; required >=4/5.",
              f"**G2: {'PASS' if passed else 'FAIL'}**.",
              "G2 FAIL is a scientific outcome; no threshold adjustment or rescue variant.",
              "", "## A seed 0-2 reproducibility against frozen T02c resnet1d", "",
              "Threshold 1e-4 is report-only; exceedance does not halt T03.",
              "| Seed | Max absolute val logit difference | New - old val AUPRC | Exceeds 1e-4? |",
              "| --- | ---: | ---: | :---: |"]
    for entry in baseline:
        lines.append(
            f"| {entry['seed']} | {entry['max_abs_delta_logit']:.9g} | "
            f"{entry['delta_auprc']:+.9g} | "
            f"{'YES - Director review' if entry['over_1e-4'] else 'NO'} |")

    lines += ["", "## Training execution audit (physical GPU1)", "",
              "GPU run wall hours measure training elapsed time with GPU1, not",
              "kernel-level GPU utilization. Checkpoint timing excludes parquet writes.",
              "| Run | Epochs | Best epoch | Reached 150? | Loss NaN/Inf? | GPU run wall hours | Peak VRAM GiB |",
              "| --- | ---: | ---: | :---: | :---: | ---: | ---: |"]
    for variant in VARIANTS:
        for seed in SEEDS:
            r = runs[(variant, seed)]
            train = r["train"]
            lines.append(
                f"| {variant}_s{seed} | {train['epochs_run']} | {train['best_epoch']} | "
                f"{'YES' if r['reached_150'] else 'NO'} | "
                f"{'YES' if r['loss_nan_inf'] else 'NO'} | "
                f"{r['gpu_run_wall_hours']:.5f} | {train['peak_vram_gib']:.3f} |")
    total_gpu_h = sum(r["gpu_run_wall_hours"] for r in runs.values())
    lines += ["", f"Total 25-run GPU training wall hours: {total_gpu_h:.5f}.",
              "All 25 required checkpoint, val logits, calib logits and log files verified.",
              "No T04/T05 or test evaluation has been started by T03.",
              "", "## Operational disposition", "",
              "T03 is complete. STOP at G2 and send this report to Director.",
              "Do not choose a variant or run another phase before Director review.",
              "Frozen T02c and all pre-existing outputs remain unchanged."]
    return "\n".join(lines) + "\n"


def run_full(root, cfg, t03, sha, frozen):
    # Only now, after frozen source hashes have passed, may torch be imported.
    import numpy as np
    import pandas as pd
    import torch
    import models
    import train_lib

    frozen.below_normal_priority()
    print(f"PYTHON={sys.version.split()[0]}; TORCH={torch.__version__}; "
          "FP32=TRUE; AMP=FALSE; CPU_THREADS<=8", flush=True)
    metrics = frozen.load_common_metrics(cfg)
    sys.path.insert(0, str(Path(cfg["common_dir"]).resolve().parent))
    from common import conformal

    device = frozen.gpu_smoke_test(torch, models, train_lib, cfg)
    fit_meta = metadata(cfg["proc_dir"], "fit",
                        ("idx", "y", "q_mean", "q_adj", "contested", "votes_missing"),
                        pd, np)
    val_meta = metadata(cfg["proc_dir"], "val", ("idx", "y", "contested"), pd, np)
    calib_meta = metadata(cfg["proc_dir"], "calib", ("idx", "record", "y"), pd, np)
    fit = train_lib.load_split(cfg["proc_dir"], "fit")
    val = train_lib.load_split(cfg["proc_dir"], "val")
    hard(np.array_equal(fit["idx"], fit_meta["idx"].to_numpy()),
         "fit array alignment")
    hard(np.array_equal(val["idx"], val_meta["idx"].to_numpy()),
         "val array alignment")
    hard(np.array_equal(val["y"].numpy(), val_meta["y"].to_numpy()),
         "val hard-label alignment")
    output = root / "outputs" / "t03"
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    started = time.perf_counter()
    for variant in VARIANTS:
        fit_copy = fit_variant(fit, fit_meta, variant, torch, np)
        run_sha = variant_hashes(sha, variant, fit_copy, np)
        for seed in SEEDS:
            run_dir = output / f"{variant}_s{seed}"
            if train_lib.completed_run(run_dir):
                result = train_lib.load_completed_result(
                    run_dir, MODEL, seed, run_sha, frozen.EXPECTED_LOCK_SHA)
                print(f"T03_SKIP_COMPLETE={variant}_s{seed}; source_sha=PASS", flush=True)
            else:
                print(f"T03_RUN_BEGIN={variant}_s{seed}", flush=True)
                result = train_lib.train_model(
                    MODEL, seed, fit_copy, val, cfg, device, metrics,
                    output_dir=run_dir, sha=run_sha)
                print(f"T03_RUN_END={variant}_s{seed}", flush=True)
            item = audit_run(run_dir, result, cfg,
                             val_meta["idx"].to_numpy(),
                             calib_meta["idx"].to_numpy(), pd, np, train_lib)
            manifest_once(cfg, [
                (run_dir / "ckpt.pt", 0),
                (run_dir / "scores_val.parquet", EXPECTED_EVENTS["val"]),
                (run_dir / "scores_calib.parquet", EXPECTED_EVENTS["calib"]),
                (run_dir / "log.csv", result["epochs_run"]),
            ], train_lib)
            runs[(variant, seed)] = item
            print(f"T03_RUN_VERIFIED={variant}_s{seed}; "
                  f"epochs={result['epochs_run']}; best={result['best_epoch']}; "
                  f"GPU_wall_h={item['gpu_run_wall_hours']:.5f}", flush=True)
        del fit_copy
        print(f"T03_VARIANT_COMPLETE={variant}; SEEDS=5/5", flush=True)

    hard(len(runs) == 25, "25-run requirement failed")
    evaluations, baseline, deltas, mean_delta, positive, passed = assemble(
        root, cfg, t03, runs, val_meta, calib_meta, metrics, conformal, np, pd)
    report_path = root / "reports" / "P05-T03.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = report_lines(cfg, sha, runs, evaluations, baseline, deltas,
                          mean_delta, positive, passed, np, frozen)
    if report_path.exists():
        hard(report_path.read_text(encoding="utf-8") == report,
             "existing T03 report differs; preserving immutable report")
        print(f"T03_REPORT_ALREADY_VERIFIED={report_path}", flush=True)
    else:
        with report_path.open("x", encoding="utf-8") as stream:
            stream.write(report)
        print(f"T03_REPORT_SAVED={report_path}", flush=True)
    manifest_once(cfg, [(report_path, 0)], train_lib)
    print("=== REPORT P05-T03 G2 ===", flush=True)
    print("T03_ALL_25_RUNS=PASS; T03_TEST=SEALED", flush=True)
    for variant in VARIANTS:
        five = [evaluations[variant][s]["alpha"]["0.02"]["fa_suppression"]
                for s in SEEDS]
        avg, sd = summarize(five, np)
        ens = evaluations[variant]["ensemble"]
        print(f"VARIANT={variant}; suppression_val_mean={avg:.6f}; "
              f"sd={sd:.6f}; ensemble={ens['alpha']['0.02']['fa_suppression']:.6f}; "
              f"ensemble_lambda={ens['alpha']['0.02']['lambda']:.8f}", flush=True)
    for seed, delta in zip(SEEDS, deltas):
        print(f"G2_DELTA_SEED_{seed}={delta:+.6f}", flush=True)
    for entry in baseline:
        print(f"A_REPRO_SEED_{entry['seed']}: MAX_ABS_LOGIT_DIFF="
              f"{entry['max_abs_delta_logit']:.9g}; DELTA_AUPRC="
              f"{entry['delta_auprc']:+.9g}; DIRECTOR_REVIEW="
              f"{'YES' if entry['over_1e-4'] else 'NO'}", flush=True)
    print(f"G2_MEAN_DELTA={mean_delta:+.6f}; REQUIRED>=+0.020000", flush=True)
    print(f"G2_POSITIVE_SEEDS={positive}/5; REQUIRED>=4/5", flush=True)
    print(f"G2={'PASS' if passed else 'FAIL'}", flush=True)
    print(f"TOTAL_GPU_TRAINING_WALL_H={sum(x['gpu_run_wall_hours'] for x in runs.values()):.5f}", flush=True)
    print(f"T03_COORDINATOR_WALL_H={(time.perf_counter()-started)/3600:.5f}", flush=True)
    print(f"REPORT_FILE={report_path}", flush=True)
    print("T03_STOP=WAIT_FOR_DIRECTOR; TEST=SEALED; NO_TEST_FILES_READ", flush=True)
    print("=== END REPORT ===", flush=True)
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("inspect", "full"))
    parser.add_argument("--director-approval", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg, t03, sha, frozen = preflight(root)
    if args.stage == "inspect":
        print("T03_INSPECT=PASS; TRAINING=NOT_STARTED; TEST=SEALED", flush=True)
        return 0
    hard(args.director_approval == "T03_AUTHORIZED", "Director approval token missing")
    return run_full(root, cfg, t03, sha, frozen)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("T03_INTERRUPTED; previous complete runs remain reusable", flush=True)
        print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
        sys.exit(130)
    except Exception as exc:
        print("=== REPORT P05-T03 COORDINATOR ===", flush=True)
        print("T03_STATUS=FAIL_TECHNICAL; preserved frozen sources and run artifacts",
              flush=True)
        text = str(exc)
        if text.startswith("T03_HARD_STOP:"):
            print(text, flush=True)
        else:
            print(f"ERROR_TYPE={type(exc).__name__}; inspect local execution log",
                  flush=True)
        print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
        print("=== END REPORT ===", flush=True)
        sys.exit(1)
