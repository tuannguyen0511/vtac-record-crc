"""P05-T05b Director-authorized one-shot test and Challenge 2015 audit.

Two stages: `inspect` (never opens labels or writes a lock) and `execute`.
Once TEST_OPENED.flag exists, neither stage can be used again. Never delete it.
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "8"

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


DIRECTOR_LOCK_SHA = "c09b6cd952319950f65b57e64086ab691946ccabac84adb106efb2bc765299c8"
VARIANTS = "ABCDE"
SEEDS = range(5)
ALPHAS = (0.01, 0.02, 0.05)


def hard(condition, reason):
    if not condition:
        raise RuntimeError("T05B_HARD_STOP: " + str(reason))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for part in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def sigmoid(z, np):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))


def redact(n):
    return "<10" if n < 10 else str(int(n))


def safe_rate(k, n):
    return float(k / n) if n else None


def check_preopened(root):
    flag = root / "reports" / "TEST_OPENED.flag"
    hard(not flag.exists(), "TEST_OPENED.flag already exists: DO NOT RERUN OR DELETE")
    return flag


def validate_director_and_sources(root, yaml, core, lock_path):
    actual = digest(lock_path)
    print(f"FINAL_LOCK_SHA256={actual}; expected={DIRECTOR_LOCK_SHA}; "
          f"status={'PASS' if actual == DIRECTOR_LOCK_SHA else 'FAIL'}", flush=True)
    hard(actual == DIRECTOR_LOCK_SHA, "Director final_lock SHA mismatch")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    hard(lock["step"] == "P05-T05b" and lock["primary"]["variant"] == "A" and
         float(lock["primary"]["alpha"]) == 0.02 and
         lock["model"]["backbone"] == "resnet1d" and
         list(lock["secondary"]["variants"]) == list("BCDE"), "Director lock parameters changed")
    frozen = lock["frozen_sha256"]
    cfg = core.validate_frozen(root / "configs" / "t02c.yaml")
    hard(Path(cfg["project_dir"]).resolve() == root.resolve(), "T02c project root differs")
    expected_paths = {
        "locked_design_yaml": Path(cfg["locked_design"]),
        "models_py": root / "src" / "models.py",
        "train_lib_py": root / "src" / "train_lib.py",
        "t03_label_variants_py": root / "src" / "t03_label_variants.py",
        "t05a_conformal_ablation_py": root / "src" / "t05a_conformal_ablation.py",
        "test_labels_parquet": Path(cfg["proc_dir"]) / "test_labels.parquet",
    }
    sha = {"final_lock.yaml": actual}
    for key, path in expected_paths.items():
        hard(path.is_file(), f"missing frozen {key}")
        actual = digest(path)  # For test_labels: bytes only, NEVER parse before flag.
        hard(actual == frozen[key], f"frozen SHA mismatch {key}")
        sha[key] = actual
        print(f"FROZEN_SHA_{key}=PASS; sha256={actual}", flush=True)
    common = core.common_source_sha256(cfg["common_dir"])
    hard(common == frozen["common_tree"], "frozen common tree mismatch")
    sha["common_tree"] = common
    with Path(cfg["locked_design"]).open("r", encoding="utf-8") as fp:
        prereg = yaml.safe_load(fp)
    prov = prereg.get("provenance_sha256", {})
    if "preprocess_lib_script" in prov:
        source = root / "src" / "preprocess_lib.py"
        hard(digest(source) == prov["preprocess_lib_script"], "preprocessing SHA mismatch")
        sha["preprocess_lib_py"] = digest(source)
    for split in ("fit", "calib", "val", "test"):
        expected_meta = prov.get("meta_files", {}).get(f"{split}_meta.parquet")
        if expected_meta:
            hard(digest(Path(cfg["proc_dir"]) / f"{split}_meta.parquet") == expected_meta,
                 f"{split} metadata SHA mismatch")
    hard(len(lock["checkpoints_sha256"]) == 25, "Director checkpoint count must be 25")
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"{variant}_s{seed}"
            checkpoint = root / "outputs" / "t03" / name / "ckpt.pt"
            hard(checkpoint.is_file() and digest(checkpoint) == lock["checkpoints_sha256"][name],
                 f"checkpoint SHA mismatch {name}")
            sha[f"checkpoint_{name}"] = lock["checkpoints_sha256"][name]
            print(f"CHECKPOINT_SHA_PASS={name}; sha256={sha[f'checkpoint_{name}']}", flush=True)
    print("FROZEN_ALL_AND_25_CHECKPOINTS=PASS; TEST_LABELS_NOT_READ=TRUE", flush=True)
    return lock, cfg, sha


def precompute_calibration(root, cfg, lock, np, pd, conformal, t03):
    meta = pd.read_parquet(Path(cfg["proc_dir"]) / "calib_meta.parquet",
                           columns=["idx", "record", "y"])
    hard(meta["idx"].is_unique and not meta.isna().any().any(), "calib metadata invalid")
    y = meta["y"].to_numpy(dtype=np.int8)
    hard(np.isin(y, [0, 1]).all() and len(y) > 0, "calibration binary labels invalid")
    idx = meta["idx"].to_numpy()
    group = meta["record"].to_numpy()
    per_seed = {}
    per_variant = {}
    lambda_rows = []
    for variant in VARIANTS:
        stack = []
        for seed in SEEDS:
            path = root / "outputs" / "t03" / f"{variant}_s{seed}" / "scores_calib.parquet"
            hard(path.is_file(), f"missing calibrated scores {variant}_s{seed}")
            z = t03.align_scores(path, idx, pd, np, f"{variant}_s{seed}/calib")
            stack.append(z)
            if variant == "A":
                per_seed[seed] = {a: float(conformal.crc_fnr_threshold(sigmoid(z, np), y, group, a))
                                  for a in ALPHAS}
        ensemble = sigmoid(np.mean(np.stack(stack), axis=0), np)
        per_variant[variant] = ensemble
        for alpha in ALPHAS:
            found = float(conformal.crc_fnr_threshold(ensemble, y, group, alpha))
            locked = float(lock["lambda"][variant][alpha])
            delta = abs(found - locked)
            print(f"LAMBDA_AUDIT={variant}; alpha={alpha}; recalculated={found:.12g}; "
                  f"locked={locked:.12g}; abs_delta={delta:.3g}; "
                  f"status={'PASS' if delta <= 1e-9 else 'FAIL'}", flush=True)
            hard(delta <= 1e-9, f"lambda mismatch variant {variant} alpha {alpha}")
            lambda_rows.append((variant, alpha, locked, found, delta))
    grid = np.unique(np.concatenate((np.linspace(.0005, .01, 20), np.linspace(.01, .40, 40))))
    hard(grid.size == 59, "LTT candidate grid altered")
    ltt = conformal.ltt_cp_threshold(per_variant["A"], y, alpha=.05, delta=.05, grid=grid)
    hard(ltt is not None and abs(ltt - float(lock["ltt_comparator"]["tau"])) <= 1e-9,
         "LTT tau not equal Director lock")
    print("LAMBDA_RECALCULATED_15_OF_15=PASS; LTT_TAU=PASS; TEST_SEALED", flush=True)
    return lambda_rows, per_seed, meta


def read_test_inputs(cfg, np, pd, torch):
    proc = Path(cfg["proc_dir"])
    meta = pd.read_parquet(proc / "test_meta.parquet", columns=["idx", "record", "event"])
    hard(list(meta.columns) == ["idx", "record", "event"] and meta["idx"].is_unique,
         "unlabeled test metadata invalid")
    x = np.load(proc / "test_W10_D0_x.npy", allow_pickle=False)
    mask = np.load(proc / "test_W10_D0_mask.npy", allow_pickle=False)
    hard(len(x) == len(meta) and x.shape == (len(meta), 4, 2500) and
         mask.shape == (len(meta), 4) and x.dtype == np.float32 and
         mask.dtype == np.bool_ and np.isfinite(x).all() and
         np.max(np.abs(x)) <= 10.000001, "test unlabeled cache invalid")
    print(f"UNLABELED_TEST_INPUTS=PASS; n={len(meta)}; LABELS_NOT_READ=TRUE", flush=True)
    return {"x": torch.from_numpy(np.array(x, copy=True, order="C")),
            "mask": torch.from_numpy(np.array(mask, copy=True, order="C")),
            "idx": meta["idx"].to_numpy().copy(), "y": None}, meta


def model_probabilities(root, lock, data, external_data, np, torch, models, train_lib, core):
    device = train_lib.cuda_device()
    hard(torch.cuda.device_count() == 1 and torch.cuda.get_device_name(0) == "Quadro M6000",
         "GPU1 Quadro M6000 not selected")
    torch.set_num_threads(8)
    test_logits = {}
    external_logits = []
    for variant in VARIANTS:
        seeds_logits = []
        for seed in SEEDS:
            run = root / "outputs" / "t03" / f"{variant}_s{seed}"
            cp = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
            hard(cp.get("model") == "resnet1d" and int(cp.get("seed", -1)) == seed,
                 f"checkpoint identity mismatch {variant}_s{seed}")
            model, count = models.make_model("resnet1d")
            hard(count == int(cp["parameters"]), "checkpoint parameter count mismatch")
            model.load_state_dict(cp["state_dict"], strict=True)
            model.to(device)
            z = train_lib.predict_logits(model, data, "resnet1d", device, 128)
            hard(len(z) == len(data["idx"]) and np.isfinite(z).all(), "test logits invalid")
            seeds_logits.append(z)
            if variant == "A":
                ze = train_lib.predict_logits(model, external_data, "resnet1d", device, 128)
                hard(np.isfinite(ze).all(), "external logits invalid")
                external_logits.append(ze)
            print(f"UNLABELED_INFERENCE=PASS; variant={variant}; seed={seed}", flush=True)
            del cp, model
            torch.cuda.empty_cache()
        test_logits[variant] = np.stack(seeds_logits)
    torch.cuda.synchronize(device)
    core.verify_nvidia_process_is_gpu1()
    return test_logits, np.mean(np.stack(external_logits), axis=0)


def split_counts(prob, y, threshold, inclusive, metrics, np):
    suppressed = prob <= threshold if inclusive else prob < threshold
    negatives, positives = y == 0, y == 1
    n_neg, n_pos = int(negatives.sum()), int(positives.sum())
    n_sn = int((suppressed & negatives).sum())
    n_sp = int((suppressed & positives).sum())
    ci = metrics.clopper_pearson(n_sp, n_pos) if n_pos else (None, None)
    def complementary(n, k):
        # If either side of n = suppressed + retained is <10, both must be
        # withheld, or the small value can be reconstructed by subtraction.
        if n < 10 or k < 10 or n-k < 10:
            return "WITHHELD", "WITHHELD"
        return str(k), str(n-k)
    sn_text, rn_text = complementary(n_neg, n_sn)
    sp_text, rp_text = complementary(n_pos, n_sp)
    return {"suppression": safe_rate(n_sn, n_neg), "fnr": safe_rate(n_sp, n_pos),
            "fnr_ci": ci, "n_false": redact(n_neg), "n_true": redact(n_pos),
            "false_suppressed": sn_text, "true_suppressed": sp_text,
            "false_retained": rn_text, "true_retained": rp_text}


def score_test_variant(logits, y, lock, variant, metrics, np):
    p = sigmoid(logits.mean(axis=0), np)
    out = {"variant": variant, "auroc": metrics.auroc(p, y),
           "auprc": metrics.auprc(p, y), "ece15": metrics.ece(p, y, bins=15),
           "challenge_at_0_5": challenge(p, y, .5, metrics, np),
           "alphas": {}}
    for alpha in ALPHAS:
        lam = float(lock["lambda"][variant][alpha])
        out["alphas"][str(alpha)] = {"lambda": lam, **split_counts(p, y, lam, False, metrics, np),
                                       "challenge_at_lambda": challenge(p, y, lam, metrics, np)}
    return out, p


def challenge(p, y, threshold, metrics, np):
    predicted = p >= threshold
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    return float(metrics.challenge_score(tp, tn, fp, fn))


def bootstrap_record(y, records, probs, lock, np):
    hard(len(records) == len(y) and len(set(records)) > 1, "test record groups missing")
    code, uniques = np.unique(np.asarray(records, dtype=str), return_inverse=True)
    members = [np.flatnonzero(uniques == i) for i in range(len(code))]
    rng = np.random.default_rng(42)
    saved = {v: {"suppression": [], "fnr": []} for v in VARIANTS}
    paired = {v: [] for v in "BCDE"}
    valid = 0
    for _ in range(2000):
        sampled = rng.integers(len(code), size=len(code))
        row_ids = np.concatenate([members[i] for i in sampled])
        suby = y[row_ids]
        nneg, npos = np.count_nonzero(suby == 0), np.count_nonzero(suby == 1)
        if not nneg or not npos:
            continue
        valid += 1
        v_supp = {}
        for v in VARIANTS:
            cut = probs[v][row_ids] < float(lock["lambda"][v][0.02])
            suppression = float(cut[suby == 0].mean())
            fnr = float(cut[suby == 1].mean())
            saved[v]["suppression"].append(suppression)
            saved[v]["fnr"].append(fnr)
            v_supp[v] = suppression
        for v in "BCDE":
            paired[v].append(v_supp[v] - v_supp["A"])
    hard(valid >= 1900, "too many invalid bootstrap samples")
    ci = lambda items: [float(x) for x in np.percentile(items, (2.5, 97.5))]
    return {"replicates_requested": 2000, "replicates_valid": valid, "seed": 42,
            "variant_ci": {v: {key: ci(saved[v][key]) for key in ("suppression", "fnr")}
                           for v in VARIANTS},
            "paired_deltas": {v + "_minus_A": {"point": float(
                                (probs[v][y == 0] < float(lock["lambda"][v][.02])).mean() -
                                (probs["A"][y == 0] < float(lock["lambda"]["A"][.02])).mean()),
                                "ci95": ci(paired[v])} for v in "BCDE"}}


def h2_by_contested(test, p, lock, metrics, np):
    y = test["y"].to_numpy(dtype=np.int8)
    contested = test["contested"].to_numpy(dtype=np.int8)
    valid = test["votes_missing"].to_numpy(dtype=np.int8) == 0
    suppressed = p < float(lock["lambda"]["A"][.02])
    rows = []
    counts = []
    for c in (0, 1):
        group = (y == 1) & valid & (contested == c)
        n = int(group.sum())
        k = int((suppressed & group).sum())
        counts.append((n, k))
        rows.append({"contested": c, "n_true": redact(n),
                     "true_suppressed": str(k) if min(k, n-k) >= 10 else "WITHHELD",
                     "true_retained": str(n-k) if min(k, n-k) >= 10 else "WITHHELD",
                     "fnr": safe_rate(k, n) if n >= 10 else None,
                     "fnr_ci": metrics.clopper_pearson(k, n) if n >= 10 else None})
    # Complementary-disclosure protection. If either subgroup is small, mask
    # all exact subgroup counts and rates; the excluded/missing group is not printed.
    if any(n < 10 for n, _ in counts):
        for row in rows:
            row.update(n_true="WITHHELD", true_suppressed="WITHHELD",
                       true_retained="WITHHELD", fnr=None, fnr_ci=None)
    return {"votes_missing_excluded": True, "rows": rows}


def seed_a_stability(logits, y, per_seed, np):
    out = {}
    for alpha in ALPHAS:
        s, f = [], []
        for seed in SEEDS:
            p = sigmoid(logits[seed], np)
            suppressed = p < per_seed[seed][alpha]
            s.append(float(suppressed[y == 0].mean()))
            f.append(float(suppressed[y == 1].mean()))
        out[str(alpha)] = {"suppression_min": min(s), "suppression_max": max(s),
                           "fnr_min": min(f), "fnr_max": max(f)}
    return out


def make_prereg(root, sha, rows, external_audit, np):
    lines = ["# P05-T05b one-shot preregistration", "",
             "Written before opening test labels. No test metrics are included.", "",
             "## Frozen SHA-256 audit", ""]
    lines.extend(f"- `{name}`: `{value}`" for name, value in sha.items())
    lines += ["", "## Calibration rederivation (pre-test)", "",
              "| Variant | alpha | Locked lambda | Recalculated lambda | Absolute error |",
              "| --- | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {v} | {a:.2f} | {lk:.12g} | {rc:.12g} | {d:.3g} |"
                 for v, a, lk, rc, d in rows)
    lines += ["", "LTT: 59 candidates, alpha=0.05, delta=0.05, tau=0.09; PASS.",
              "Primary: A, resnet1d, alpha=0.02; B/C/D/E exploratory.",
              "No hyperparameter, threshold, or architecture change permitted.",
              "", "## External public dataset preflight", "",
              f"SHA256SUMS: PASS; public VT records: {len(external_audit['y'])}.",
              "", "TEST_OPENED.flag must be created after this file is durably written."]
    path = root / "reports" / "PREREG_LOCK.md"
    hard(not path.exists(), "PREREG_LOCK.md already exists; STOP for Director review")
    with path.open("x", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(lines) + "\n")
        file.flush()
        os.fsync(file.fileno())
    return path


def one_shot_flag(flag, lock_sha):
    fd = os.open(flag, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write("P05-T05b test opened permanently\n")
        fp.write(f"UTC={datetime.now(timezone.utc).isoformat()}\n")
        fp.write(f"FINAL_LOCK_SHA256={lock_sha}\n")
        fp.flush()
        os.fsync(fp.fileno())
    print("TEST_OPENED=TRUE; DO_NOT_RERUN=TRUE", flush=True)


def read_sealed_labels_after_flag(cfg, flag, test_meta, pd, np):
    hard(flag.exists(), "TEST_OPENED.flag missing: test labels inaccessible")
    labels = pd.read_parquet(Path(cfg["proc_dir"]) / "test_labels.parquet")
    expected = ["idx", "y", "q_mean", "q_adj", "contested", "n_votes", "votes_missing"]
    hard(list(labels.columns) == expected and len(labels) == len(test_meta) and
         labels["idx"].is_unique and
         np.array_equal(labels["idx"].to_numpy(), test_meta["idx"].to_numpy()) and
         all(np.isin(labels[k].to_numpy(), [0, 1]).all()
             for k in ("y", "contested", "votes_missing")), "test labels schema or alignment mismatch")
    return labels


def write_results(root, summary, prereg, np):
    output = root / "outputs" / "t05b"
    hard(not output.exists(), "t05b output directory already exists")
    output.mkdir(parents=True, exist_ok=False)
    report = root / "reports" / "P05-T05b.md"
    hard(not report.exists(), "T05b report exists; refuse overwrite")
    artifact = output / "summary.json"
    with artifact.open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    lines = ["# P05-T05b final test and external Challenge 2015", "",
             "**ONE SHOT. TEST_OPENED.flag exists permanently.**", "",
             f"Final Director lock SHA256: `{summary['final_lock_sha256']}`", "",
             "All frozen SHA hashes, 25 checkpoints, 15 calibration lambdas and LTT passed before test opening.",
             "Primary A, alpha=0.02; B/C/D/E secondary. No recalibration or retraining.",
             "", "## Official sealed test: all variants", "",
             "| Variant | AUROC | AUPRC | ECE15 | Challenge@0.5 | Challenge@locked λ0.02 | Suppression | FNR | CP95 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for v, item in summary["test"]["variants"].items():
        a = item["alphas"]["0.02"]
        lines.append(f"| {v} | {item['auroc']:.6f} | {item['auprc']:.6f} | "
                     f"{item['ece15']:.6f} | {item['challenge_at_0_5']:.5f} | "
                     f"{a['challenge_at_lambda']:.5f} | {a['suppression']:.6f} | "
                     f"{a['fnr']:.6f} | {a['fnr_ci']} |")
    lines += ["", "## Fully specified results and secondary analyses", "",
              "The JSON artifact contains alpha=0.01/0.02/0.05, bootstrap (record B=2000, seed=42),",
              "paired contrasts B/C/D/E−A, individual A seed range, contested H2, and LTT.",
              "All public counts are redacted if <10, including complements and split derivatives.", "",
              "```json", json.dumps(summary["test"], ensure_ascii=False, indent=2), "```",
              "", "## Challenge 2015 (external)", "",
              "```json", json.dumps(summary["external"], ensure_ascii=False, indent=2), "```",
              "", "No record ID, event ID, individual waveform or patient labels are included.",
              "", "**Stop. Send this report to Director. Do not rerun T05b.**"]
    with report.open("x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    manifest = root / "manifest.csv"
    now = datetime.now(timezone.utc).isoformat()
    with manifest.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for path, n in ((prereg, 0), (artifact, 0), (report, 0)):
            writer.writerow([str(path), digest(path), n, now, "P05-T05b"])
    print(f"REPORT_FILE={report}", flush=True)
    print("P05_T05B=COMPLETE; TEST_OPENED=TRUE; NO_RERUN; WAIT_FOR_DIRECTOR", flush=True)


def run(stage):
    root = Path(__file__).resolve().parents[1]
    flag = check_preopened(root)  # MUST be the first project I/O operation.
    import numpy as np
    import pandas as pd
    import yaml
    from scipy import signal as scipy_signal
    import wfdb
    import t02c_pilot as core

    core.below_normal_priority()
    lock, cfg, sha = validate_director_and_sources(root, yaml, core,
                                                   root / "configs" / "final_lock.yaml")
    sys.path.insert(0, str(Path(cfg["common_dir"]).resolve().parent))
    from common import metrics, conformal
    import t03_label_variants as t03
    import preprocess_lib as preprocess
    import t05b_external_c2015 as external
    lambdas, per_seed, calib_meta = precompute_calibration(root, cfg, lock,
                                                           np, pd, conformal, t03)
    public_folder = external.external_root(cfg)
    vt = external.audit_training(public_folder)
    public = external.prepare_public_vt(cfg, vt, public_folder,
                                        preprocess, np, wfdb, scipy_signal)
    refcfg = dict(cfg)
    refcfg["raw_dir"] = str(root.parent / "data_raw" / "vtac" / "1.1")
    refs = external.read_reference_fingerprints(refcfg, preprocess, np, pd,
                                                wfdb, scipy_signal)
    overlaps, pairs = external.flag_overlap(public, refs, np)
    print("C2015_OVERLAP_PRECOMPUTED=PASS; TEST_SEALED=TRUE", flush=True)
    import torch
    import models
    import train_lib

    test_data, test_meta = read_test_inputs(cfg, np, pd, torch)
    external_data = {
        "x": torch.from_numpy(np.array(public["x"], dtype=np.float32, copy=True, order="C")),
        "mask": torch.from_numpy(np.array(public["mask"], dtype=np.bool_, copy=True, order="C")),
        "idx": np.arange(len(public["y"]), dtype=np.int64), "y": None,
    }
    test_logits, external_logits = model_probabilities(root, lock, test_data,
                         external_data, np, torch, models, train_lib, core)
    # Public external analysis is completed before the irreversible flag. Its
    # results never influence model choice or thresholds (Director-locked).
    external_result = external.score_external(public, external_logits, lock["lambda"]["A"],
                         float(lock["ltt_comparator"]["tau"]), metrics, np, overlaps, redact)
    if stage == "inspect":
        print("T05B_INSPECT=PASS; ALL_PREFLIGHT_CHECKS_PASSED", flush=True)
        print("TEST_OPENED_FLAG=ABSENT; TEST_LABELS=SEALED; PREREG_LOCK=NOT_WRITTEN", flush=True)
        return
    hard(stage == "execute", "unknown execution mode")
    prereg = make_prereg(root, sha, lambdas, public, np)
    hard(prereg.is_file() and digest(prereg), "prereg file write failure")
    one_shot_flag(flag, sha["final_lock.yaml"])
    # The ONLY point at which sealed labels are read. Any error from here:
    # STOP, DO NOT DELETE FLAG, DO NOT RERUN, SEND TRACEBACK TO DIRECTOR.
    labels = read_sealed_labels_after_flag(cfg, flag, test_meta, pd, np)
    y = labels["y"].to_numpy(dtype=np.int8)
    hard({0, 1} == set(y), "test labels contain only one class")
    results, prob = {}, {}
    for variant in VARIANTS:
        results[variant], prob[variant] = score_test_variant(
            test_logits[variant], y, lock, variant, metrics, np)
    bootstrap = bootstrap_record(y, test_meta["record"].to_numpy(), prob, lock, np)
    h2 = h2_by_contested(labels, prob["A"], lock, metrics, np)
    stability = seed_a_stability(test_logits["A"], y, per_seed, np)
    ltt = split_counts(prob["A"], y, float(lock["ltt_comparator"]["tau"]), True,
                       metrics, np)
    summary = {"final_lock_sha256": sha["final_lock.yaml"],
               "test": {"variants": results, "bootstrap": bootstrap, "h2": h2,
                        "a_seed_stability": stability, "ltt_at_tau_0_09": ltt,
                        "alpha_0_005": "N/A; locked infeasible"},
               "external": {"total_vt": redact(len(public["y"])),
                            "positive_vt": redact(int(public["y"].sum())),
                            "negative_vt": redact(int((public["y"] == 0).sum())),
                            "overlap_pairs_by_split": {k: redact(v) for k, v in pairs.items()},
                            "overlap_records": redact(int(overlaps.sum())),
                            "results": external_result}}
    write_results(root, summary, prereg, np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P05-T05b one-shot (Director approved)")
    parser.add_argument("--stage", choices=("inspect", "execute"), default="inspect")
    arguments = parser.parse_args()
    try:
        run(arguments.stage)
    except BaseException:
        print("T05B_HARD_FAILURE. If TEST_OPENED.flag exists: DO NOT RERUN or delete it. Send log to Director.",
              file=sys.stderr, flush=True)
        raise
