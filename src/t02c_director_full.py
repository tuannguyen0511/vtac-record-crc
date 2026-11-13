"""Director-approved P05-T02c full-run coordinator, without frozen-code edits.

Run: python -u P05_VTaC_FalseAlarm/src/t02c_director_full.py \
    --director-approval T02C_PILOT_ACCEPTED

Keeps t02c_pilot.py, models.py, train_lib.py and t02c.yaml byte-identical
to the approved HF01-HF03 Pilot. Imports the original entry point and changes
only its in-memory execution order for the nine full runs.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

# The original entry point configures GPU visibility before importing PyTorch.
import t02c_pilot as core


ORDER = ("resnet1d", "fcn", "s4d")
SEEDS = (0, 1, 2)
APPROVAL = "T02C_PILOT_ACCEPTED"
EXPECTED_CODE_HASHES = {
    "models.py": "4c082cbe1f14de69fda9b251e94970099fa900b31f37715c0ce156b45d2b070c",
    "train_lib.py": "8015e99d240cbc87198e51c3c04cebc3758899f885be13f3d7f7ba9b08c581bc",
    "t02c_pilot.py": "7eeac13d574e37e5c0b4692dc603e580c8bd0c0e770d3cce65790a0245db443b",
}
METRIC_KEYS = (
    "auroc", "auprc", "ece", "score_threshold_0p5",
    "score_val_optimized_optimistic",
)
METRIC_LABELS = (
    "AUROC", "AUPRC", "ECE", "Challenge@0.5",
    "Challenge@val-opt (optimistic)",
)


def hard(condition, message):
    if not condition:
        raise RuntimeError("DIRECTOR_HARD_STOP: " + message)


def sha256_file(path):
    return core.sha256_file(path)


def preflight(root):
    cfg_file = root / "configs" / "t02c.yaml"
    expected = {}
    for name, digest in EXPECTED_CODE_HASHES.items():
        path = root / "src" / name
        hard(path.is_file(), f"missing {name}")
        actual = sha256_file(path)
        hard(actual == digest, f"locked source mismatch: {name}")
        expected[{"models.py": "models_py", "train_lib.py": "train_lib_py",
                  "t02c_pilot.py": "t02c_pilot_py"}[name]] = actual
        print(f"DIRECTOR_LOCKED_{name}_SHA256={actual}", flush=True)
    cfg = core.validate_frozen(cfg_file)
    hard(tuple(cfg["seeds"]) == SEEDS, "seed list modified")
    hard(tuple(cfg["backbone_candidates"]) == ("resnet1d", "s4d"),
         "backbone candidates modified")
    expected["t02c_yaml"] = sha256_file(cfg_file)

    output = Path(cfg["project_dir"]) / "outputs" / "t02c"
    pilot_files = sorted(output.glob("pilot_*/pilot_metrics.json"))
    matches = []
    for path in pilot_files:
        with path.open("r", encoding="utf-8") as stream:
            pilot = json.load(stream)
        if pilot.get("status") == "PASS" and pilot.get("sha256") == expected:
            matches.append(path)
    hard(bool(matches), "no accepted-Pilot-compatible file hashes")
    print(f"PILOT_SHA_MATCH=PASS; source={matches[-1]}", flush=True)
    print("DIRECTOR_ORDER=resnet1d,fcn,s4d; SEEDS=0,1,2", flush=True)
    print("FROZEN_CODE=PASS; TEST=SEALED", flush=True)
    return cfg, cfg_file, expected


def check_completed_artifact(run_dir, result, cfg):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(run_dir / "log.csv")
    hard(not df.empty, f"empty log {run_dir.name}")
    hard(len(df) == int(result["epochs_run"]),
         f"epoch count mismatch {run_dir.name}")
    hard((df["epoch"].to_numpy() ==
          np.arange(1, len(df) + 1)).all(),
         f"nonconsecutive epochs {run_dir.name}")

    loss = pd.to_numeric(df["train_loss"], errors="coerce").to_numpy(dtype=float)
    val_ap = pd.to_numeric(df["val_auprc"], errors="coerce").to_numpy(dtype=float)
    nonfinite_loss = bool(np.any(~np.isfinite(loss)))
    hard(not nonfinite_loss, f"NaN/Inf loss in {run_dir.name}")
    hard(bool(np.isfinite(val_ap).all()),
         f"NaN/Inf validation metric in {run_dir.name}")
    hard(1 <= int(result["best_epoch"]) <= len(df) <= int(cfg["max_epochs"]),
         f"invalid best epoch {run_dir.name}")
    hard(math.isfinite(float(result["wall_seconds"])) and
         float(result["wall_seconds"]) > 0,
         f"invalid run wall time {run_dir.name}")

    for key in (*METRIC_KEYS, "val_optimized_threshold"):
        hard(math.isfinite(float(result["metrics"][key])),
             f"nonfinite {key} in {run_dir.name}")

    for split, expected_rows in (("val", 495), ("calib", 1230)):
        scores = pd.read_parquet(run_dir / f"scores_{split}.parquet")
        hard(list(scores.columns) == ["idx", "logit"],
             f"unexpected {split} score schema in {run_dir.name}")
        hard(len(scores) == expected_rows and scores["idx"].is_unique,
             f"unexpected {split} score rows in {run_dir.name}")
        hard(bool(np.isfinite(scores["logit"].to_numpy(dtype=float)).all()),
             f"nonfinite {split} logits in {run_dir.name}")

    return {
        **result,
        "reached_150": int(result["epochs_run"]) == 150,
        "loss_nan_inf": nonfinite_loss,
        # Training wall time on physical GPU1, NOT kernel-level GPU busy time.
        "gpu_run_wall_hours": float(result["wall_seconds"]) / 3600,
    }


def load_and_audit_runs(root, cfg, expected):
    import train_lib

    output = root / "outputs" / "t02c"
    runs = []
    for model in ORDER:
        for seed in SEEDS:
            run_dir = output / f"{model}_s{seed}"
            hard(train_lib.completed_run(run_dir),
                 f"missing completed run {run_dir.name}")
            result = train_lib.load_completed_result(
                run_dir, model, seed, expected,
                core.EXPECTED_LOCK_SHA)
            runs.append(check_completed_artifact(run_dir, result, cfg))
    hard(len(runs) == 9, "nine-run completeness check failed")
    return runs


def aggregate(runs, cfg):
    import numpy as np

    summary = {}
    for model in ORDER:
        subset = [r for r in runs if r["model"] == model]
        hard(len(subset) == 3 and
             tuple(r["seed"] for r in subset) == SEEDS,
             f"missing seeds for {model}")
        summary[model] = {}
        for key in METRIC_KEYS:
            values = np.array([r["metrics"][key] for r in subset], dtype=float)
            hard(bool(np.isfinite(values).all()), f"nonfinite {model} {key}")
            summary[model][key] = (
                float(values.mean()), float(values.std(ddof=1)))

    candidates = tuple(cfg["backbone_candidates"])
    chosen = max(candidates, key=lambda name: summary[name]["auprc"][0])
    gate_auroc = summary[chosen]["auroc"][0]
    gate_challenge = summary[chosen]["score_val_optimized_optimistic"][0]
    passed = (gate_auroc >= float(cfg["g1_val_auroc_min"])
              and gate_challenge >= float(cfg["g1_val_challenge_score_min"]))
    return summary, chosen, gate_auroc, gate_challenge, passed


def write_director_report(root, cfg, expected, runs, summary, chosen,
                          gate_auroc, gate_challenge, passed):
    import train_lib

    output = root / "outputs" / "t02c"
    report = output / "REPORT_T02c_DIRECTOR_FULL.md"
    wrapper_sha = sha256_file(Path(__file__).resolve())
    standard_report = output / "REPORT_T02c.md"
    hard(standard_report.is_file(), "standard full-run report missing")
    report_utc = datetime.fromtimestamp(
        standard_report.stat().st_mtime, timezone.utc).isoformat()
    lines = [
        "# P05-T02c: Director Full-Run Report", "",
        f"Status: {'G1 PASS' if passed else 'G1 FAIL; STOP T03'}.",
        "Completed report timestamp (UTC): " + report_utc,
        "Approval: T02C_PILOT_ACCEPTED (November 10, 2026).",
        "Execution order: resnet1d -> fcn -> s4d; seeds 0, 1, 2.",
        "Data: fit/val training and validation; calib only receives saved logits.",
        "Test labels and test files: SEALED; NOT READ.",
        "Hardware: physical GPU1 Quadro M6000; FP32, AMP off, CPU <= 8, BelowNormal.",
        "Early stopping: validation AUPRC only; patience=20; max_epochs=150.",
        "No post-Pilot architecture or hyperparameter changes.",
        f"Locked design SHA256: {core.EXPECTED_LOCK_SHA}",
        f"Common SHA256: {core.EXPECTED_COMMON_SHA}",
        "Pilot-compatible code hashes: `" + json.dumps(expected, sort_keys=True) + "`",
        f"New independent Director coordinator SHA256: {wrapper_sha}", "",
        "## Per-model validation metrics (mean +/- sample SD, n=3 seeds)", "",
        "Challenge@val-opt optimizes the threshold on validation itself and is optimistic.",
        "| Model | AUROC | AUPRC | ECE | Challenge@0.5 | Challenge@val-opt (optimistic) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in ORDER:
        cells = []
        for key in METRIC_KEYS:
            mean, sd = summary[model][key]
            fmt = ".2f" if key.startswith("score_") else ".4f"
            cells.append(f"{format(mean, fmt)} +/- {format(sd, fmt)}")
        lines.append("| " + model + " | " + " | ".join(cells) + " |")

    lines += [
        "", "## Per-run execution audit", "",
        "GPU run hours measure actual training wall-clock time while using GPU1; "
        "they are not kernel-level GPU utilization hours. "
        "The source checkpoint excludes subsequent calibration scoring/file I/O.",
        "| Run | Epochs | Best epoch | Reached 150? | NaN/Inf train loss? | GPU run wall hours | Peak VRAM GiB |",
        "| --- | ---: | ---: | :---: | :---: | ---: | ---: |",
    ]
    for r in runs:
        lines.append(
            f"| {r['model']}_s{r['seed']} | {r['epochs_run']} | {r['best_epoch']} | "
            f"{'YES' if r['reached_150'] else 'NO'} | "
            f"{'YES' if r['loss_nan_inf'] else 'NO'} | "
            f"{r['gpu_run_wall_hours']:.4f} | {r['peak_vram_gib']:.3f} |")

    total_gpu_wall_h = sum(r["gpu_run_wall_hours"] for r in runs)
    lines += [
        "", "## G1 decision", "",
        "Eligible backbones: resnet1d and s4d only (FCN excluded).",
        f"Selected backbone: {chosen} (highest mean validation AUPRC).",
        f"Selected mean validation AUROC: {gate_auroc:.6f}; required >= 0.92.",
        f"Selected mean optimistic validation Challenge Score: {gate_challenge:.6f}; required >= 75.",
        f"G1: {'PASS' if passed else 'FAIL'}.",
        f"Sum of nine GPU training run wall hours: {total_gpu_wall_h:.4f}.",
        "", "## Operational disposition", "",
        ("G1 PASS: hold T03 pending a separate Director decision."
         if passed else
         "G1 FAIL: STOP. Do not run T03 or adjust training to rescue G1."),
        "Frozen code and checkpoints remain unchanged.",
    ]
    content = "\n".join(lines) + "\n"
    if report.exists():
        # A completed report is immutable. Do not overwrite audit artifacts.
        hard(report.read_text(encoding="utf-8") == content,
             "Director report already exists; preserve without overwrite")
        print(f"DIRECTOR_REPORT_ALREADY_VERIFIED={report}", flush=True)
    else:
        with report.open("x", encoding="utf-8") as stream:
            stream.write(content)
        core._append_manifest(cfg, [(report, 0)], train_lib, "P05-T02c-director-full")
        print(f"DIRECTOR_REPORT_SAVED={report}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--director-approval", required=True)
    parser.add_argument("--stage", choices=("inspect", "full"), default="full")
    args = parser.parse_args()
    hard(args.director_approval == APPROVAL, "Director token incorrect")

    root = Path(__file__).resolve().parents[1]
    cfg, cfg_file, expected = preflight(root)
    if args.stage == "inspect":
        print("DIRECTOR_INSPECT=PASS; FULL_RUNS=NOT_STARTED", flush=True)
        return 0

    output = root / "outputs" / "t02c"
    standard_report = output / "REPORT_T02c.md"
    if not standard_report.exists():
        print("DIRECTOR_FULL_BEGIN; NO_TEST_FILES_READ", flush=True)
        core.EXPECTED_MODELS = ORDER  # In-memory order only; no frozen file edits.
        saved_argv = sys.argv
        try:
            sys.argv = [str(Path(core.__file__).resolve()), "--stage", "full",
                        "--config", str(cfg_file), "--director-approval", APPROVAL]
            result_code = core.main()
        finally:
            sys.argv = saved_argv
        if not standard_report.exists():
            hard(result_code == 0, "training interrupted or failed; preserve outputs")
            raise RuntimeError("DIRECTOR_HARD_STOP: full-run report not produced")
    else:
        print("EXISTING_FINAL_REPORT=FOUND; audit completed runs only", flush=True)

    runs = load_and_audit_runs(root, cfg, expected)
    summary, chosen, auroc, challenge, passed = aggregate(runs, cfg)
    write_director_report(root, cfg, expected, runs, summary, chosen,
                          auroc, challenge, passed)
    print("=== REPORT P05-T02c DIRECTOR FULL ===", flush=True)
    print("ALL_NINE_RUNS=PASS; CHECKPOINT_AND_SCORES_VERIFIED=PASS", flush=True)
    for model in ORDER:
        parts = []
        for key, label in zip(METRIC_KEYS, METRIC_LABELS):
            mean, sd = summary[model][key]
            parts.append(f"{label}={mean:.5f}+/-{sd:.5f}")
        print(model + ": " + "; ".join(parts), flush=True)
    print(f"SELECTED_BACKBONE={chosen}; G1={'PASS' if passed else 'FAIL'}", flush=True)
    print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
    if not passed:
        print("DIRECTOR_HARD_STOP=G1_FAIL; NO_T03; NO_RESCUE_TUNING", flush=True)
    else:
        print("T03=NOT_STARTED; WAIT_FOR_DIRECTOR", flush=True)
    print("=== END REPORT ===", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("DIRECTOR_FULL_INTERRUPTED; completed runs remain reusable", flush=True)
        sys.exit(130)
    except Exception as exc:
        print("=== REPORT P05-T02c DIRECTOR COORDINATOR ===", flush=True)
        print("status: FAIL; no source or partial output deleted", flush=True)
        message = str(exc)
        if message.startswith("DIRECTOR_HARD_STOP:"):
            print(message, flush=True)
        else:
            print(f"error_type={type(exc).__name__}; inspect local run log", flush=True)
        print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
        print("=== END REPORT ===", flush=True)
        sys.exit(1)
