"""P05-T02c entry point: GPU1 test, three-epoch pilot, or approved full runs.

By default this script does NOT train. It never reads any test data.
Requires Director's acceptance of pilot before invoking --stage full.
"""

# Environment must be set BEFORE any import of PyTorch.
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
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


EXPECTED_LOCK_SHA = "b977927a66ed50c69ea1f98c72e8bdb0639cd89e5949bf526b3dcc888261f819"
EXPECTED_COMMON_SHA = "90b7ed5d87d174e38ad801430341c4167f70a623ad8b8d5173229029aa9d9fa6"
EXPECTED_MODELS = ("fcn", "resnet1d", "s4d")


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for data in iter(lambda: fp.read(1024 * 1024), b""):
            hasher.update(data)
    return hasher.hexdigest()


def common_source_sha256(common_dir):
    base = Path(common_dir)
    sources = sorted((p for p in base.rglob("*.py")
                      if "__pycache__" not in p.parts),
                     key=lambda p: p.relative_to(base).as_posix())
    if len(sources) != 6:
        raise RuntimeError("FROZEN_COMMON_MISMATCH: source file count")
    hasher = hashlib.sha256()
    for source in sources:
        hasher.update(source.relative_to(base).as_posix().encode("utf-8") + b"\0")
        hasher.update(bytes.fromhex(sha256_file(source)))
    return hasher.hexdigest()


def validate_frozen(yaml_path):
    """Run before PyTorch, datasets, or model imports. Hard stop on mismatch."""
    with Path(yaml_path).open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid T02c configuration")
    if (cfg.get("expected_locked_sha256") != EXPECTED_LOCK_SHA
            or cfg.get("expected_common_tree_sha256") != EXPECTED_COMMON_SHA):
        raise RuntimeError("Frozen hashes in T02c config are not Director-approved")
    locked_actual = sha256_file(cfg["locked_design"])
    print(f"FROZEN_LOCKED_DESIGN_SHA256={locked_actual}", flush=True)
    if locked_actual != EXPECTED_LOCK_SHA:
        raise RuntimeError("FROZEN_LOCKED_DESIGN_MISMATCH: STOP")
    common_actual = common_source_sha256(cfg["common_dir"])
    print(f"FROZEN_COMMON_SHA256={common_actual}", flush=True)
    if common_actual != EXPECTED_COMMON_SHA:
        raise RuntimeError("FROZEN_COMMON_MISMATCH: STOP")
    if ((int(cfg["window_w"]), int(cfg["window_d"])) != (10, 0)
            or int(cfg["cpu_threads"]) not in range(1, 9)
            or int(cfg["gpu_physical_index"]) != 1
            or int(cfg["pilot_epochs"]) != 3
            or list(cfg["seeds"]) != [0, 1, 2]
            or int(cfg["max_epochs"]) != 150
            or int(cfg["patience"]) != 20
            or bool(cfg["amp"]) or cfg["dtype"] != "float32"):
        raise RuntimeError("Director-locked T02c settings differ")
    required = {
        "fcn": ("Adam", 1e-4, 5e-3, 64, 4.0, False),
        "resnet1d": ("AdamW", 1e-3, 1e-2, 128, 1.0, True),
        "s4d": ("AdamW", 1e-3, 1e-2, 128, 1.0, True),
    }
    if set(cfg["model_specs"]) != set(required):
        raise RuntimeError("Model list differs from Director-approved settings")
    for name, expected in required.items():
        opt = cfg["model_specs"][name]
        actual = (opt["optimizer"], float(opt["lr"]),
                  float(opt["weight_decay"]), int(opt["batch_size"]),
                  float(opt["pos_weight"]), bool(opt["cosine_decay"]))
        if actual != expected:
            raise RuntimeError(f"Director training hyperparameters mismatch: {name}")
    if (float(cfg["g1_val_auroc_min"]) != 0.92
            or float(cfg["g1_val_challenge_score_min"]) != 75.0
            or list(cfg["backbone_candidates"]) != ["resnet1d", "s4d"]):
        raise RuntimeError("G1 criteria or backbone candidates have changed")
    print("FROZEN_INPUTS=PASS; model training may initialize", flush=True)
    return cfg


def below_normal_priority():
    if os.name == "nt":
        import ctypes
        # Win64 HANDLE is pointer-sized. ctypes defaults to c_int, which can
        # truncate GetCurrentProcess()'s pseudo-handle and cause ERROR_INVALID_HANDLE.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel32.GetPriorityClass.restype = ctypes.c_uint32
        process = kernel32.GetCurrentProcess()
        ctypes.set_last_error(0)
        success = kernel32.SetPriorityClass(process, 0x4000)
        if not success:
            code = ctypes.get_last_error()
            raise RuntimeError(
                "Unable to set Windows BELOW_NORMAL_PRIORITY_CLASS; "
                f"winerror={code}")
        actual = kernel32.GetPriorityClass(process)
        if actual != 0x4000:
            raise RuntimeError(
                "Unable to verify Windows BELOW_NORMAL_PRIORITY_CLASS; "
                f"actual_class={actual}")
        print("WINDOWS_PROCESS_PRIORITY=BelowNormal; PASS", flush=True)
    else:
        print("PROCESS_PRIORITY=non-Windows; Windows BelowNormal check deferred", flush=True)


def run_nvidia_smi(*args):
    try:
        proc = subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, timeout=20, check=True)
        return proc.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired) as exc:
        raise RuntimeError("nvidia-smi verification unavailable; STOP") from exc


def verify_nvidia_process_is_gpu1():
    """Require the *current PID* in the compute-process list for GPU1 UUID."""
    gpu_lines = run_nvidia_smi("--query-gpu=index,uuid,name",
                               "--format=csv,noheader")
    gpu_info = {}
    for line in gpu_lines.splitlines():
        fields = [piece.strip() for piece in line.split(",", 2)]
        if len(fields) == 3 and fields[0].isdigit():
            gpu_info[int(fields[0])] = (fields[1], fields[2])
    if 1 not in gpu_info:
        raise RuntimeError("Physical GPU index 1 absent from nvidia-smi; STOP")
    target_uuid, target_name = gpu_info[1]
    print(f"NVIDIA_SMI_GPU1_NAME={target_name}; GPU1_UUID={target_uuid}",
          flush=True)
    compute_lines = run_nvidia_smi("--query-compute-apps=pid,gpu_uuid",
                                   "--format=csv,noheader")
    match = False
    for line in compute_lines.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            if int(parts[0]) == os.getpid():
                if parts[1] != target_uuid:
                    raise RuntimeError("Current Python PID found on wrong GPU; STOP")
                match = True
    if not match:
        raise RuntimeError(
            "Current Python PID not verifiable on physical GPU1 via "
            "nvidia-smi compute-apps; STOP and report to Director")
    print(f"NVIDIA_SMI_PROCESS_GPU1=PASS; pid={os.getpid()}", flush=True)


def gpu_smoke_test(torch, models, train_lib, cfg):
    device = train_lib.cuda_device()
    train_lib.seed_everything(42, cfg["cpu_threads"])
    x = torch.randn(64, 8, 2500, device=device, dtype=torch.float32)
    y = torch.fft.irfft(torch.fft.rfft(x, dim=-1), n=2500, dim=-1)
    diff = float(torch.max(torch.abs(y - x)))
    if not torch.allclose(x, y, rtol=1e-5, atol=1e-5):
        raise RuntimeError("GPU1 rfft/irfft forward identity failed")
    print(f"GPU1_RFFT_IRFFT=PASS; shape=[64,8,2500]; max_abs_error={diff:.3g}",
          flush=True)
    del x, y
    model, parameter_count = models.make_model("s4d")
    model = model.to(device)
    model.train()
    xi = torch.randn(2, 8, 2500, device=device, dtype=torch.float32,
                     requires_grad=True)
    logits = model(xi)
    logits.square().mean().backward()
    if (logits.shape != (2,) or not torch.isfinite(logits).all()
            or xi.grad is None or not torch.isfinite(xi.grad).all()
            or not any(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters())):
        raise RuntimeError("S4D forward/backward failed on CUDA GPU1")
    print(f"GPU1_S4D_FORWARD_BACKWARD=PASS; parameters={parameter_count}",
          flush=True)
    torch.cuda.synchronize(device)
    verify_nvidia_process_is_gpu1()
    print(f"GPU1_SMOKE=PASS; VRAM_peak_GiB="
          f"{torch.cuda.max_memory_allocated(device)/2**30:.3f}", flush=True)
    del model, xi, logits
    torch.cuda.empty_cache()
    return device


def load_common_metrics(cfg):
    # Only after exact hash of entire frozen common/ tree is established.
    sys.path.insert(0, str(Path(cfg["common_dir"]).resolve()))
    return importlib.import_module("metrics")


def _append_manifest(cfg, files, train_lib, stage):
    manifest = Path(cfg["project_dir"]) / "manifest.csv"
    existing = manifest.exists()
    now = datetime.now(timezone.utc).isoformat()
    with manifest.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        if not existing:
            writer.writerow(["path", "sha256", "n_samples",
                             "timestamp_utc", "step"])
        for path, n in files:
            writer.writerow([str(path), train_lib.sha256_file(path),
                             n, now, stage])


def pilot(cfg, device, common, train_lib, sha):
    print("=== P05-T02c PILOT: THREE EPOCHS PER MODEL ONLY ===", flush=True)
    print("FIT_VALIDATION_ONLY=TRUE; NO_TEST_FILES_ACCESSED=TRUE", flush=True)
    fit = train_lib.load_split(cfg["proc_dir"], "fit")
    val = train_lib.load_split(cfg["proc_dir"], "val")
    results = []
    started = time.perf_counter()
    for model in EXPECTED_MODELS:
        print(f"PILOT_MODEL_BEGIN={model}", flush=True)
        result = train_lib.train_model(
            model, int(cfg["pilot_seed"]), fit, val, cfg, device, common,
            max_epochs=3, is_pilot=True)
        results.append(result)
        print(f"PILOT_MODEL_END={model}; seconds_per_epoch="
              f"{result['seconds_per_epoch_mean']:.2f}; "
              f"VRAM_GiB={result['peak_vram_gib']:.3f}", flush=True)
    pilot_wall = time.perf_counter() - started
    out_root = Path(cfg["project_dir"]) / "outputs" / "t02c"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = out_root / f"pilot_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    report_lines = [
        "# REPORT P05-T02c PILOT",
        "",
        "Status: PASS, pending Director acceptance of pilot.",
        "Data: fit+val W10D0 only. No test data accessed.",
        "Device: physical NVIDIA GPU1 only; nvidia-smi PID verified.",
        "Precision: FP32 only, no AMP. CPU threads <= 8; BelowNormal.",
        "Epochs: exactly three per model, seed 0. Pilot is not final model selection.",
        f"Locked design SHA-256: {EXPECTED_LOCK_SHA}",
        f"Frozen common SHA-256: {EXPECTED_COMMON_SHA}",
        f"Scripts SHA-256: `{json.dumps(sha, sort_keys=True)}`",
        "",
        "| Model | Parameters | Epochs | s/epoch (mean) | s/epoch (sd) | VRAM peak (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        report_lines.append(
            f"| {item['model']} | {item['parameters']} | 3 | "
            f"{item['seconds_per_epoch_mean']:.2f} | "
            f"{item['seconds_per_epoch_sd']:.2f} | "
            f"{item['peak_vram_gib']:.3f} |")
    t02c_hours = sum(3 * int(cfg["max_epochs"]) *
                     item["seconds_per_epoch_mean"] for item in results) / 3600
    report_lines += [
        "",
        f"Pilot wall time (s): {pilot_wall:.2f}",
        f"T02c 9-run 150-epoch upper-schedule estimate (h): {t02c_hours:.2f}",
        "T03 estimate assumes 5 label variants x 5 seeds = 25 runs of "
        "the selected backbone, 150 epochs each, with no early stopping.",
    ]
    for item in results:
        t03_hours = 25 * int(cfg["max_epochs"]) * item["seconds_per_epoch_mean"] / 3600
        report_lines.append(
            f"T03 if backbone {item['model']} (h): {t03_hours:.2f}")
    report_lines += [
        "", "**STOP: Send this REPORT PILOT to Director.**",
        "Do not start 9 full runs until Director accepts this pilot.",
        "", "Pilot validation metrics are diagnostic only; not used to "
        "choose backbone or change architecture.",
    ]
    report_file = output / "REPORT_PILOT.md"
    report_file.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    json_file = output / "pilot_metrics.json"
    json_file.write_text(json.dumps({
        "status": "PASS", "pilot_wall_seconds": pilot_wall,
        "results": results, "sha256": sha,
        "t02c_full_9_run_estimate_hours": t02c_hours,
        "t03_25run_estimates_hours": {
            item["model"]: 25 * int(cfg["max_epochs"]) *
            item["seconds_per_epoch_mean"] / 3600 for item in results},
    }, indent=2), encoding="utf-8")
    _append_manifest(cfg, [(report_file, 0), (json_file, 0)], train_lib,
                     "P05-T02c-pilot")
    print("=== REPORT P05-T02c PILOT ===", flush=True)
    print("status: PASS", flush=True)
    for item in results:
        print(f"{item['model']}: s_per_epoch={item['seconds_per_epoch_mean']:.2f}; "
              f"vram_peak_GiB={item['peak_vram_gib']:.3f}; "
              f"parameters={item['parameters']}", flush=True)
    print(f"estimate_T02c_9run_full_hours_upper={t02c_hours:.2f}", flush=True)
    for item in results:
        t03 = 25 * int(cfg["max_epochs"]) * item["seconds_per_epoch_mean"] / 3600
        print(f"estimate_T03_25run_{item['model']}_hours_upper={t03:.2f}",
              flush=True)
    print(f"pilot_report={report_file}", flush=True)
    print("FULL_RUNS=NOT_STARTED; WAIT_FOR_DIRECTOR", flush=True)
    print("=== END REPORT ===", flush=True)


def full_runs(cfg, device, common, train_lib, sha):
    import numpy as np

    print("=== P05-T02c FULL RUNS, DIRECTOR-APPROVED PILOT ===", flush=True)
    output = Path(cfg["project_dir"]) / "outputs" / "t02c"
    output.mkdir(parents=True, exist_ok=True)
    fit = train_lib.load_split(cfg["proc_dir"], "fit")
    val = train_lib.load_split(cfg["proc_dir"], "val")
    results = []
    started = time.perf_counter()
    for model in EXPECTED_MODELS:
        for seed in cfg["seeds"]:
            run_dir = output / f"{model}_s{seed}"
            if train_lib.completed_run(run_dir):
                item = train_lib.load_completed_result(
                    run_dir, model, seed, sha, EXPECTED_LOCK_SHA)
                print(f"RUN_SKIPPED_COMPLETE={model}_s{seed}; source_sha_verified=PASS",
                      flush=True)
            else:
                item = train_lib.train_model(
                    model, int(seed), fit, val, cfg, device, common,
                    output_dir=run_dir, sha=sha)
                _append_manifest(cfg, [
                    (run_dir / "ckpt.pt", 0),
                    (run_dir / "scores_val.parquet", len(val["y"])),
                    (run_dir / "scores_calib.parquet", 1230),
                    (run_dir / "log.csv", item["epochs_run"]),
                ], train_lib, "P05-T02c-full")
            results.append(item)
    summaries = {}
    for model in EXPECTED_MODELS:
        group = [r for r in results if r["model"] == model]
        summaries[model] = {}
        for key in ("auroc", "auprc", "ece", "score_threshold_0p5",
                    "score_val_optimized_optimistic"):
            values = np.array([r["metrics"][key] for r in group],
                              dtype=np.float64)
            summaries[model][key] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
            }
    candidates = list(cfg["backbone_candidates"])
    chosen = max(candidates, key=lambda name: summaries[name]["auprc"]["mean"])
    gate_auroc = summaries[chosen]["auroc"]["mean"]
    gate_score = summaries[chosen]["score_val_optimized_optimistic"]["mean"]
    g1_pass = (gate_auroc >= float(cfg["g1_val_auroc_min"])
               and gate_score >= float(cfg["g1_val_challenge_score_min"]))
    gpu_hours = sum(r["wall_seconds"] for r in results) / 3600
    wall_time = time.perf_counter() - started
    selected_epoch = np.mean([r["seconds_per_epoch_mean"] for r in results
                              if r["model"] == chosen])
    t03_upper = 25 * int(cfg["max_epochs"]) * selected_epoch / 3600
    lines = [
        "# REPORT P05-T02c FULL RUNS", "",
        f"Status: {'PASS G1' if g1_pass else 'FAIL G1; STOP'}.\n",
        "Data: fit/val/calib W10D0. No test files read.",
        "GPU: physical GPU1, FP32. CPU <=8, BelowNormal.",
        f"Locked design SHA-256: {EXPECTED_LOCK_SHA}",
        f"Frozen common SHA-256: {EXPECTED_COMMON_SHA}",
        f"Script SHA-256: `{json.dumps(sha, sort_keys=True)}`", "",
        "All entries mean ± sample SD over seeds 0, 1, 2.",
        "Challenge Score@val-opt uses val-optimized threshold; "
        "this metric is optimistic.", "",
        "| Model | AUROC | AUPRC | ECE | Challenge@0.5 | Challenge@val-opt (optimistic) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = ["auroc", "auprc", "ece", "score_threshold_0p5",
            "score_val_optimized_optimistic"]
    for model in EXPECTED_MODELS:
        cells = [f"{summaries[model][k]['mean']:.4f} ± "
                 f"{summaries[model][k]['sd']:.4f}" for k in keys]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines += [
        "", f"Chosen backbone (highest mean val AUPRC, candidates resnet1d/s4d): {chosen}",
        f"G1 mean val AUROC: {gate_auroc:.4f}; required >=0.92.",
        f"G1 mean val-optimized Challenge Score: {gate_score:.4f}; required >=75.",
        f"G1 status: {'PASS' if g1_pass else 'FAIL; STOP' }.",
        f"Aggregate run GPU time (h): {gpu_hours:.3f}.",
        f"Total elapsed wall time (h): {wall_time / 3600:.3f}.",
        f"Estimated T03 25-run 150-epoch schedule for chosen backbone (h): {t03_upper:.2f}.",
        "", "| Model and seed | Epochs | s/epoch | Peak VRAM GiB | Best epoch | val-opt threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['model']}_s{item['seed']} | {item['epochs_run']} | "
            f"{item['seconds_per_epoch_mean']:.2f} | "
            f"{item['peak_vram_gib']:.3f} | {item['best_epoch']} | "
            f"{item['metrics']['val_optimized_threshold']:.8f} |")
    lines += ["", "G1 FAIL means no further experiment without Director decision."
              if not g1_pass else "G1 passed; await Director acceptance before T03."]
    report = output / "REPORT_T02c.md"
    if report.exists():
        raise FileExistsError("T02c final report exists; refusing overwrite")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _append_manifest(cfg, [(report, 0)], train_lib, "P05-T02c-full")
    print("=== REPORT P05-T02c ===", flush=True)
    print(f"status: {'PASS' if g1_pass else 'FAIL'}", flush=True)
    for model in EXPECTED_MODELS:
        values = summaries[model]
        print(f"{model}: AUROC={values['auroc']['mean']:.4f}±{values['auroc']['sd']:.4f}; "
              f"AUPRC={values['auprc']['mean']:.4f}±{values['auprc']['sd']:.4f}; "
              f"ECE={values['ece']['mean']:.4f}±{values['ece']['sd']:.4f}; "
              f"Challenge@0.5={values['score_threshold_0p5']['mean']:.2f}±"
              f"{values['score_threshold_0p5']['sd']:.2f}; "
              f"Challenge@val-opt_optimistic="
              f"{values['score_val_optimized_optimistic']['mean']:.2f}±"
              f"{values['score_val_optimized_optimistic']['sd']:.2f}",
              flush=True)
    print(f"backbone_chosen={chosen}; G1={'PASS' if g1_pass else 'FAIL'}",
          flush=True)
    print(f"total_gpu_hours={gpu_hours:.3f}; t03_hours_upper={t03_upper:.2f}",
          flush=True)
    print(f"report={report}", flush=True)
    print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
    print("=== END REPORT ===", flush=True)
    if not g1_pass:
        raise RuntimeError("G1 FAIL; STOP, report to Director")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gpu-check", "pilot", "full"),
                        required=True)
    parser.add_argument("--config", default=str(
        Path(__file__).resolve().parents[1] / "configs" / "t02c.yaml"))
    parser.add_argument("--director-approval", default="")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        cfg = validate_frozen(args.config)  # No model or data import yet.
        if args.stage == "full" and args.director_approval != "T02C_PILOT_ACCEPTED":
            raise RuntimeError("Pilot not approved: refuse to launch 9 full runs")
        below_normal_priority()
        # CUDA visibility was set at top of script, before any torch import.
        import torch
        import models
        import train_lib
        print(f"PYTHON={platform.python_version()}; TORCH={torch.__version__}; "
              "FP32=TRUE; AMP=FALSE; CPU_THREADS<=8", flush=True)
        common = load_common_metrics(cfg)
        sha = train_lib.script_hashes(cfg["project_dir"], args.config, __file__)
        if args.stage == "full":
            pilots = sorted((Path(cfg["project_dir"]) / "outputs" / "t02c").glob(
                "pilot_*/pilot_metrics.json"))
            prior_matching_pilot = False
            for pilot_json in pilots:
                with pilot_json.open(encoding="utf-8") as fp:
                    prior = json.load(fp)
                if prior.get("status") == "PASS" and prior.get("sha256") == sha:
                    prior_matching_pilot = True
                    break
            if not prior_matching_pilot:
                raise RuntimeError("Pilot missing or pilot SHA mismatch; refuse full runs")
        device = gpu_smoke_test(torch, models, train_lib, cfg)
        if args.stage == "gpu-check":
            print("=== REPORT P05-T02c STEP0 ===", flush=True)
            print("status: PASS; physical_gpu=1; rfft_irfft=PASS; s4d_backward=PASS",
                  flush=True)
            print("next: run --stage pilot; FULL_RUNS=NOT_STARTED", flush=True)
            print(f"runtime_s={time.perf_counter()-started:.2f}", flush=True)
            print("=== END REPORT ===", flush=True)
        elif args.stage == "pilot":
            pilot(cfg, device, common, train_lib, sha)
        else:
            full_runs(cfg, device, common, train_lib, sha)
        return 0
    except Exception as exc:
        print("=== REPORT P05-T02c ===", flush=True)
        print("status: FAIL", flush=True)
        print(f"error_type: {type(exc).__name__}", flush=True)
        # Controlled messages are safe; raw third-party exceptions can include PHI paths.
        message = str(exc)
        safe_prefixes = (
            "FROZEN_", "Physical GPU", "Current Python PID",
            "nvidia-smi", "GPU1 ", "Pilot not approved", "Pilot missing",
            "Director-locked", "Director training", "G1 FAIL",
            "S4D forward", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
            "Unable to set Windows", "GPU1 isolation",
        )
        reason = message if message.startswith(safe_prefixes) else "halted at safety gate"
        print(f"detail: {reason}", flush=True)
        print("TEST=SEALED; NO_TEST_FILES_READ", flush=True)
        print("=== END REPORT ===", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
