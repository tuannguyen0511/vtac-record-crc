"""P05-T02c training helpers. Uses only fit, val and calib PROC files.

Never import this module before verifying Director-frozen SHA-256 values.
Never request the test split or sealed labels from this library.
"""

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from models import make_model


ALLOWED_SPLITS = frozenset(("fit", "calib", "val"))
EXPECTED_EVENTS = {"fit": 2830, "calib": 1230, "val": 495}
EXPECTED_RECORDS = {"fit": 1265, "calib": 543, "val": 226}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed, cpu_threads=8):
    if int(cpu_threads) not in range(1, 9):
        raise ValueError("CPU threads must be between 1 and 8")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Configure process-wide CPU thread caps exactly once. The GPU smoke test
    # calls this function before the pilot repeats its per-model RNG seeding.
    if not getattr(seed_everything, "_threads_initialized", False):
        torch.set_num_threads(int(cpu_threads))
        try:
            torch.set_num_interop_threads(min(4, int(cpu_threads)))
        except RuntimeError:
            pass  # Inter-op threads were initialized by the runtime already.
        seed_everything._threads_initialized = True
    torch.use_deterministic_algorithms(True)


def cuda_device():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("GPU1 isolation failed; expected exactly one visible CUDA GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal physical GPU index 1")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    device = torch.device("cuda:0")
    print(f"CUDA_VISIBLE_DEVICE=0; PHYSICAL_GPU_INDEX=1; GPU_NAME={torch.cuda.get_device_name(device)}", flush=True)
    return device


def load_split(proc_dir, split, *, w=10, d=0, labels=True):
    """Read only an explicit allowlisted split; do not enumerate PROC paths."""
    if split not in ALLOWED_SPLITS or (int(w), int(d)) != (10, 0):
        raise ValueError("T02c is locked to fit/calib/val, W=10, D=0")
    root = Path(proc_dir)
    x = np.load(root / f"{split}_W10_D0_x.npy", allow_pickle=False)
    mask = np.load(root / f"{split}_W10_D0_mask.npy", allow_pickle=False)
    if split == "calib" and labels:
        raise RuntimeError("Calibration labels must not enter T02c training")
    meta = pd.read_parquet(
        root / f"{split}_meta.parquet",
        columns=None if labels else ["idx", "record"])
    expected_cols = ["idx", "record", "event", "y", "q_mean", "q_adj",
                     "contested", "n_votes", "votes_missing"]
    if list(meta.columns) != (expected_cols if labels else ["idx", "record"]):
        raise RuntimeError(f"{split}: unexpected meta schema")
    n = EXPECTED_EVENTS[split]
    if len(meta) != n or x.shape != (n, 4, 2500) or mask.shape != (n, 4):
        raise RuntimeError(f"{split}: shape/count mismatch with frozen design")
    if len(set(meta["record"])) != EXPECTED_RECORDS[split]:
        raise RuntimeError(f"{split}: record count differs from frozen design")
    if (x.dtype != np.float32 or mask.dtype != np.bool_
            or not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 10.000001):
        raise RuntimeError(f"{split}: corrupted or incompatible PROC cache")
    if not meta["idx"].is_unique:
        raise RuntimeError(f"{split}: idx must be unique")
    if labels:
        y = np.asarray(meta["y"], dtype=np.float32)
        if not np.all((y == 0) | (y == 1)):
            raise RuntimeError(f"{split}: invalid binary training labels")
    # Fully materialize on CPU RAM. Batch copies to GPU happen during training.
    return {
        "x": torch.from_numpy(np.array(x, copy=True, order="C")),
        "mask": torch.from_numpy(np.array(mask, copy=True, order="C")),
        "y": torch.from_numpy(y.copy()) if labels else None,
        "idx": np.array(meta["idx"].to_numpy(), copy=True),
    }


def model_input(x_cpu, mask_cpu, row_ids, model_name, device):
    ids = torch.as_tensor(row_ids, dtype=torch.long)
    x = x_cpu.index_select(0, ids).to(device=device, dtype=torch.float32)
    if model_name == "fcn":
        return x  # Wang 2017 benchmark explicitly excludes channel masks.
    mask = mask_cpu.index_select(0, ids).to(
        device=device, dtype=torch.float32)
    return torch.cat((x, mask.unsqueeze(-1).expand(-1, -1, x.shape[-1])), 1)


def predict_logits(model, data, model_name, device, batch_size):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(data["idx"]), int(batch_size)):
            ids = range(start, min(len(data["idx"]), start + int(batch_size)))
            inputs = model_input(data["x"], data["mask"], ids,
                                 model_name, device)
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite logits in model evaluation")
            pieces.append(logits.detach().cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def _sigmoid(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -logits))


def _challenge_from_prediction(pred, y, challenge_fn):
    y = np.asarray(y, dtype=np.int8)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.count_nonzero(pred & (y == 1)))
    tn = int(np.count_nonzero(~pred & (y == 0)))
    fp = int(np.count_nonzero(pred & (y == 0)))
    fn = int(np.count_nonzero(~pred & (y == 1)))
    return float(challenge_fn(tp, tn, fp, fn))


def best_val_challenge_threshold(probabilities, y, challenge_fn):
    """Exhaustively maximize the val-only score; explicitly optimistic."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    if len(p) != len(y) or len(p) == 0 or not np.all(np.isfinite(p)):
        raise ValueError("Invalid validation probabilities")
    # A threshold just above the maximum predicts no positive events.
    best_thr = float(np.nextafter(np.max(p), np.inf))
    best_score = _challenge_from_prediction(np.zeros(len(y), dtype=bool),
                                            y, challenge_fn)
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    sorted_y = y[order]
    npos = int(np.sum(sorted_y))
    tp = fp = 0
    n = len(y)
    for i in range(n):
        if sorted_y[i]:
            tp += 1
        else:
            fp += 1
        if i + 1 < n and sorted_p[i + 1] == sorted_p[i]:
            continue  # Treat tied scores as one threshold change.
        fn = npos - tp
        tn = n - npos - fp
        score = float(challenge_fn(tp, tn, fp, fn))
        if score > best_score + 1e-12:
            best_score = score
            best_thr = float(sorted_p[i])
    return best_thr, best_score


def val_metrics(logits, val_data, common_metrics, ece_bins=15):
    y = val_data["y"].numpy().astype(np.int8)
    probs = _sigmoid(logits)
    challenge = common_metrics.challenge_score
    threshold, optimistic = best_val_challenge_threshold(
        probs, y, challenge)
    return {
        "auroc": float(common_metrics.auroc(logits, y)),
        "auprc": float(common_metrics.auprc(logits, y)),
        "ece": float(common_metrics.ece(probs, y, bins=int(ece_bins))),
        "score_threshold_0p5": _challenge_from_prediction(probs >= 0.5, y,
                                                            challenge),
        "score_val_optimized_optimistic": float(optimistic),
        "val_optimized_threshold": float(threshold),
    }


def script_hashes(root, yaml_path, entry_script):
    root = Path(root)
    return {
        "t02c_yaml": sha256_file(yaml_path),
        "models_py": sha256_file(root / "src" / "models.py"),
        "train_lib_py": sha256_file(root / "src" / "train_lib.py"),
        "t02c_pilot_py": sha256_file(entry_script),
    }


def _write_log(path, rows, cfg, script_sha, extra):
    metadata = {
        "used_config_json": json.dumps(cfg, sort_keys=True,
                                        separators=(",", ":")),
        "script_sha256_json": json.dumps(script_sha, sort_keys=True),
        "locked_design_sha256": cfg["expected_locked_sha256"],
        "common_tree_sha256": cfg["expected_common_tree_sha256"],
        "run_extra_json": json.dumps(extra, sort_keys=True),
    }
    fieldnames = ["epoch", "train_loss", "val_auprc", "epoch_seconds",
                  "peak_vram_gib"] + list(metadata)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **metadata})


def train_model(model_name, seed, fit_data, val_data, cfg, device,
                common_metrics, *, max_epochs=None, patience=None,
                output_dir=None, sha=None, yaml_path=None, entry_script=None,
                is_pilot=False):
    """Train one model; pilot never writes ckpt or sealed scores."""
    seed_everything(seed, cfg["cpu_threads"])
    model, param_count = make_model(model_name)
    model = model.to(device)
    train_cfg = cfg["model_specs"][model_name]
    batch_size = int(train_cfg["batch_size"])
    optimizer_class = (torch.optim.Adam if train_cfg["optimizer"] == "Adam"
                       else torch.optim.AdamW)
    optimizer = optimizer_class(model.parameters(), lr=float(train_cfg["lr"]),
                                weight_decay=float(train_cfg["weight_decay"]))
    epochs = int(max_epochs or cfg["max_epochs"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg["max_epochs"]))
        if train_cfg["cosine_decay"] else None)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(train_cfg["pos_weight"]),
                                device=device))
    rows = []
    best_auprc = -float("inf")
    best_epoch = 0
    best_state = None
    stop_count = 0
    wall_start = time.perf_counter()
    peak_vram_bytes = 0
    for epoch in range(1, epochs + 1):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        rng = np.random.default_rng(int(seed) * 1_000_003 + epoch)
        permutation = rng.permutation(len(fit_data["y"]))
        sum_loss = 0.0
        for offset in range(0, len(permutation), batch_size):
            ids = permutation[offset:offset + batch_size]
            x = model_input(fit_data["x"], fit_data["mask"], ids,
                            model_name, device)
            target = fit_data["y"].index_select(
                0, torch.as_tensor(ids, dtype=torch.long)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss; refusing to train further")
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.detach()) * len(ids)
        if scheduler is not None:
            scheduler.step()
        val_logits = predict_logits(model, val_data, model_name, device,
                                    batch_size)
        val_ap = float(common_metrics.auprc(
            val_logits, val_data["y"].numpy()))
        if val_ap > best_auprc + 1e-12:
            best_auprc = val_ap
            best_epoch = epoch
            best_state = {name: p.detach().cpu().clone()
                          for name, p in model.state_dict().items()}
            stop_count = 0
        else:
            stop_count += 1
        torch.cuda.synchronize(device)
        epoch_sec = time.perf_counter() - epoch_start
        peak_vram_bytes = max(peak_vram_bytes,
                              torch.cuda.max_memory_allocated(device))
        row = {
            "epoch": epoch,
            "train_loss": round(sum_loss / len(permutation), 9),
            "val_auprc": round(val_ap, 9),
            "epoch_seconds": round(epoch_sec, 4),
            "peak_vram_gib": round(peak_vram_bytes / 2**30, 4),
        }
        rows.append(row)
        print(f"EPOCH model={model_name} seed={seed} "
              f"epoch={epoch}/{epochs} seconds={epoch_sec:.2f} "
              f"vram_peak_GiB={peak_vram_bytes / 2**30:.3f} "
              f"val_AUPRC={val_ap:.6f}", flush=True)
        if not is_pilot and stop_count >= int(patience or cfg["patience"]):
            print(f"EARLY_STOP model={model_name} seed={seed} "
                  f"best_epoch={best_epoch}", flush=True)
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    best_val_logits = predict_logits(model, val_data, model_name, device,
                                     batch_size)
    metrics = val_metrics(best_val_logits, val_data, common_metrics,
                          cfg["ece_bins"])
    elapsed = time.perf_counter() - wall_start
    result = {
        "model": model_name, "seed": int(seed), "parameters": param_count,
        "epochs_run": len(rows), "best_epoch": best_epoch,
        "seconds_per_epoch_mean": float(np.mean([r["epoch_seconds"] for r in rows])),
        "seconds_per_epoch_sd": float(np.std([r["epoch_seconds"] for r in rows],
                                           ddof=1)) if len(rows) > 1 else 0.0,
        "peak_vram_gib": peak_vram_bytes / (2**30),
        "wall_seconds": elapsed, "metrics": metrics,
    }
    if not is_pilot:
        if output_dir is None or sha is None:
            raise ValueError("Full training requires output_dir and hashes")
        out = Path(output_dir)
        if out.exists():
            raise FileExistsError("Run output exists; refusing overwrite")
        out.mkdir(parents=True, exist_ok=False)
        # No calib labels are consulted for tuning; save only calib logits+idx.
        calib_data = load_split(cfg["proc_dir"], "calib", labels=False)
        calib_logits = predict_logits(model, calib_data, model_name, device,
                                      batch_size)
        torch.save({
            "state_dict": best_state, "model": model_name, "seed": int(seed),
            "best_epoch": best_epoch, "parameters": param_count,
            "config": cfg, "script_sha256": sha,
            "val_metrics": metrics, "torch_version": torch.__version__,
            "wall_seconds": elapsed,
            "cuda_device_name": torch.cuda.get_device_name(device),
        }, out / "ckpt.pt")
        pd.DataFrame({"idx": val_data["idx"],
                      "logit": best_val_logits}).to_parquet(
                          out / "scores_val.parquet", index=False)
        pd.DataFrame({"idx": calib_data["idx"],
                      "logit": calib_logits}).to_parquet(
                          out / "scores_calib.parquet", index=False)
        _write_log(out / "log.csv", rows, cfg, sha, result)
        print(f"RUN_SAVED={model_name}_s{seed}; "
              "files=ckpt.pt,scores_val.parquet,scores_calib.parquet,log.csv",
              flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def completed_run(path):
    run = Path(path)
    names = {"ckpt.pt", "scores_val.parquet", "scores_calib.parquet",
             "log.csv"}
    if not run.exists():
        return False
    present = {name for name in names if (run / name).is_file()}
    if present != names:
        raise RuntimeError(f"Incomplete run directory at {run.name}; "
                           "preserved for diagnosis, no overwrite")
    return True


def load_completed_result(path, expected_model, expected_seed, sha,
                          expected_locked_hash):
    """Read checkpoint metadata and ensure it matches the current source."""
    # weights_only was introduced after torch 2.0.1; only trusted local ckpts.
    ckpt = torch.load(Path(path) / "ckpt.pt", map_location="cpu")
    if (ckpt["model"] != expected_model or ckpt["seed"] != expected_seed
            or ckpt["script_sha256"] != sha
            or ckpt["config"]["expected_locked_sha256"] != expected_locked_hash):
        raise RuntimeError("Existing run does not match frozen configuration")
    log = pd.read_csv(Path(path) / "log.csv")
    if log.empty:
        raise RuntimeError("Existing run log is empty")
    return {
        "model": expected_model, "seed": expected_seed,
        "parameters": ckpt["parameters"],
        "epochs_run": len(log), "best_epoch": ckpt["best_epoch"],
        "seconds_per_epoch_mean": float(log["epoch_seconds"].mean()),
        "seconds_per_epoch_sd": float(log["epoch_seconds"].std(ddof=1))
        if len(log) > 1 else 0.0,
        "peak_vram_gib": float(log["peak_vram_gib"].max()),
        "wall_seconds": float(ckpt.get("wall_seconds",
                                          log["epoch_seconds"].sum())),
        "metrics": ckpt["val_metrics"],
    }
