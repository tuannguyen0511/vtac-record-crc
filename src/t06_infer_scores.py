"""P05-T06 step 0: read-only GPU1 inference of five frozen A checkpoints.

This is NEW inference, never a T05b rerun. Rejects any mismatch against the
Director's previously accepted AUROCs before creating T06 output files.
"""
import release_paths
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "8"

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def require(condition, message):
    if not condition:
        raise RuntimeError("P05_T06_STEP0_STOP: " + message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def sig(z, np):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))


def stage0():
    start = time.monotonic()
    project = Path(__file__).resolve().parents[1]
    require((project / "reports" / "TEST_OPENED.flag").is_file(),
            "T05b test is not officially opened; STOP")
    require((project / "reports" / "PREREG_LOCK.md").is_file() and
            (project / "reports" / "P05-T05b.md").is_file(),
            "accepted T05b preregistration or report missing")
    with (project / "configs" / "t06.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    require(cfg["step"] == "P05-T06" and cfg["authorization"] == "T06_AUTHORIZED"
            and cfg["variant"] == "A" and cfg["seeds"] == [0,1,2,3,4]
            and cfg["no_training"] is True and cfg["physical_gpu"] == 1
            and cfg["cpu_threads"] <= 8, "T06 protocol not locked")
    import t02c_pilot as frozen  # No torch import; priority before hashes/data work.
    frozen.below_normal_priority()
    out = project / cfg["outputs_subdir"]
    require(not out.exists(), "T06 outputs already exist; preserve previous run")
    lockpath = project / "configs" / "final_lock.yaml"
    require(sha(lockpath) == cfg["director_final_lock_sha256"], "final lock SHA mismatch")
    with lockpath.open("r", encoding="utf-8") as f:
        lock = yaml.safe_load(f)
    require(lock["primary"]["variant"] == "A" and lock["model"]["backbone"] == "resnet1d",
            "wrong locked backbone/variant")
    src = project / "src"
    freeze = {
        "models_py": src / "models.py",
        "train_lib_py": src / "train_lib.py",
        "test_labels_parquet": release_paths.get("data_proc") / "test_labels.parquet",
    }
    for name, path in freeze.items():
        require(sha(path) == lock["frozen_sha256"][name], "frozen SHA mismatch: " + name)
    for name, path in (
            ("t03_label_variants_py",src / "t03_label_variants.py"),
            ("t05a_conformal_ablation_py",src / "t05a_conformal_ablation.py")):
        require(sha(path)==lock["frozen_sha256"][name],
                "frozen source SHA mismatch: " + name)
    require(sha(src / "t05b_final_test.py") ==
            "f98c932267be52775282dc8c502dda0cd4e6d9ac1eb2f287a2a9f0f7a2de9057",
            "T05b source changed")
    require(sha(src / "t05b_external_c2015.py") ==
            "20f53b605d795a4ce7f6ca509e083653628aed37a468efdeacabad61501447ab",
            "T05b external source changed")
    require(len(lock["checkpoints_sha256"])==25, "Director checkpoint registry incomplete")
    for name, expected in lock["checkpoints_sha256"].items():
        require(sha(project / "outputs" / "t03" / name / "ckpt.pt") ==
                expected, "checkpoint SHA mismatch: " + name)

    import numpy as np
    import pandas as pd
    from scipy import signal
    import wfdb
    old = frozen.validate_frozen(project / "configs" / "t02c.yaml")
    require(frozen.common_source_sha256(old["common_dir"]) ==
            lock["frozen_sha256"]["common_tree"], "common source SHA mismatch")
    with Path(old["locked_design"]).open("r", encoding="utf-8") as f:
        original = yaml.safe_load(f)
    preproc_path = project / "src" / "preprocess_lib.py"
    require(sha(preproc_path) == original["provenance_sha256"]["preprocess_lib_script"],
            "frozen D preprocessing SHA mismatch")
    sys.path.insert(0, str(Path(old["common_dir"]).resolve().parent))
    from common import metrics
    import preprocess_lib as prep
    import t05b_external_c2015 as ext
    import torch
    import models
    import train_lib
    torch.set_num_threads(8)
    device = train_lib.cuda_device()
    require(torch.cuda.device_count() == 1 and
            "Quadro M6000" in torch.cuda.get_device_name(0), "GPU1 not Quadro M6000")

    proc = Path(old["proc_dir"])
    test_meta = pd.read_parquet(proc / "test_meta.parquet", columns=["idx", "record", "event"])
    label = pd.read_parquet(proc / "test_labels.parquet", columns=["idx", "y"])
    require(len(label) == len(test_meta) == cfg["expected_count"]["test"] and
            np.array_equal(label["idx"].to_numpy(),test_meta["idx"].to_numpy()),
            "official test label alignment/count mismatch")
    tx = np.load(proc / "test_W10_D0_x.npy", allow_pickle=False)
    tm = np.load(proc / "test_W10_D0_mask.npy", allow_pickle=False)
    require(tx.shape == (len(label),4,2500) and tm.shape == (len(label),4)
            and tx.dtype == np.float32 and tm.dtype == np.bool_, "test signal/mask shape mismatch")
    test = {"x":torch.from_numpy(np.array(tx,copy=True)),
            "mask":torch.from_numpy(np.array(tm,copy=True)),
            "idx":test_meta["idx"].to_numpy(),"y":None}

    folder = ext.external_root(old)
    vt = ext.audit_training(folder)
    require(len(vt) == cfg["expected_count"]["c2015_vt"], "external VT count != 341")
    payload = ext.prepare_public_vt(old,vt,folder,prep,np,wfdb,signal)
    c2015 = {"x":torch.from_numpy(np.array(payload["x"],dtype=np.float32,copy=True)),
             "mask":torch.from_numpy(np.array(payload["mask"],dtype=np.bool_,copy=True)),
             "idx":np.arange(len(vt),dtype=np.int64),"y":None}
    logits = {"test":[],"c2015":[]}
    for seed in cfg["seeds"]:
        cpfile = project / "outputs" / "t03" / f"A_s{seed}" / "ckpt.pt"
        ckpt = torch.load(cpfile,map_location="cpu",weights_only=False)
        require(ckpt["model"] == "resnet1d" and int(ckpt["seed"]) == seed,
                "checkpoint identity mismatch")
        model, n_params = models.make_model("resnet1d")
        require(n_params == ckpt["parameters"], "parameter count mismatch")
        model.load_state_dict(ckpt["state_dict"],strict=True)
        model.to(device)
        for name, data in (("test",test),("c2015",c2015)):
            z = train_lib.predict_logits(model,data,"resnet1d",device,128)
            require(np.isfinite(z).all(), "nonfinite inference: " + name)
            logits[name].append(z)
        print(f"T06_A_CKPT_INFERENCE=PASS; seed={seed}",flush=True)
        del model,ckpt
        torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    frozen.verify_nvidia_process_is_gpu1()

    aucs = {}
    mean_z = {}
    for name, y in (("test",label["y"].to_numpy(dtype=np.int8)),
                    ("c2015",np.asarray([v[1] for v in vt],dtype=np.int8))):
        mean_z[name] = np.mean(np.stack(logits[name]),axis=0)
        auc = float(metrics.auroc(sig(mean_z[name],np),y))
        target = float(cfg["expected_auroc"][name])
        diff = abs(auc-target)
        print(f"T06_AUROC_{name.upper()}={auc:.9f}; expected={target:.6f}; "
              f"diff={diff:.9g}; status={'PASS' if diff <= cfg['auroc_absolute_tolerance'] else 'FAIL'}",
              flush=True)
        require(diff <= cfg["auroc_absolute_tolerance"], "AUROC gate failed: " + name)
        aucs[name] = auc
    # ALL hard checks above must pass before ANY output is written.
    out.mkdir(parents=True,exist_ok=False)
    to_manifest = []
    for name, mask in (("test",tm),("c2015",payload["mask"])):
        n = len(mask)
        order = np.tile(np.arange(n,dtype=np.int64),6)
        seed_names = np.repeat([str(s) for s in cfg["seeds"]]+["ensemble"],n)
        zs = np.concatenate(logits[name]+[mean_z[name]])
        dest = out / f"scores_{name}.parquet"
        pd.DataFrame({"idx":order,"seed":seed_names,"logit":zs}).to_parquet(dest,index=False)
        to_manifest.append((dest,6*n))
        a,b = mask[:,2].astype(bool),mask[:,3].astype(bool)
        strata = np.where(a & b,"S2",np.where(a|b,"S1","S0"))
        mp = out / f"modality_{name}.parquet"
        pd.DataFrame({"idx":np.arange(n),"ppg":a,"abp":b,"stratum":strata}).to_parquet(mp,index=False)
        to_manifest.append((mp,n))
    manifest_file = out / "step0_manifest.json"
    record = {"status":"PASS","source_lock_sha256":sha(lockpath),"script_sha256":sha(__file__),
              "auroc":aucs,"n":{"test":len(label),"c2015":len(vt)},
              "c2015_training_manifest_sha256":sha(folder/"SHA256SUMS"),
              "c2015_alarm_labels_sha256":sha(folder/"ALARMS"),
              "files":{p.name:sha(p) for p,_ in to_manifest}}
    with manifest_file.open("x",encoding="utf-8") as f:
        json.dump(record,f,indent=2)
    to_manifest.append((manifest_file,1))
    with (project / "manifest.csv").open("a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        now=datetime.now(timezone.utc).isoformat()
        for path,n in to_manifest:
            w.writerow([str(path),sha(path),n,now,"P05-T06-STEP0"])
    print("=== REPORT P05-T06-STEP0 ===",flush=True)
    print("status: PASS; pc: reference-workstation; gpu: physical GPU1; training: NONE",flush=True)
    print(f"runtime_s: {time.monotonic()-start:.3f}; script_sha256_8: {sha(__file__)[:8]}",flush=True)
    print(f"AUROC_TEST={aucs['test']:.9f}; expected=0.946049; PASS",flush=True)
    print(f"AUROC_C2015={aucs['c2015']:.9f}; expected=0.942304; PASS",flush=True)
    print("outputs: outputs/t06/scores_test.parquet; scores_c2015.parquet; modality files",flush=True)
    print("warnings: TEST and C2015 are POST-HOC; T05b frozen/unchanged",flush=True)
    print("=== END ===",flush=True)


if __name__ == "__main__":
    try:
        stage0()
    except BaseException as exc:
        print(f"P05_T06_STEP0_STOP={type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
