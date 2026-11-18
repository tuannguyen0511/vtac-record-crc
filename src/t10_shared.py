"""P05-T10 POST-HOC shared helpers. All input paths read-only; no raw IDs in outputs.

This is a NEW post-hoc analysis, not part of the frozen confirmatory procedure.
Use only P05 original sources with SHA checks and the locked data/score artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# CPU caps and NO module-level torch import.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

EXPECTED_LOCK = "c09b6cd952319950f65b57e64086ab691946ccabac84adb106efb2bc765299c8"
ALPHAS = (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10)
CORE_ALPHAS = (0.01, 0.02, 0.05)
EXPECTED_COUNTS = {"fit":2830, "calib":1230, "val":495, "test":482, "c2015":341}
EXPECTED_LAMBDA_A = {0.01:7.61028e-5,0.02:0.0139079025,0.05:0.0314556314}


def must(flag, gate):
    ok = bool(flag)
    print("T10_" + gate + "=" + ("PASS" if ok else "FAIL"),flush=True)
    if not ok:
        raise RuntimeError("T10_HARD_STOP:" + gate)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def file_snapshot(paths):
    return {str(p):sha(p) for p in paths}


def forbid_overwrite_bytes(path, raw):
    p = Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        must(p.is_file() and p.read_bytes()==raw,"IMMUTABLE_OUTPUT_"+p.stem.upper())
        return p
    with p.open("xb") as f:f.write(raw)
    return p


def save_json(path, obj):
    return forbid_overwrite_bytes(path,(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False,ensure_ascii=True)+"\n").encode())


def save_csv(path, rows, fields):
    import io
    s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=fields,lineterminator="\n",extrasaction="raise")
    w.writeheader();w.writerows(rows)
    return forbid_overwrite_bytes(path,s.getvalue().encode())


def red(n, *, whole_fnr=False):
    x=int(n)
    return x if (x>=10 or whole_fnr) else "<10"


def safe_count_fraction(k,n,allow_small=False):
    # For whole-cohort FNR k/n, explicit Director exception.
    if n<=0:return "N/A"
    if not allow_small and (k<10 or n<10):return "<10"
    return f"{int(k)}/{int(n)}"


def sigmoid(z):
    import numpy as np
    z=np.asarray(z,dtype=np.float64)
    return np.exp(-np.logaddexp(0.0,-z))


def logit(p):
    import numpy as np
    p=np.asarray(p,dtype=np.float64)
    return np.log(p)-np.log1p(-p)


def record_fnr(p,y,rec,threshold,inclusive=False):
    import numpy as np
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);rec=np.asarray(rec,dtype=str)
    v=[]
    suppress=(p<=threshold if inclusive else p<threshold)
    for name in np.unique(rec[y==1]):
        mask=(rec==name)&(y==1)
        v.append(float(suppress[mask].mean()))
    return float(np.mean(v)) if v else None


def out_metrics(p,y,lam,metrics,scope="cohort",inclusive=False):
    import numpy as np
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=np.int8)
    must(len(p)==len(y) and np.isfinite(p).all() and np.isin(y,[0,1]).all(),"METRIC_INPUTS_VALID")
    pos=y==1;neg=y==0
    if not pos.any() or not neg.any():
        return {"status":"WITHHELD_EMPTY_CLASS"}
    cut=p<=lam if inclusive else p<lam
    sf=int((cut&neg).sum());sp=int((cut&pos).sum())
    nt=int(pos.sum());nf=int(neg.sum())
    low,high=metrics.clopper_pearson(sp,nt)
    return {"status":"POST-HOC" if scope in ("test","c2015") else "POST-HOC_EXPLORATORY",
            "auroc":float(metrics.auroc(p,y)),"suppression":float(sf/nf),
            "suppressed_false":safe_count_fraction(sf,nf),
            "false_alarm_n":red(nf),
            "fnr":float(sp/nt),"fnr_count":safe_count_fraction(sp,nt,allow_small=(scope in ("test","c2015","val","calib"))),
            "fnr_ci_low":float(low),"fnr_ci_high":float(high)}


def set_priority():
    # Import original audited function without importing torch.
    import t02c_pilot
    t02c_pilot.below_normal_priority()


class Context:
    def __init__(self, project):
        import numpy as np
        import pandas as pd
        import yaml
        self.np=np;self.pd=pd
        self.project=Path(project).resolve()
        self.workspace=self.project.parent
        self.proc=self.workspace/"data_proc"/"vtac"/"1.1"/"P05_v1"
        self.raw=self.workspace/"data_raw"/"challenge-2015"/"1.0.0"
        self.out=self.project/"outputs"/"t10"
        self.report=self.project/"reports"/"P05-T10.md"
        self.repo=self.project/"release"/"vtac-record-crc"
        self.assets=self.project/"release"/"release_assets"
        must(self.project.name=="P05_VTaC_FalseAlarm" and self.workspace.is_dir(),"WORKSPACE")
        must((self.project/"reports"/"TEST_OPENED.flag").is_file(),"TEST_FLAG_PRESENT")
        self.flag_sha=sha(self.project/"reports"/"TEST_OPENED.flag")
        self.lock_path=self.project/"configs"/"final_lock.yaml"
        must(self.lock_path.is_file() and sha(self.lock_path)==EXPECTED_LOCK,"FROZEN_LOCK_SHA")
        self.lock=yaml.safe_load(self.lock_path.read_text(encoding="utf-8"))
        must(self.lock["primary"]["alpha"]==.02 and self.lock["primary"]["variant"]=="A","DIRECTOR_PRIMARY")
        must(self.lock["model"]["backbone"]=="resnet1d","FROZEN_BACKBONE")
        for a,exp in EXPECTED_LAMBDA_A.items():
            must(abs(float(self.lock["lambda"]["A"][a])-exp)<1e-12,"LAMBDA_A_"+str(a).replace(".",""))
        must(self.proc.is_dir(),"FROZEN_DATA_PROC_PRESENT")
        must(sha(self.proc/"test_labels.parquet")==self.lock["frozen_sha256"]["test_labels_parquet"],"FROZEN_TEST_LABELS_SHA")
        for key,f in [("models_py","models.py"),("train_lib_py","train_lib.py"),
                       ("t03_label_variants_py","t03_label_variants.py"),
                       ("t05a_conformal_ablation_py","t05a_conformal_ablation.py")]:
            must(sha(self.project/"src"/f)==self.lock["frozen_sha256"][key],"FROZEN_SRC_"+key.upper())
        # Never import a modified common package.
        sys.path.insert(0,str(self.workspace))
        from common import conformal,metrics
        from t02c_pilot import common_source_sha256
        must(common_source_sha256(self.workspace/"common")==self.lock["frozen_sha256"]["common_tree"],"FROZEN_COMMON_SHA")
        self.conformal=conformal;self.metrics=metrics

    def check_unchanged(self):
        must(sha(self.lock_path)==EXPECTED_LOCK,"FINAL_LOCK_UNCHANGED")
        must(sha(self.project/"reports"/"TEST_OPENED.flag")==self.flag_sha,"TEST_FLAG_UNCHANGED")

    def meta(self,split):
        p=self.proc/f"{split}_meta.parquet"
        m=self.pd.read_parquet(p)
        must(len(m)==EXPECTED_COUNTS[split] and m["idx"].is_unique,"META_"+split.upper())
        return m

    def labels(self,split):
        np=self.np
        if split=="test":
            m=self.meta("test")
            labels=self.pd.read_parquet(self.proc/"test_labels.parquet",columns=["idx","y"])
            must(np.array_equal(m["idx"],labels["idx"]),"TEST_LABEL_ORDER")
            y=labels["y"].to_numpy(dtype=np.int8)
            rec=m["record"].to_numpy(dtype=str)
            return m,y,rec
        m=self.meta(split)
        return m,m["y"].to_numpy(dtype=np.int8),m["record"].to_numpy(dtype=str)

    def vt(self):
        folder=self.raw/"training" if (self.raw/"training"/"ALARMS").exists() else self.raw
        must((folder/"ALARMS").is_file(),"C2015_ALARMS_PRESENT")
        manifest=self.project/"outputs"/"t06"/"step0_manifest.json"
        expected=json.loads(manifest.read_text(encoding="utf-8"))
        must(sha(folder/"ALARMS")==expected["c2015_alarm_labels_sha256"] and
             sha(folder/"SHA256SUMS")==expected["c2015_training_manifest_sha256"],"C2015_FROZEN_SOURCE_SHA")
        with (folder/"ALARMS").open("r",encoding="utf-8",newline="") as f:
            rows=list(csv.reader(f))
        must(len(rows)==750,"C2015_ALARMS_750")
        vt=[(r[0],int(r[2])) for r in rows if len(r)==3 and r[1]=="Ventricular_Tachycardia"]
        must(len(vt)==341 and sum(v[1] for v in vt)==89,"C2015_VT_341_89")
        return folder,vt

    def scores(self,split,variant="A"):
        import t03_label_variants as t03
        np=self.np
        m=self.meta(split)
        ix=m["idx"].to_numpy()
        stacks=[t03.align_scores(self.project/"outputs"/"t03"/f"{variant}_s{s}"/f"scores_{split}.parquet",
                                ix,self.pd,np,"T10") for s in range(5)]
        return np.mean(np.stack(stacks),axis=0)

    def t06_scores(self,split):
        import numpy as np
        manifest=self.project/"outputs"/"t06"/"step0_manifest.json"
        d=json.loads(manifest.read_text(encoding="utf-8"))
        must(d["status"]=="PASS" and d["source_lock_sha256"]==EXPECTED_LOCK,"T06_MANIFEST_LOCK")
        name=f"scores_{split}.parquet"
        f=manifest.parent/name
        must(sha(f)==d["files"][name],"T06_SCORES_SHA_"+split.upper())
        frame=self.pd.read_parquet(f,columns=["idx","seed","logit"])
        n=EXPECTED_COUNTS[split]
        must(len(frame)==6*n,"T06_SCORE_COUNT_"+split.upper())
        zs=[]
        for seed in ["0","1","2","3","4","ensemble"]:
            arr=frame[frame["seed"].astype(str)==seed]
            must(len(arr)==n and np.array_equal(arr["idx"].to_numpy(),np.arange(n)),"T06_SCORE_ALIGN_"+split.upper()+"_"+seed)
            zs.append(arr["logit"].to_numpy(dtype=np.float64))
        must(np.allclose(np.mean(zs[:5],axis=0),zs[5],atol=1e-12,rtol=0),"T06_ENSEMBLE_"+split.upper())
        return zs[5]

    def cohort(self,split,variant="A"):
        if split=="c2015":
            folder,vt=self.vt()
            z=self.t06_scores("c2015") if variant=="A" else None
            must(z is not None,"ONLY_A_EXTERNAL_SCORES")
            return {"p":sigmoid(z),"z":z,"y":self.np.array([v[1] for v in vt],dtype=self.np.int8),
                    "rec":self.np.array([v[0] for v in vt],dtype=str),"name":"c2015"}
        m,y,rec=self.labels(split)
        z=self.t06_scores("test") if split=="test" else self.scores(split,variant)
        must(len(z)==len(y),"COHORT_ALIGN_"+split.upper())
        return {"p":sigmoid(z),"z":z,"y":y,"rec":rec,"meta":m,"name":split}

    def assert_locked_A(self):
        c=self.cohort("calib")
        y=c["y"];r=c["rec"];p=c["p"]
        must(len(y)==1230 and len(set(r[y==1]))==188,"CALIB_1230_POSREC_188")
        for a,target in EXPECTED_LAMBDA_A.items():
            actual=float(self.conformal.crc_fnr_threshold(p,y,r,a))
            must(abs(actual-target)<=5e-9,"CRC_REPRODUCE_"+str(a).replace(".",""))
        v=self.cohort("val");t=self.cohort("test");e=self.cohort("c2015")
        must(len(v["y"])==495 and len(t["y"])==482 and len(e["y"])==341,"COHORT_COUNTS")
        must(int(t["y"].sum())==137 and int(e["y"].sum())==89,"COHORT_POSITIVES")
        must(abs(self.metrics.auroc(t["p"],t["y"])-.946049)<1e-6,"FROZEN_TEST_AUROC")
        must(abs(self.metrics.auroc(e["p"],e["y"])-.942304)<1e-6,"FROZEN_EXTERNAL_AUROC")
        return c,v,t,e


def safe_exception(exc):
    # Do not print raw exception messages, which may include private event IDs or paths.
    print("T10_FAILURE_TYPE="+type(exc).__name__,flush=True)
    print("T10_FAILURE_CATEGORY="+str(exc).split(":",1)[0][:55],flush=True)
