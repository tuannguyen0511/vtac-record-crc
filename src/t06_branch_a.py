"""P05-T06 CPU-only frozen post-hoc branch A (A1-A5).

No model/checkpoint inference, no torch import, no training, no changes to any
T02c-T05b files or TEST_OPENED.flag. All test/C2015 results are POST-HOC.
"""
import release_paths
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "8"

import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta


def hard(flag, reason):
    if not flag:
        raise RuntimeError("P05_T06_BRANCH_A_STOP: " + str(reason))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1048576), b""):
            hasher.update(piece)
    return hasher.hexdigest()


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=np.float64)))


def pct(values, probabilities=(2.5, 50, 97.5)):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)],dtype=float)
    return [float(x) for x in np.percentile(a, probabilities)] if a.size else [None]*len(probabilities)


def redact(n):
    return "<10" if int(n) < 10 else str(int(n))


def protected_stratum_counts(raw):
    """Mask complementary cells so all/other groups cannot reveal a small cell."""
    if any(int(raw[s])<10 for s in ("S0","S1","S2")):
        return {s:("<10" if int(raw[s])<10 else "WITHHELD")
                for s in ("S0","S1","S2")}
    return {s:str(int(raw[s])) for s in ("S0","S1","S2")}


def groupings(groups):
    _, codes = np.unique(np.asarray(groups,dtype=str),return_inverse=True)
    return [np.flatnonzero(codes==i) for i in range(int(codes.max())+1)]


def rates(p, y, lam, inclusive=False):
    suppress = p <= lam if inclusive else p < lam
    neg, pos = y==0, y==1
    return (float(suppress[neg].mean()) if neg.any() else None,
            float(suppress[pos].mean()) if pos.any() else None)


def format_rate(x):
    return "N/A" if x is None else f"{x:.6f}"


def fmt_ci(x):
    return " / ".join("N/A" if v is None else f"{v:.6f}" for v in x)


def stratum_from_mask(mask):
    ppg, abp = mask[:,2].astype(bool),mask[:,3].astype(bool)
    return np.where(ppg & abp,"S2",np.where(ppg | abp,"S1","S0"))


def read_score_file(path, n):
    df = pd.read_parquet(path)
    hard(list(df.columns)==["idx","seed","logit"] and len(df)==6*n,
         "step0 score schema/count invalid: " + path.name)
    zs = {}
    for seed in ["0","1","2","3","4","ensemble"]:
        rows = df.loc[df["seed"].astype(str)==seed]
        hard(len(rows)==n and np.array_equal(rows["idx"].to_numpy(),np.arange(n)),
             "step0 score index ordering invalid")
        zs[seed] = rows["logit"].to_numpy(dtype=float)
        hard(np.isfinite(zs[seed]).all(), "nonfinite step0 scores")
    hard(np.allclose(np.mean([zs[str(i)] for i in range(5)],axis=0),
                        zs["ensemble"],rtol=0,atol=1e-12), "ensemble logit differs")
    return {key:sigmoid(z) for key,z in zs.items()}


def source_data(project, cfg, lock, conformal, metrics):
    out = project / cfg["outputs_subdir"]
    manifest_path = out / "step0_manifest.json"
    hard(manifest_path.is_file() and (project / "reports" / "TEST_OPENED.flag").is_file(),
         "T05b opened flag or T06 step0 manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hard(manifest["status"] == "PASS" and
         manifest["source_lock_sha256"] == cfg["director_final_lock_sha256"],
         "T06 step0 did not pass Director AUROC gates")
    for name, expected in manifest["files"].items():
        hard(sha(out/name)==expected,"step0 frozen artifact changed: "+name)
    proc = release_paths.get("data_proc")
    hard(sha(proc/"test_labels.parquet")==lock["frozen_sha256"]["test_labels_parquet"],
         "test label SHA changed")
    data={}
    for split in ("calib","val"):
        meta=pd.read_parquet(proc/f"{split}_meta.parquet")
        need=("idx","record","event","y","votes_missing")
        hard(all(x in meta for x in need),"missing metadata columns")
        mask=np.load(proc/f"{split}_W10_D0_mask.npy",allow_pickle=False)
        hard(mask.shape == (len(meta),4),"mask shape mismatch: "+split)
        ix=meta["idx"].to_numpy()
        stacks=[]
        for seed in range(5):
            scores=pd.read_parquet(project/"outputs"/"t03"/f"A_s{seed}"/f"scores_{split}.parquet")
            hard(list(scores.columns)==["idx","logit"] and len(scores)==len(meta)
                 and scores["idx"].is_unique and
                 pd.Index(ix).isin(scores["idx"]).all(),"T03 scores index mismatch")
            aligned=scores.set_index("idx")["logit"].reindex(pd.Index(ix)).to_numpy(dtype=float)
            hard(np.isfinite(aligned).all(),"T03 scores nonfinite")
            stacks.append(aligned)
        logits=np.stack(stacks)
        data[split]={"meta":meta,"y":meta["y"].to_numpy(dtype=np.int8),
                     "records":meta["record"].to_numpy(dtype=str),
                     "strata":stratum_from_mask(mask),
                     "prob":sigmoid(logits.mean(axis=0)),
                     "seeds":[sigmoid(row) for row in logits]}
    test_meta=pd.read_parquet(proc/"test_meta.parquet")
    test_labels=pd.read_parquet(proc/"test_labels.parquet")
    hard(len(test_meta)==len(test_labels)==482 and
         np.array_equal(test_meta["idx"].to_numpy(),test_labels["idx"].to_numpy()),
         "test label alignment mismatch")
    public_root = release_paths.get("data_raw")/"challenge-2015"/"1.0.0"
    folder=public_root/"training" if (public_root/"training"/"ALARMS").is_file() else public_root
    hard(sha(folder/"SHA256SUMS")==manifest["c2015_training_manifest_sha256"]
         and sha(folder/"ALARMS")==manifest["c2015_alarm_labels_sha256"],
         "external source changed since checksum-audited T06 step0")
    with (folder/"ALARMS").open("r",encoding="utf-8",newline="") as f:
        all_rows=list(csv.reader(f))
    vt=[row for row in all_rows if len(row)==3 and row[1]=="Ventricular_Tachycardia"]
    hard(len(all_rows)==750 and len(vt)==341,"C2015 class labels/count changed")
    for name, meta, y, rec in (
            ("test",test_meta,test_labels["y"].to_numpy(dtype=np.int8),
             test_meta["record"].to_numpy(dtype=str)),
            ("c2015",None,np.asarray([int(row[2]) for row in vt],dtype=np.int8),
             np.asarray([row[0] for row in vt],dtype=str))):
        probs=read_score_file(out/f"scores_{name}.parquet",len(y))
        modality=pd.read_parquet(out/f"modality_{name}.parquet")
        hard(np.array_equal(modality["idx"].to_numpy(),np.arange(len(y)))
             and set(modality["stratum"]).issubset({"S0","S1","S2"}),
             "modality file invalid")
        auroc=metrics.auroc(probs["ensemble"],y)
        hard(abs(auroc-float(cfg["expected_auroc"][name]))<=1e-6,
             "post-hoc scores changed from T06 step0")
        data[name]={"y":y,"records":rec,"strata":modality["stratum"].to_numpy(),
                    "prob":probs["ensemble"],"seeds":[probs[str(i)] for i in range(5)]}
    return data


def a1_theory(cfg):
    rows=[]
    for n in cfg["a1"]["theoretical_positive_record_counts"]:
        for a in cfg["a1"]["alphas"]:
            # CRC for binary record-level miss L_j: (number_missed+1)/(n+1)<=alpha.
            # Strict suppression permits a threshold equal to the k-th record minimum.
            k=int(math.floor(float(a)*(n+1)+1e-12))
            if k<1:
                rows.append({"n_pos":n,"alpha":a,"kstar":None,"misses_allowed":None,
                             "beta_q05":None,"beta_median":None,"beta_q95":None,"status":"INFEASIBLE"})
            else:
                q=beta.ppf([.05,.5,.95],k,n+1-k)
                rows.append({"n_pos":n,"alpha":a,"kstar":k,"misses_allowed":k-1,
                             "beta_q05":float(q[0]),"beta_median":float(q[1]),
                             "beta_q95":float(q[2]),"status":"THEORY_BINARY_RECORD"})
    return rows


def all_record_resamples(calib, reps, seed):
    """Exactly the locked T05a bootstrap order, including duplicate record IDs."""
    members=groupings(calib["records"])
    rng=np.random.default_rng(seed)
    for iteration in range(reps):
        chosen=rng.integers(0,len(members),size=len(members))
        ids=np.concatenate([members[i] for i in chosen])
        groups=np.repeat(np.arange(len(chosen)),[len(members[i]) for i in chosen])
        yield iteration,ids,groups


def a1_resample(cfg,data,conformal):
    calib,val=data["calib"],data["val"]
    group=calib["records"]; y=calib["y"]; p=calib["prob"]
    members=groupings(group)
    positive=np.array([i for i,loc in enumerate(members) if (y[loc]==1).any()],dtype=int)
    negative=np.array([i for i,loc in enumerate(members) if not (y[loc]==1).any()],dtype=int)
    hard(len(positive)==188,"expected 188 positive calibration records")
    rows=[]
    rng=np.random.default_rng(cfg["a1"]["seed"])
    for npos in [50,100,150]:
        nneg=int(round(npos*len(negative)/len(positive)))
        for rep in range(cfg["a1"]["subsample_replicates"]):
            choice=np.r_[rng.choice(positive,npos,replace=False),
                         rng.choice(negative,nneg,replace=False)]
            ids=np.concatenate([members[i] for i in choice])
            selected_groups=group[ids]
            lam=conformal.crc_fnr_threshold(p[ids],y[ids],selected_groups,.02)
            suppression,fnr=rates(val["prob"],val["y"],lam)
            rows.append({"n_pos_nominal":npos,"model":"ensemble","rep":rep,
                         "lambda":lam,"suppression":suppression,"fnr":fnr})
    # At n=188, T05a resamples ALL 543 calibration records with replacement,
    # not exactly 188 positive records per draw. This preserves its benchmark.
    anchor=[]
    for rep,ids,groups in all_record_resamples(calib,
            cfg["a1"]["t05a_reference_bootstrap_replicates"],42):
        lam=conformal.crc_fnr_threshold(p[ids],y[ids],groups,.02)
        anchor.append(lam)
        if rep<1000:
            suppression,fnr=rates(val["prob"],val["y"],lam)
            rows.append({"n_pos_nominal":188,"model":"ensemble","rep":rep,
                         "lambda":lam,"suppression":suppression,"fnr":fnr})
    for seed in range(5):
        for rep,ids,groups in all_record_resamples(calib,1000,42):
            lam=conformal.crc_fnr_threshold(calib["seeds"][seed][ids],y[ids],groups,.02)
            suppression,fnr=rates(val["seeds"][seed],val["y"],lam)
            rows.append({"n_pos_nominal":188,"model":f"seed{seed}","rep":rep,
                         "lambda":lam,"suppression":suppression,"fnr":fnr})
    summary=[]
    df=pd.DataFrame(rows)
    for (n,model),frame in df.groupby(["n_pos_nominal","model"],sort=True):
        summary.append({"n_pos":int(n),"model":model,"replicates":len(frame),
                        "lambda_q025_q50_q975":pct(frame["lambda"]),
                        "supp_q025_q50_q975":pct(frame["suppression"]),
                        "fnr_q025_q50_q975":pct(frame["fnr"]),
                        "p_supp_below_030":float((frame["suppression"]<.30).mean())})
    t05a=pct(anchor)
    expected=np.asarray(cfg["a1"]["t05a_anchor_lambda_percentiles"],dtype=float)
    tolerance=np.asarray(cfg["a1"]["t05a_anchor_absolute_tolerance"],dtype=float)
    diff=abs(np.asarray(t05a)-expected)
    hard(bool(np.all(diff<=tolerance)),
         "[KH] T05a 2000-bootstrap locked lambda quantiles differ from Director anchor")
    reference={"recomputed_t05a_2000":t05a,"director_rounded_anchor":expected.tolist(),
               "absolute_difference":diff.tolist(),"absolute_tolerance":tolerance.tolist(),
               "status":"PASS",
               "first_1000_percentiles":pct(df.loc[(df.n_pos_nominal==188)&(df.model=="ensemble"),"lambda"]),
               "note":"Exact T05a full-record resampling; nominal 188 positive records, actual may vary"}
    return rows,summary,reference


def pac_threshold(scores,y,records,alpha,delta):
    members=groupings(records)
    minima=np.array([float(np.min(scores[loc][y[loc]==1]))
                     for loc in members if (y[loc]==1).any()],dtype=float)
    n=len(minima)
    if n==0 or 1-delta**(1/n)>alpha:
        return 0.0,n,"INFEASIBLE"
    best=0.0
    for cut in np.unique(minima):
        lost=int(np.count_nonzero(minima<cut))
        upper=1.0 if lost==n else float(beta.ppf(1-delta,lost+1,n-lost))
        if upper<=alpha and cut>best:
            best=float(cut)
    return best,n,"OK"


def a1_pac(cfg,data):
    rows=[]
    for alpha in cfg["a1"]["pac_alphas"]:
        delta=cfg["a1"]["pac_delta"]
        lam,n,status=pac_threshold(data["calib"]["prob"],data["calib"]["y"],
                                   data["calib"]["records"],alpha,delta)
        nmin=math.ceil(math.log(delta)/math.log(1-alpha))
        for cohort in ("val","test","c2015"):
            suppression,fnr=rates(data[cohort]["prob"],data[cohort]["y"],lam)
            rows.append({"alpha":alpha,"delta":delta,"n_positive_records":redact(n),
                         "n_minimum":nmin,"lambda_pac":lam,"status":status,
                         "cohort":cohort,"evaluation":"VALIDATION" if cohort=="val" else "POST-HOC",
                         "suppression":suppression,"fnr":fnr})
    return rows


def a2_mondrian(cfg,data,lock,conformal):
    alpha=.02
    calib=data["calib"]
    marginal=float(lock["lambda"]["A"][alpha])
    thresholds={}
    raw_counts={}
    for s in ("S0","S1","S2"):
        mask=calib["strata"]==s
        cp,cy,cg=calib["prob"][mask],calib["y"][mask],calib["records"][mask]
        n=len(set(cg[cy==1]))
        raw_counts[s]=n
        if n==0 or 1/(n+1)>alpha:
            thresholds[s]=0.0
        else:
            thresholds[s]=float(conformal.crc_fnr_threshold(cp,cy,cg,alpha))
    counts=protected_stratum_counts(raw_counts)
    rows=[]
    for cohort in ("val","test","c2015"):
        d=data[cohort]
        positives_by_stratum={s:int(np.sum((d["strata"]==s)&(d["y"]==1)))
                              for s in ("S0","S1","S2")}
        negatives_by_stratum={s:int(np.sum((d["strata"]==s)&(d["y"]==0)))
                              for s in ("S0","S1","S2")}
        protected_positive=protected_stratum_counts(positives_by_stratum)
        protected_negative=protected_stratum_counts(negatives_by_stratum)
        for method in ("marginal","mondrian"):
            cuts=np.full(len(d["y"]),marginal) if method=="marginal" else np.asarray(
                [thresholds[str(s)] for s in d["strata"]])
            suppressed=d["prob"]<cuts
            for stratum in ("ALL","S0","S1","S2"):
                keep=np.ones(len(d["y"]),dtype=bool) if stratum=="ALL" else d["strata"]==stratum
                negatives=keep & (d["y"]==0)
                positives=keep & (d["y"]==1)
                group_small=(stratum!="ALL" and
                             (positives_by_stratum[stratum]<10 or negatives_by_stratum[stratum]<10))
                rows.append({"cohort":cohort,"evaluation":"VALIDATION" if cohort=="val" else "POST-HOC",
                             "method":method,"stratum":stratum,"alpha":alpha,
                             "calib_positive_records":counts.get(stratum,"N/A"),
                             "lambda":marginal if method=="marginal" else thresholds.get(stratum,None),
                             "n_true":redact(int(positives.sum())) if stratum=="ALL" else protected_positive[stratum],
                             "n_false":redact(int(negatives.sum())) if stratum=="ALL" else protected_negative[stratum],
                             "suppression":float(suppressed[negatives].mean()) if negatives.any() and not group_small else None,
                             "fnr":float(suppressed[positives].mean()) if positives.any() and not group_small else None,
                             "status":"INFEASIBLE_LAMBDA_ZERO" if method=="mondrian" and stratum!="ALL"
                                      and thresholds[stratum]==0 else "OK"})
    return rows,thresholds,counts


def gated_votes(csv_path,allowed_keys):
    """Two-pass key gate: never load test annotator decision values."""
    header=pd.read_csv(csv_path,usecols=["record","event","is_adjudication"],
                       dtype=str,keep_default_na=False)
    keep=np.fromiter(((rec,ev) in allowed_keys for rec,ev in
                      zip(header["record"],header["event"])),dtype=bool,count=len(header))
    selected=set(np.flatnonzero(keep).tolist())
    decisions=pd.read_csv(csv_path,usecols=["decision"],dtype=str,keep_default_na=False,
                          skiprows=lambda n: n>0 and n-1 not in selected)
    hard(len(decisions)==int(keep.sum()),"annotation two-pass gate mismatched")
    chosen=header.loc[keep].reset_index(drop=True)
    chosen["decision"]=decisions["decision"].to_numpy()
    result=defaultdict(list)
    for rec,ev,adj,dec in chosen.itertuples(index=False,name=None):
        if str(adj).strip().lower() in ("true","1"):
            continue
        dec=str(dec).strip().lower()
        if dec in ("true","false","uncertain"):
            result[(rec,ev)].append(dec)
        else:
            hard(dec in ("reject","save for later"),"unrecognized annotation decision")
    return result


def sample_annotation_labels(meta,votes,rng):
    result=np.asarray(meta["y"],dtype=np.int8).copy()
    for i,(rec,ev,y,missing) in enumerate(meta[["record","event","y","votes_missing"]].itertuples(index=False,name=None)):
        if missing:
            continue
        choices=votes.get((str(rec),str(ev)),[])
        if not choices:
            continue  # Existing nonnumeric votes: no usable distribution; freeze y.
        decision=choices[int(rng.integers(0,len(choices)))]
        result[i] = (1 if decision=="true" else 0 if decision=="false"
                     else int(rng.binomial(1,.5)))
    return result


def a3_annotators(cfg,data,project,lock,conformal):
    calib,val=data["calib"],data["val"]
    allowed=set(zip(calib["meta"]["record"],calib["meta"]["event"])) | \
            set(zip(val["meta"]["record"],val["meta"]["event"]))
    source=release_paths.get("data_raw")/"vtac"/"1.1"/"event_label_per_annotator.csv"
    votes=gated_votes(source,allowed)
    disagreement={}
    uncertainty={}
    for split,d in (("calib",calib),("val",val)):
        disagreement[split]=redact(sum(
            {"true","false"}.issubset(set(votes.get((str(rec),str(ev)),[])))
            for rec,ev in zip(d["meta"]["record"],d["meta"]["event"])))
        uncertainty[split]=redact(sum(
            "uncertain" in votes.get((str(rec),str(ev)),[])
            for rec,ev in zip(d["meta"]["record"],d["meta"]["event"])))
    rng=np.random.default_rng(cfg["a3"]["seed"])
    rows=[]
    for rep in range(cfg["a3"]["replicates"]):
        yc=sample_annotation_labels(calib["meta"],votes,rng)
        yv=sample_annotation_labels(val["meta"],votes,rng)
        npos=len(set(calib["records"][yc==1]))
        lam=(float(conformal.crc_fnr_threshold(calib["prob"],yc,calib["records"],.02))
             if npos and 1/(npos+1)<=.02 else 0.0)
        supp,fnr_orig=rates(val["prob"],val["y"],lam)
        _,fnr_draw=rates(val["prob"],yv,lam)
        rows.append({"rep":rep,"lambda":lam,"suppression_val_adjudicated_false":supp,
                     "fnr_adjudicated":fnr_orig,"fnr_drawn_labels":fnr_draw,
                     "positive_calib_records":npos,"feasible":bool(npos and 1/(npos+1)<=.02)})
    frame=pd.DataFrame(rows)
    pmed=float(frame["lambda"].median())
    base_supp,_=rates(val["prob"],val["y"],float(lock["lambda"]["A"][.02]))
    med_supp,_=rates(val["prob"],val["y"],pmed)
    summary={"replicates":len(frame),"seed":42,"disagreeing_true_false_votes":disagreement,
             "uncertain_votes_events":uncertainty,"lambda_q025_q50_q975":pct(frame["lambda"]),
             "suppression_q025_q50_q975":pct(frame["suppression_val_adjudicated_false"]),
             "fnr_adjudicated_q025_q50_q975":pct(frame["fnr_adjudicated"]),
             "fnr_drawn_q025_q50_q975":pct(frame["fnr_drawn_labels"]),
             "n_infeasible":redact(int((~frame["feasible"]).sum())),
             "cost_locked_minus_mc_median_suppression":float(base_supp-med_supp),
             "note":"No numeric annotator votes => original y; votes_missing=1 fixed y"}
    return rows,summary


def stratified_record_plan(records,y,target):
    """Fixed-size record bootstrap within positive- and negative-record classes."""
    members=groupings(records)
    n=np.asarray([len(loc) for loc in members],dtype=float)
    positive_events=np.asarray([np.count_nonzero(y[loc]==1) for loc in members],dtype=float)
    pos=np.flatnonzero(positive_events>0)
    neg=np.flatnonzero(positive_events==0)
    hard(len(pos)>0 and len(neg)>0,"record class missing")
    a=float(np.mean(positive_events[pos]))
    b=float(np.mean(n[pos]))
    c=float(np.mean(n[neg]))
    q=target*c/(a-target*b+target*c)
    hard(np.isfinite(q) and 0<q<1,
         "target event prevalence infeasible by uniform within-class record draws")
    mpos=int(round(q*len(members)))
    mneg=len(members)-mpos
    hard(mpos>0 and mneg>0,"record class allocation became empty")
    expected=mpos*a/(mpos*b+mneg*c)
    return members,pos,neg,mpos,mneg,expected


def a4_prevalence(cfg,data,lock):
    rows=[]
    tau=float(lock["ltt_comparator"]["tau"])
    hard(abs(tau-.09)<1e-12,"Director LTT tau changed")
    for name in ("val","test","c2015"):
        d=data[name]
        for target in cfg["a4"]["prevalence"]:
            members,pos,neg,npos,nneg,expected=stratified_record_plan(d["records"],d["y"],target)
            rng=np.random.default_rng(cfg["a4"]["seed"])
            for rep in range(cfg["a4"]["bootstrap_replicates"]):
                choice=np.r_[rng.choice(pos,size=npos,replace=True),
                             rng.choice(neg,size=nneg,replace=True)]
                idx=np.concatenate([members[i] for i in choice])
                y,p=d["y"][idx],d["prob"][idx]
                for method,lam,inclusive in (("CRC",float(lock["lambda"]["A"][.02]),False),
                                             ("LTT",tau,True)):
                    masked=p<=lam if inclusive else p<lam
                    sup_count=int(masked.sum())
                    rows.append({"cohort":name,"evaluation":"VALIDATION" if name=="val" else "POST-HOC",
                                 "target_prevalence":target,"expected_prevalence":expected,
                                 "actual_prevalence":float(y.mean()),"rep":rep,"method":method,
                                 "fnr":float(masked[y==1].mean()) if np.any(y==1) else None,
                                 "p_true_given_suppressed":float(y[masked].mean()) if sup_count else None})
    frame=pd.DataFrame(rows)
    summary=[]
    for (cohort,target,method),group in frame.groupby(["cohort","target_prevalence","method"],sort=True):
        summary.append({"cohort":cohort,"target_prevalence":target,"method":method,
                        "replicates":len(group),"attained_prevalence_q025_q50_q975":pct(group.actual_prevalence),
                        "fnr_q025_q50_q975":pct(group.fnr),
                        "risk_y1_given_suppressed_q025_q50_q975":pct(group.p_true_given_suppressed)})
    return rows,summary


def a5_burden(cfg,data,lock):
    rows=[]
    tau=float(lock["ltt_comparator"]["tau"])
    hard(abs(tau-.09)<1e-12,"Director LTT tau changed")
    for name in ("test","c2015"):
        d=data[name]
        p0=float(cfg["a5"]["prevalence"]["vtac" if name=="test" else "c2015"])
        members=groupings(d["records"])
        methods=[(f"CRC_{a:.2f}",float(lock["lambda"]["A"][a]),False)
                 for a in (.01,.02,.05)]+[("LTT",tau,True)]
        rng=np.random.default_rng(cfg["a5"]["seed"])
        for rep in range(cfg["a5"]["bootstrap_replicates"]):
            choice=rng.integers(0,len(members),size=len(members))
            idx=np.concatenate([members[i] for i in choice])
            y,p=d["y"][idx],d["prob"][idx]
            for method,lam,inclusive in methods:
                suppression,fnr=rates(p,y,lam,inclusive=inclusive)
                rows.append({"cohort":name,"evaluation":"POST-HOC","rep":rep,"method":method,
                             "prevalence":p0,"false_avoided_per_1000":
                             float(1000*(1-p0)*suppression) if suppression is not None else None,
                             "true_blocked_per_1000":float(1000*p0*fnr) if fnr is not None else None})
    frame=pd.DataFrame(rows)
    summary=[]
    for (name,method),f in frame.groupby(["cohort","method"],sort=True):
        summary.append({"cohort":name,"evaluation":"POST-HOC","method":method,
                        "prevalence":float(f.prevalence.iloc[0]),
                        "false_avoided_q025_q50_q975":pct(f.false_avoided_per_1000),
                        "true_blocked_q025_q50_q975":pct(f.true_blocked_per_1000)})
    return rows,summary


def figure_a1(summary,figpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5.5),dpi=150)
    primary=sorted([r for r in summary if r["model"]=="ensemble"],key=lambda r:r["n_pos"])
    x=np.asarray([r["n_pos"] for r in primary])
    lo=np.asarray([r["supp_q025_q50_q975"][0] for r in primary])
    med=np.asarray([r["supp_q025_q50_q975"][1] for r in primary])
    hi=np.asarray([r["supp_q025_q50_q975"][2] for r in primary])
    ax.plot(x,med,"o-",label="Ensemble A (median)",color="#155e75",linewidth=2)
    ax.fill_between(x,lo,hi,alpha=.23,color="#155e75",label="Ensemble 95% interval")
    for row in summary:
        if row["model"].startswith("seed"):
            p=row["supp_q025_q50_q975"]
            xx=188+(int(row["model"][-1])-2)*2.2
            ax.errorbar([xx],[p[1]],yerr=[[p[1]-p[0]],[p[2]-p[1]]],fmt=".",capsize=2,
                        color="#a16207",alpha=.65)
    ax.scatter([],[],marker=".",color="#a16207",label="Individual seeds (n=188 only)")
    ax.axhline(.30,color="#777",linestyle=":",label="Suppression = 0.30")
    ax.set(xlabel="Nominal positive calibration records",ylabel="Val false-alarm suppression",
           title="T06 A1: calibration size and uncertainty (alpha=0.02)")
    ax.set_ylim(-.03,1.03)
    ax.grid(alpha=.2)
    ax.legend(loc="best",fontsize=8)
    fig.tight_layout()
    fig.savefig(figpath,dpi=180)
    plt.close(fig)


def report_md(cfg,refs,a1,a1anchor,pac,a2,a2thresholds,a2counts,a3,a4,a5):
    lines=["# P05-T06 branch A: preregistered A1-A5", "",
           "Status: COMPLETE; test and Challenge 2015 results are **POST-HOC**.",
           "No training, no retuning, no alteration of locked T05b outputs.",
           "",f"Final lock SHA256: `{cfg['director_final_lock_sha256']}`", "",
           "Step 0 reproduced AUROC: test **0.946049**, Challenge 2015 **0.942304** within 1e-6.",
           "", "## A1. Calibration size and PAC", "",
           "### A1a. Binary record-loss theory", "",
           "For record j, L_j(lambda)=1 if any true event in that record has p<lambda; else 0.",
           "CRC criterion: (sum_j L_j(lambda)+1)/(n+1) <= alpha.",
           "For ordered positive-record minima r_(1)<=...<=r_(n), the largest",
           "eligible strict threshold is r_(k*) with k*=floor(alpha*(n+1))>=1.",
           "Its population miss probability approximately follows Beta(k*,n+1-k*).",
           "**Important:** the frozen common CRC averages event-level misses within each positive record;",
           "the binary loss and Beta model are explanatory approximations when a record has multiple positives.", "",
           "| n positive | alpha | k* | Beta 5/50/95% |",
           "|---:|---:|---:|---|"]
    for row in refs:
        q=[row["beta_q05"],row["beta_median"],row["beta_q95"]]
        lines.append(f"| {row['n_pos']} | {row['alpha']:.2f} | {row['kstar'] if row['kstar'] else 'N/A'} | {fmt_ci(q)} |")
    lines += ["", "### A1b. Record subsampling and bootstrap on validation", "",
              "At nominal n=188 the T05a full-record bootstrap is reproduced (all 543 records sampled with replacement);",
              "the number of positive records varies across bootstrap draws.", "",
              "| n nominal | model | Lambda q2.5/50/97.5 | Suppression q2.5/50/97.5 | FNR q2.5/50/97.5 | P(supp<0.30) |",
              "|---:|---|---|---|---|---:|"]
    for row in a1:
        lines.append(f"| {row['n_pos']} | {row['model']} | {fmt_ci(row['lambda_q025_q50_q975'])} | "
                     f"{fmt_ci(row['supp_q025_q50_q975'])} | {fmt_ci(row['fnr_q025_q50_q975'])} | "
                     f"{row['p_supp_below_030']:.5f} |")
    lines += ["", "Theoretical Beta versus observed validation FNR (alpha=0.02):", "",
              "| n nominal | Beta miss q5/50/95% | Validation FNR q2.5/50/97.5% |",
              "|---:|---|---|"]
    beta_at_alpha={r['n_pos']:r for r in refs if abs(r['alpha']-.02)<1e-12}
    for row in a1:
        if row['model']!='ensemble':
            continue
        b=beta_at_alpha[row['n_pos']]
        lines.append(f"| {row['n_pos']} | {fmt_ci([b['beta_q05'],b['beta_median'],b['beta_q95']])} | "
                     f"{fmt_ci(row['fnr_q025_q50_q975'])} |")
    lines += ["", "The Beta model concerns binary any-event record loss, whereas the displayed",
              "validation FNR is calculated over events. Their quantiles need not coincide.",
              "", "T05a benchmark recalculated using 2000 identical seeded full-record resamples:",
              f"`{a1anchor}`", "", "### A1c. PAC record miss", "",
              "One-sided CP upper bound <= alpha, delta=0.05; n_min=ceil(log(delta)/log(1-alpha)).", "",
              "| alpha | cohort | scope | n_min | lambda PAC | Status | Suppression | FNR |",
              "|---:|---|---|---:|---:|---|---:|---:|"]
    for row in pac:
        lines.append(f"| {row['alpha']:.2f} | {row['cohort']} | {row['evaluation']} | "
                     f"{row['n_minimum']} | {row['lambda_pac']:.9g} | {row['status']} | "
                     f"{format_rate(row['suppression'])} | {format_rate(row['fnr'])} |")
    lines += ["", "A1d plot: `outputs/t06/figs_A1_calibration_suppression.png`.",
              "", "## A2. Mondrian CRC versus locked marginal CRC", "",
              "Positive calibration records and Mondrian thresholds (infeasible -> lambda=0):", "",
              "| Stratum | Calibration positive records | Lambda |",
              "|---|---:|---:|"]
    for s in ("S0","S1","S2"):
        lines.append(f"| {s} | {a2counts[s]} | {a2thresholds[s]:.9g} |")
    lines += ["",
              "| Cohort | Scope | Method | Stratum | True count | False count | Suppression | FNR |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for row in a2:
        lines.append(f"| {row['cohort']} | {row['evaluation']} | {row['method']} | {row['stratum']} | "
                     f"{row['n_true']} | {row['n_false']} | {format_rate(row['suppression'])} | "
                     f"{format_rate(row['fnr'])} |")
    lines += ["", "## A3. Annotator uncertainty (calib and val ONLY)", "",
              f"Monte Carlo: {a3['replicates']} draws, random seed {a3['seed']}. ",
              f"Disagreeing True/False votes: {a3['disagreeing_true_false_votes']}; "
              f"events with uncertain votes: {a3['uncertain_votes_events']}.", "",
              "| Statistic | q2.5/50/97.5% |", "|---|---|"]
    for label,key in (("CRC lambda","lambda_q025_q50_q975"),
                      ("False-alarm suppression on val","suppression_q025_q50_q975"),
                      ("Val FNR, adjudicated","fnr_adjudicated_q025_q50_q975"),
                      ("Val FNR, sampled votes","fnr_drawn_q025_q50_q975")):
        lines.append(f"| {label} | {fmt_ci(a3[key])} |")
    lines += ["",f"Infeasible Monte Carlo draws: {a3['n_infeasible']}.",
              f"Ambiguity cost (locked suppression - median-draw suppression): "
              f"{a3['cost_locked_minus_mc_median_suppression']:.6f}.",
              "Missing annotator votes keep the adjudicated event label.",
              "", "## A4. Record-bootstrap prevalence illustration", "",
              "Fixed CRC alpha=0.02 and LTT tau=0.09; no threshold selection.",
              "Records are sampled with replacement **within positive/negative record classes**.",
              "The class allocation targets expected event prevalence; rounding and record",
              "event counts make actual prevalence vary per draw.", "",
              "| Cohort | Target | Method | Attained prevalence q2.5/50/97.5 | FNR q2.5/50/97.5 | P(true|suppressed) q2.5/50/97.5 |",
              "|---|---:|---|---|---|---|"]
    for row in a4:
        lines.append(f"| {row['cohort']} | {row['target_prevalence']:.3f} | {row['method']} | "
                     f"{fmt_ci(row['attained_prevalence_q025_q50_q975'])} | "
                     f"{fmt_ci(row['fnr_q025_q50_q975'])} | "
                     f"{fmt_ci(row['risk_y1_given_suppressed_q025_q50_q975'])} |")
    lines += ["", "A4 is an illustration, **not** a hypothesis test.",
              "", "## A5. Clinical burden per 1000 VT alarms", "",
              "Fixed prevalence 0.286 (VTaC) and 0.261 (C2015); record bootstrap B=2000, seed=42.",
              "All estimates from test/C2015 are POST-HOC. No bed-day conversion.", "",
              "| Cohort | Prevalence | Method | False alarms avoided q2.5/50/97.5 | True VT blocked q2.5/50/97.5 |",
              "|---|---:|---|---|---|"]
    for row in a5:
        lines.append(f"| {row['cohort']} | {row['prevalence']:.3f} | {row['method']} | "
                     f"{fmt_ci(row['false_avoided_q025_q50_q975'])} | "
                     f"{fmt_ci(row['true_blocked_q025_q50_q975'])} |")
    lines += ["", "Privacy: counts below 10 are withheld. No record/event IDs or waveforms reported.",
              "T05b final lock, TEST_OPENED.flag, all prior metrics and outputs remain unchanged.",
              "", "**WAIT_FOR_DIRECTOR: no unregistered further analysis.**", ""]
    return "\n".join(lines)


def run():
    started=time.monotonic()
    project=Path(__file__).resolve().parents[1]
    config=project/"configs"/"t06.yaml"
    cfg=yaml.safe_load(config.read_text(encoding="utf-8"))
    hard(cfg["step"]=="P05-T06" and cfg["branch"]=="A" and cfg["no_training"]
         and cfg["cpu_threads"]<=8,"incorrect T06 protocol")
    lock_path=project/"configs"/"final_lock.yaml"
    hard(sha(lock_path)==cfg["director_final_lock_sha256"],"Director lock SHA mismatch")
    lock=yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    import t02c_pilot as core  # Does not import torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # t02c_pilot's import sets it to 1.
    core.below_normal_priority()
    hard("torch" not in sys.modules,"CPU T06 imported torch unexpectedly")
    common_dir=release_paths.get("repo") / "common"
    hard(core.common_source_sha256(common_dir)==lock["frozen_sha256"]["common_tree"],
         "frozen common SHA mismatch")
    sys.path.insert(0,str(common_dir.parent))
    from common import conformal,metrics
    data=source_data(project,cfg,lock,conformal,metrics)
    hard(np.isfinite(data["calib"]["prob"]).all() and
         np.isfinite(data["val"]["prob"]).all(),"invalid calibration/validation probabilities")
    report_path=project/cfg["report_file"]
    hard(not report_path.exists(),"T06 report already exists; refuse overwrite")
    print("P05_T06_A1_THEORY_START",flush=True)
    theory=a1_theory(cfg)
    print("P05_T06_A1_RESAMPLE_START",flush=True)
    resample,summary,anchor=a1_resample(cfg,data,conformal)
    print("P05_T06_A1_PAC_START",flush=True)
    pac=a1_pac(cfg,data)
    print("P05_T06_A2_START",flush=True)
    a2,thresholds,counts=a2_mondrian(cfg,data,lock,conformal)
    print("P05_T06_A3_START",flush=True)
    a3,asummary=a3_annotators(cfg,data,project,lock,conformal)
    print("P05_T06_A4_START",flush=True)
    a4,a4summary=a4_prevalence(cfg,data,lock)
    print("P05_T06_A5_START",flush=True)
    a5,a5summary=a5_burden(cfg,data,lock)
    text=report_md(cfg,theory,summary,anchor,pac,a2,thresholds,counts,
                   asummary,a4summary,a5summary)
    # No derived files are written until all A1-A5 calculations have succeeded.
    out=project/cfg["outputs_subdir"]
    generated=[]
    for name,rows in (("A1_theory",theory),("A1_resampling",resample),
                      ("A1_pac",pac),("A2_mondrian",a2),
                      ("A3_annotators",a3),("A4_prevalence",a4),("A5_burden",a5)):
        path=out/f"{name}.parquet"
        hard(not path.exists(),"T06 output already exists: "+name)
        pd.DataFrame(rows).to_parquet(path,index=False)
        generated.append((path,len(rows)))
    fig=out/"figs_A1_calibration_suppression.png"
    hard(not fig.exists(),"T06 A1 figure already exists")
    figure_a1(summary,fig)
    generated.append((fig,1))
    with report_path.open("x",encoding="utf-8") as f:
        f.write(text)
    generated.append((report_path,0))
    with (project/"manifest.csv").open("a",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        now=datetime.now(timezone.utc).isoformat()
        for path,n in generated:
            writer.writerow([str(path),sha(path),n,now,"P05-T06"])
    print("=== REPORT P05-T06 ===",flush=True)
    print("status: COMPLETE; WAIT_FOR_DIRECTOR; pc: reference-workstation; CPU<=8; BelowNormal; GPU: NONE",flush=True)
    print(f"runtime_s: {time.monotonic()-started:.3f}; script_sha256_8: {sha(__file__)[:8]}",flush=True)
    print("data: frozen T03 calib/val; T06 step0 test482/C2015-VT341; no training",flush=True)
    print("step0: AUROC test0.946049/C2015-0.942304 PASS",flush=True)
    print("A1: theory+calibration bootstrap+PAC+plot PASS",flush=True)
    print(f"A1_T05A_ANCHOR_RECALC={anchor['recomputed_t05a_2000']}",flush=True)
    print("A2: marginal/Mondrian modality strata PASS",flush=True)
    print("A3: annotator Monte Carlo calib/val only PASS",flush=True)
    print("A4: fixed-threshold prevalence illustration PASS",flush=True)
    print("A5: burden per 1000 record bootstrap PASS",flush=True)
    print(f"outputs: {out}; report: {report_path}",flush=True)
    print("warnings: test/C2015 POST-HOC; T05b untouched; small counts redacted",flush=True)
    print("=== END ===",flush=True)


if __name__=="__main__":
    try:
        run()
    except BaseException as exc:
        print(f"P05_T06_A_STOP={type(exc).__name__}: {exc}",file=sys.stderr,flush=True)
        raise
