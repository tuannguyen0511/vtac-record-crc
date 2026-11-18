"""P05-T10 POST-HOC FCN and optional matched-size mixed-profile control.

Training schedule is copied from frozen train_lib.train_model but does not save
per-alarm score files. Only new T10 checkpoint and aggregate metrics are written.
GPU isolation must occur before torch import; physical GPU index 1 (Quadro M6000).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from t10_shared import must,sha,save_json,save_csv,sigmoid,out_metrics,CORE_ALPHAS,red


def gpu_modules(ctx):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torch
    import train_lib
    import models
    import t02c_pilot as frozen
    cfg=frozen.validate_frozen(ctx.project/"configs"/"t02c.yaml")
    must(sha(ctx.project/"src"/"models.py")==ctx.lock["frozen_sha256"]["models_py"],"GPU_MODELS_FROZEN")
    must(sha(ctx.project/"src"/"train_lib.py")==ctx.lock["frozen_sha256"]["train_lib_py"],"GPU_TRAIN_LIB_FROZEN")
    must(torch.cuda.is_available(),"CUDA_AVAILABLE")
    device=train_lib.cuda_device()
    must(torch.cuda.device_count()==1 and "Quadro M6000" in torch.cuda.get_device_name(0),"PHYSICAL_GPU1_M6000")
    torch.set_num_threads(8)
    return cfg,torch,train_lib,models,frozen,device


def snapshot_cfg(ctx,cfg):
    # Original T02c config is relative to workspace. Ensure no private paths
    # are serialized into newly created T10 checkpoints.
    cfg=dict(cfg)
    cfg["project_dir"]=ctx.project.name
    cfg["proc_dir"]=str(ctx.proc.relative_to(ctx.workspace)).replace("\\","/")
    cfg["common_dir"]="common"
    cfg["locked_design"]=str((ctx.project/"configs"/"locked_design.yaml").relative_to(ctx.workspace)).replace("\\","/")
    cfg["cpu_threads"]=8
    return cfg


def checkpoint_dir(ctx,kind,seed):
    return ctx.out/("fcn" if kind=="N3" else "n11")/(f"fcn_s{seed}" if kind=="N3" else f"mixed_s{seed}")


def train_once(ctx,kind,seed,fit,val,deadline=None):
    cfg,torch,train_lib,models,frozen,device=gpu_modules(ctx)
    cfg=snapshot_cfg(ctx,cfg)
    model_name="fcn" if kind=="N3" else "resnet1d"
    run=checkpoint_dir(ctx,kind,seed)
    ckptfile=run/"ckpt.pt"
    resultfile=run/"aggregate_training.json"
    if ckptfile.exists() or resultfile.exists():
        must(ckptfile.is_file() and resultfile.is_file(),"TRAIN_EXISTING_COMPLETE_"+kind)
        d=json.loads(resultfile.read_text(encoding="utf-8"))
        must(d.get("checkpoint_sha256")==sha(ckptfile) and d["seed"]==seed and
             d["model"]==model_name and d["source_lock_sha256"]==sha(ctx.lock_path)
             and d.get("trainer_sha256")==sha(Path(__file__))
             and d.get("t02c_cfg_sha256")==sha(ctx.project/"configs"/"t02c.yaml"),
             "TRAIN_REUSE_INTEGRITY_"+kind)
        print(f"T10_{kind}_TRAIN_EXISTING_VERIFIED=PASS; seed={seed}",flush=True)
        return d
    must(not run.exists(),"TRAIN_OUTPUT_ABSENT_"+kind)
    train_lib.seed_everything(seed,8)
    net,parameters=models.make_model(model_name)
    net=net.to(device)
    sp=cfg["model_specs"][model_name]
    optimizer=(torch.optim.Adam if sp["optimizer"]=="Adam" else torch.optim.AdamW)(
        net.parameters(),lr=float(sp["lr"]),weight_decay=float(sp["weight_decay"]))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(cfg["max_epochs"])) if sp["cosine_decay"] else None
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(sp["pos_weight"]),device=device))
    batch_size=int(sp["batch_size"])
    epochs=int(cfg["max_epochs"]);patience=int(cfg["patience"])
    best_ap=-float("inf");best_state=None;best_epoch=0;stalled=0
    tic=time.perf_counter();epoch_no=0
    for epoch in range(1,epochs+1):
        if deadline is not None and time.time()>deadline:
            print("T10_N11_TIME_LIMIT_REACHED=YES; checkpoint_not_created=YES",flush=True)
            return {"status":"TIME_LIMIT", "seed":seed, "epochs_completed":epoch_no}
        net.train()
        rng=np.random.default_rng(int(seed)*1000003+epoch)
        order=rng.permutation(len(fit["y"]))
        for offset in range(0,len(order),batch_size):
            ids=order[offset:offset+batch_size]
            inputs=train_lib.model_input(fit["x"],fit["mask"],ids,model_name,device)
            targets=fit["y"].index_select(0,torch.as_tensor(ids,dtype=torch.long)).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred=net(inputs)
            loss=loss_fn(pred,targets)
            must(torch.isfinite(loss).item(),"TRAIN_LOSS_FINITE")
            loss.backward();optimizer.step()
        if scheduler is not None:scheduler.step()
        logits=train_lib.predict_logits(net,val,model_name,device,batch_size)
        ap=float(ctx.metrics.auprc(logits,val["y"].numpy()))
        if ap>best_ap+1e-12:
            best_ap=ap;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            stalled=0
        else:stalled+=1
        epoch_no=epoch
        print(f"T10_{kind}_EPOCH=PASS; seed={seed}; epoch={epoch}; val_auprc={ap:.6f}",flush=True)
        if stalled>=patience:break
    must(best_state is not None,"TRAIN_BEST_STATE")
    # No per-event validation/calibration logits are persisted.
    run.mkdir(parents=True,exist_ok=False)
    metadata={"status":"COMPLETE_POST-HOC","model":model_name,"seed":seed,
              "best_epoch":best_epoch,"epochs_run":epoch_no,"best_val_auprc":best_ap,
              "training_seconds":time.perf_counter()-tic,
              "source_lock_sha256":sha(ctx.lock_path),
              "frozen_models_sha256":sha(ctx.project/"src"/"models.py"),
              "frozen_trainlib_sha256":sha(ctx.project/"src"/"train_lib.py"),
              "trainer_sha256":sha(Path(__file__)),
              "t02c_cfg_sha256":sha(ctx.project/"configs"/"t02c.yaml"),
              "fit_n":len(fit["y"]),"val_earlystop_n":len(val["y"]),
              "seed_method":"frozen train_lib.seed_everything",
              "schedule":"T02c: frozen optimizer/BCE/early stopping/epoch order",
              "post_hoc":True}
    torch.save({"state_dict":best_state,"model":model_name,"seed":seed,
                "best_epoch":best_epoch,"parameters":parameters,
                "source_lock_sha256":sha(ctx.lock_path),"post_hoc":True,
                "fit_count":len(fit["y"]),"config":cfg},ckptfile)
    metadata["checkpoint_sha256"]=sha(ckptfile)
    save_json(resultfile,metadata)
    del net
    torch.cuda.empty_cache()
    print(f"T10_{kind}_TRAIN=PASS; seed={seed}; best_epoch={best_epoch}",flush=True)
    return metadata


def load_fixed_data(ctx,torch):
    import train_lib
    fit=train_lib.load_split(ctx.proc,"fit",w=10,d=0,labels=True)
    val=train_lib.load_split(ctx.proc,"val",w=10,d=0,labels=True)
    # Read-only test cached input; never fit/tune on test.
    tx=np.load(ctx.proc/"test_W10_D0_x.npy",allow_pickle=False)
    tm=np.load(ctx.proc/"test_W10_D0_mask.npy",allow_pickle=False)
    must(tx.shape==(482,4,2500) and tm.shape==(482,4),"GPU_TEST_CACHE_SHAPE")
    test={"x":torch.from_numpy(np.array(tx,dtype=np.float32,copy=True,order="C")),
          "mask":torch.from_numpy(np.array(tm,dtype=np.bool_,copy=True,order="C")),
          "idx":np.arange(482),"y":None}
    return fit,val,test


def external_data(ctx,torch):
    import wfdb
    from scipy import signal
    import preprocess_lib as prep
    import t05b_external_c2015 as ext
    folder,vt=ctx.vt()
    must((folder/"SHA256SUMS").is_file(),"GPU_EXTERNAL_SOURCE_MANIFEST")
    # Validate source integrity. Source functions never emit raw record names.
    audited=ext.audit_training(folder)
    must(len(audited)==341,"GPU_C2015_AUDIT_COUNT")
    src={"project_dir":str(ctx.project),"proc_dir":str(ctx.proc)}
    payload=ext.prepare_public_vt(src,vt,folder,prep,np,wfdb,signal)
    d={"x":torch.from_numpy(np.array(payload["x"],dtype=np.float32,copy=True,order="C")),
       "mask":torch.from_numpy(np.array(payload["mask"],dtype=np.bool_,copy=True,order="C")),
       "idx":np.arange(341),"y":None}
    return d


def inference_ensemble(ctx,kind,cohorts):
    cfg,torch,train_lib,models,frozen,device=gpu_modules(ctx)
    model_name="fcn" if kind=="N3" else "resnet1d"
    zs={k:[] for k in cohorts}
    for seed in range(5):
        run=checkpoint_dir(ctx,kind,seed)
        ckpt=run/"ckpt.pt";meta=run/"aggregate_training.json"
        must(ckpt.is_file() and meta.is_file(),"GPU_ALL_SEEDS_PRESENT_"+kind)
        d=json.loads(meta.read_text(encoding="utf-8"))
        must(sha(ckpt)==d["checkpoint_sha256"],"GPU_CHECKPOINT_INTEGRITY_"+kind)
        obj=torch.load(ckpt,map_location="cpu",weights_only=False)
        must(obj["model"]==model_name and obj["seed"]==seed and obj["source_lock_sha256"]==sha(ctx.lock_path),
             "GPU_CHECKPOINT_IDENTITY_"+kind)
        net,n_params=models.make_model(model_name)
        must(int(obj["parameters"])==n_params,"GPU_MODEL_PARAMETER_COUNT")
        net.load_state_dict(obj["state_dict"],strict=True)
        net.to(device)
        for name,dataset in cohorts.items():
            z=train_lib.predict_logits(net,dataset,model_name,device,128 if kind=="N11" else 64)
            must(len(z)==len(dataset["idx"]) and np.isfinite(z).all(),"GPU_INFERENCE_"+name.upper())
            zs[name].append(z)
        del net,obj
        torch.cuda.empty_cache()
        print(f"T10_{kind}_INFERENCE_SEED=PASS; seed={seed}",flush=True)
    frozen.verify_nvidia_process_is_gpu1()
    return {k:sigmoid(np.mean(np.stack(zz),axis=0)) for k,zz in zs.items()}


def n3_train(ctx):
    cfg,torch,train_lib,models,frozen,device=gpu_modules(ctx)
    fit,val,_=load_fixed_data(ctx,torch)
    must(len(fit["y"])==2830 and len(val["y"])==495,"N3_HARD_LABEL_FIT")
    for seed in range(5):
        train_once(ctx,"N3",seed,fit,val)
    ctx.check_unchanged()
    print("T10_N3_TRAIN_5_OF_5=PASS; FCN_WANG2017_HARD_A",flush=True)


def n3_eval(ctx):
    cfg,torch,train_lib,models,frozen,device=gpu_modules(ctx)
    fit,val,test=load_fixed_data(ctx,torch)
    cal=train_lib.load_split(ctx.proc,"calib",w=10,d=0,labels=False)
    ext=external_data(ctx,torch)
    preds=inference_ensemble(ctx,"N3",{"calib":cal,"val":val,"test":test,"c2015":ext})
    cohorts={k:ctx.cohort(k) for k in ("calib","val","test","c2015")}
    ys={k:cohorts[k]["y"] for k in cohorts}
    # Independent calibration, never use locked ResNet lambda for FCN.
    lambdas={a:float(ctx.conformal.crc_fnr_threshold(preds["calib"],ys["calib"],
                    cohorts["calib"]["rec"],a)) for a in CORE_ALPHAS}
    rows=[]
    for a,lam in lambdas.items():
        for name in ("val","test","c2015"):
            row={"cohort":name,"alpha":a,"lambda_fcn":lam,
                 "model":"FCNWang2017","ensemble":"sigmoid(mean 5 seed logits)"}
            row.update(out_metrics(preds[name],ys[name],lam,ctx.metrics,name))
            rows.append(row)
    keys=["cohort","alpha","lambda_fcn","model","ensemble","status","auroc",
          "suppression","suppressed_false","false_alarm_n","fnr","fnr_count","fnr_ci_low","fnr_ci_high"]
    save_csv(ctx.out/"N3_fcn_crc_results.csv",rows,keys)
    save_json(ctx.out/"N3_summary.json",{"status":"POST-HOC","model":"FCNWang2017",
       "training":"Exact frozen T02c FCN schedule; fit hard A; val early-stop by AUPRC",
       "seeds":[0,1,2,3,4],"lambda_fcn":{str(k):v for k,v in lambdas.items()},
       "note":"No benchmark weights downloaded. Model trained after test/Challenge labels opened."})
    ctx.check_unchanged()
    print("T10_N3_EVAL=PASS; ALPHAS=3; COHORTS=VAL,TEST,C2015; POST-HOC",flush=True)
    return rows


def take(data,selection,torch):
    indices=np.flatnonzero(selection).astype(np.int64)
    must(len(indices)>0,"N11_NONEMPTY_SUBSET")
    ids=torch.as_tensor(indices,dtype=torch.long)
    return {"x":data["x"].index_select(0,ids),
            "mask":data["mask"].index_select(0,ids),
            "y":data["y"].index_select(0,ids),"idx":data["idx"][indices]}


def n11_plan(ctx):
    from scipy.optimize import milp,LinearConstraint,Bounds
    profiles_path=ctx.project/"outputs"/"t06b"/"profiles_fit.parquet"
    must(profiles_path.is_file(),"N11_PROFILE_FIT_PRESENT")
    meta=ctx.meta("fit")
    profiles=ctx.pd.read_parquet(profiles_path)
    must(np.array_equal(meta["idx"].to_numpy(),profiles["idx"].to_numpy()),"N11_PROFILE_IDX_ALIGN")
    must(set(profiles["profile"]).issubset({"P1","P2","P3"}),"N11_PROFILE_CATEGORIES")
    records=meta["record"].astype(str).to_numpy()
    prof=profiles["profile"].astype(str).to_numpy()
    yy=meta["y"].to_numpy(dtype=int)
    names=np.unique(records)
    record_index={r:i for i,r in enumerate(names)}
    nc=len(names);ids=np.array([record_index[r] for r in records],dtype=int)
    pcounts=np.bincount(ids,weights=yy,minlength=nc).astype(int)
    fcounts=np.bincount(ids,weights=1-yy,minlength=nc).astype(int)
    rprof=[]
    for r in names:
        rr=prof[records==r]
        must(len(set(rr))==1,"N11_RECORD_PROFILE_CONSISTENT")
        rprof.append(rr[0])
    rprof=np.asarray(rprof)
    p1=rprof=="P1"
    target=np.array([int(p1.sum()),int(pcounts[p1].sum()),int(fcounts[p1].sum())])
    must(target[0]>=10 and target[1]>=10 and target[2]>=10,"N11_TARGET_COUNTS")
    # Require at least one P2 AND P3 record when both categories exist. This
    # makes the selection genuinely mixed, rather than resampling P1-only.
    constraints=[np.ones(nc),pcounts,fcounts]
    lb=target.astype(float).tolist();ub=target.astype(float).tolist()
    for klass in ("P2","P3"):
        if np.any(rprof==klass):
            constraints.append((rprof==klass).astype(float));lb.append(1);ub.append(np.inf)
    rng=np.random.default_rng(42)
    objective=rng.uniform(.01,1.0,size=nc)
    matrix=np.stack(constraints)
    result=milp(c=objective,integrality=np.ones(nc),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,np.asarray(lb),np.asarray(ub)),
                options={"time_limit":180.0,"mip_rel_gap":0.0})
    must(result.x is not None,"N11_MATCHED_MILP_FEASIBLE")
    chosen=np.rint(result.x).astype(bool)
    must(np.all(matrix[:3]@chosen==target),"N11_EXACT_MATCHED_SIZE")
    must(set(rprof[chosen])>=set([p for p in ("P2","P3") if p in rprof]),"N11_MIXED_PROFILES_PRESENT")
    selected=names[chosen]
    # Stable salted pseudonyms: the output is not a patient-level dataset.
    hashes=sorted(hashlib.sha256(("P05-T10|POST-HOC|"+s).encode()).hexdigest() for s in selected)
    rows=[{"record_sha256":h} for h in hashes]
    save_csv(ctx.out/"N11_selected_record_hashes.csv",rows,["record_sha256"])
    summary={"status":"POST-HOC","seed":42,"matched_record_count":int(target[0]),
             "matched_true_alarms":int(target[1]),"matched_false_alarms":int(target[2]),
             "selected_record_count":len(selected),
             "selected_profile_counts":{p:red(np.sum(rprof[chosen]==p)) for p in ("P1","P2","P3")},
             "solver_status":str(result.status),
             "selection_sha256":hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
             "note":"Hashes are pseudonyms, not anonymization; do not publish this file."}
    save_json(ctx.out/"N11_plan.json",summary)
    print("T10_N11_MATCHED_PLAN=PASS; no_record_ids_exported=YES",flush=True)
    return np.isin(records,selected),p1[ids]


def n11_train(ctx):
    # Only after N1-N9 complete, as ordered by Director.
    require_files=["T10_cpu_status.json","N3_summary.json","N6_fingerprint.json"]
    must(all((ctx.out/p).is_file() for p in require_files),"N11_CORE_ANALYSES_COMPLETE")
    cfg,torch,train_lib,models,frozen,device=gpu_modules(ctx)
    fit,val,_=load_fixed_data(ctx,torch)
    selected,original_p1=n11_plan(ctx)
    vp=ctx.project/"outputs"/"t06b"/"profiles_val.parquet"
    must(vp.is_file(),"N11_PROFILE_VAL_PRESENT")
    vprof=pd.read_parquet(vp)
    vm=ctx.meta("val")
    must(np.array_equal(vprof["idx"],vm["idx"]),"N11_VAL_PROFILE_ALIGN")
    stop=take(val,vprof["profile"].to_numpy()=="P1",torch)
    training=take(fit,selected,torch)
    timefile=ctx.out/"n11"/"gpu_runtime.json"
    if timefile.exists():
        progress=json.loads(timefile.read_text(encoding="utf-8"))
        must(progress["source_lock_sha256"]==sha(ctx.lock_path),"N11_RUNTIME_LOCK")
        elapsed=float(progress["cumulative_wall_seconds"])
    else:elapsed=0.0
    must(elapsed<14400,"N11_TIME_BUDGET_AVAILABLE")
    deadline=time.time()+(14400-elapsed)
    start=time.time()
    for seed in range(5):
        if time.time()>deadline:
            print("T10_N11_TIME_LIMIT=STOP; remaining seeds untrained",flush=True)
            break
        info=train_once(ctx,"N11",seed,training,stop,deadline=deadline)
        if info.get("status")=="TIME_LIMIT":break
    elapsed_new=elapsed+(time.time()-start)
    status="COMPLETE" if all((checkpoint_dir(ctx,"N11",s)/"ckpt.pt").is_file() for s in range(5)) else "TIME_LIMIT_INCOMPLETE"
    progress={"source_lock_sha256":sha(ctx.lock_path),"cumulative_wall_seconds":elapsed_new,
              "status":status,"completed_seeds":sum((checkpoint_dir(ctx,"N11",s)/"ckpt.pt").is_file() for s in range(5))}
    timefile.parent.mkdir(parents=True,exist_ok=True)
    if timefile.exists():timefile.write_text(json.dumps(progress,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    else:save_json(timefile,progress)
    ctx.check_unchanged()
    print("T10_N11_TRAIN_STATUS="+status+"; completed_seeds="+str(progress["completed_seeds"]),flush=True)
    return status


def n11_eval(ctx):
    cfg,torch,train_lib,models,frozen,device=gpu_modules(ctx)
    must(all((checkpoint_dir(ctx,"N11",s)/"ckpt.pt").exists() for s in range(5)),"N11_ALL_SEEDS_TRAINED")
    fit,val,test=load_fixed_data(ctx,torch)
    cal=train_lib.load_split(ctx.proc,"calib",w=10,d=0,labels=False)
    external=external_data(ctx,torch)
    pred=inference_ensemble(ctx,"N11",{"calib":cal,"val":val,"test":test,"c2015":external})
    cohorts={name:ctx.cohort(name) for name in ("calib","val","test","c2015")}
    lam=float(ctx.conformal.crc_fnr_threshold(pred["calib"],cohorts["calib"]["y"],
                                              cohorts["calib"]["rec"],.05))
    rows=[]
    for name in ("val","test","c2015"):
        if name=="c2015":
            use=np.ones(341,dtype=bool);target="C2015_ALL"
        else:
            profiles=pd.read_parquet(ctx.project/"outputs"/"t06b"/f"profiles_{name}.parquet")
            must(np.array_equal(profiles["idx"],cohorts[name]["meta"]["idx"]),"N11_EVAL_PROFILE_ALIGN")
            use=profiles["profile"].to_numpy()=="P2";target="P2"
        data=out_metrics(pred[name][use],cohorts[name]["y"][use],lam,ctx.metrics,name)
        rows.append({"cohort":name,"evaluated_profile":target,"model":"matched_mixed_resnet1d",
                     "alpha":.05,"lambda":lam,**data})
    # Original T06b single-profile and locked model values only read as context.
    src=ctx.project/"outputs"/"t06b"/"trained_cross_profile.parquet"
    must(src.is_file(),"N11_P1_ONLY_REFERENCE_PRESENT")
    references=pd.read_parquet(src)
    for name in ("val","test","c2015"):
        group=references[(references["cohort"]==name)&(references["source_profile"]=="P1")&
                         (references["model"]=="ensemble")&(np.isclose(references["alpha"].astype(float),.05))]
        if name!="c2015":group=group[group["evaluated_profile"]=="P2"]
        must(len(group)>=1,"N11_P1_ONLY_REFERENCE_"+name.upper())
        row=group.iloc[0]
        rows.append({"cohort":name,"evaluated_profile":"P2" if name!="c2015" else "C2015_ALL",
                     "model":"frozen_P1_only_reference","alpha":.05,
                     "lambda":None,"suppression":float(row["suppression"]),"fnr":float(row["fnr"]),
                     "status":"POST-HOC_REFERENCE", "note":"P1-only calibration differs; descriptive comparator"})
    for name in ("val","test","c2015"):
        original=cohorts[name];use=np.ones(len(original["y"]),dtype=bool)
        if name!="c2015":
            profiles=pd.read_parquet(ctx.project/"outputs"/"t06b"/f"profiles_{name}.parquet")
            use=profiles["profile"].to_numpy()=="P2"
        metric=out_metrics(original["p"][use],original["y"][use],
                           float(ctx.lock["lambda"]["A"][.05]),ctx.metrics,name)
        rows.append({"cohort":name,"evaluated_profile":"P2" if name!="c2015" else "C2015_ALL",
                     "model":"locked_A_5seed","alpha":.05,"lambda":float(ctx.lock["lambda"]["A"][.05]),**metric})
    save_json(ctx.out/"N11_evaluation.json",{"status":"POST-HOC","rows":rows,
                   "note":"Matched fit by record/true/false counts; validation P1 early-stop, evaluation P2. "
                          "Calibration all calib, compare P1-only as descriptive non-matched-calibration reference."})
    ctx.check_unchanged()
    print("T10_N11_EVAL=PASS; POST-HOC",flush=True)
    return rows
