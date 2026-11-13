"""P05-T06b: preregistered monitor-profile shift experiment.

Stages: audit (CPU, header labels only), train (GPU1 ten A runs), report
(GPU1 inference, locked profile CRC). Never modifies T02c-T06/HF1 inputs.
"""
import release_paths
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='8'

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta


def must(value,message):
    if not value: raise RuntimeError('P05_T06B_STOP: '+message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1048576),b''):h.update(buf)
    return h.hexdigest()


def red(n):
    return '<10' if int(n)<10 else str(int(n))


def safe_profile_counts(counts):
    # Protect small cells and any complementary count recoverable from totals.
    vals={s:int(counts.get(s,0)) for s in ('P1','P2','P3')}
    small=[s for s in vals if vals[s]<10]
    if small:
        hidden=set(small)
        visible=[s for s in vals if s not in hidden]
        if visible:hidden.add(max(visible,key=lambda x:vals[x]))
        return {s:('WITHHELD' if s in hidden else str(vals[s])) for s in vals}
    return {s:str(vals[s]) for s in vals}


def append_manifest(project,entries,step):
    now=datetime.now(timezone.utc).isoformat()
    with (project/'manifest.csv').open('a',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        for path,n in entries:w.writerow([str(path),sha(path),int(n),now,step])


def context(project):
    config=project/'configs'/'t06b.yaml'
    c=yaml.safe_load(config.read_text(encoding='utf-8'))
    must(c['step']=='P05-T06b' and c['authorization']=='T06B_AUTHORIZED'
         and c['seeds']==[0,1,2,3,4] and c['alphas']==[.02,.05]
         and c['physical_gpu']==1 and c['cpu_threads']<=8,
         'Director protocol differs')
    must((project/'reports'/'TEST_OPENED.flag').is_file(),'accepted T05b flag absent')
    must((project/'reports'/'P05-T06-HF1.md').is_file(),
         'HF1 must complete before T06b')
    lockpath=project/'configs'/'final_lock.yaml'
    must(sha(lockpath)==c['source_lock_sha256'],'Director final lock changed')
    lock=yaml.safe_load(lockpath.read_text(encoding='utf-8'))
    must(sha(project/'src'/'t06_branch_a.py')==c['accepted_t06_source_sha256'],
         'accepted T06 source changed')
    for key,f in [('models_py','models.py'),('train_lib_py','train_lib.py'),
                  ('t03_label_variants_py','t03_label_variants.py'),
                  ('t05a_conformal_ablation_py','t05a_conformal_ablation.py')]:
        must(sha(project/'src'/f)==lock['frozen_sha256'][key],'frozen source changed: '+f)
    proc=release_paths.get("data_proc")
    must(sha(proc/'test_labels.parquet')==lock['frozen_sha256']['test_labels_parquet'],
         'frozen official test labels SHA changed')
    sys.path.insert(0,str(project/'src'))
    import t02c_pilot as core
    core.below_normal_priority()
    must(core.common_source_sha256(release_paths.get("repo") / "common")==lock['frozen_sha256']['common_tree'],
         'frozen common SHA changed')
    sys.path.insert(0,str(project.parent))
    from common import conformal,metrics
    return c,lock,proc,core,conformal,metrics


def profile_for_signal_names(names):
    names={str(s).strip().upper() for s in names}
    if 'AVR' in names:return 'P1'
    if 'I' in names and 'III' in names:return 'P2'
    return 'P3'


def build_profiles(project,proc):
    import wfdb
    raw=release_paths.get("data_raw")/'vtac'/'1.1'/'annotated_waveforms'
    out={};results=[];preval=[];records_by_split={}
    for split,n in [('fit',2830),('calib',1230),('val',495),('test',482)]:
        meta=pd.read_parquet(proc/f'{split}_meta.parquet') if split!='test' else pd.read_parquet(proc/'test_meta.parquet')
        must(len(meta)==n and {'idx','record','event'}.issubset(meta.columns),'metadata schema/count differs')
        events=[]
        for rec,ev in meta[['record','event']].itertuples(index=False,name=None):
            hea=raw/str(rec)/(str(ev)+'.hea')
            must(hea.is_file(),'frozen VTaC header missing')
            hdr=wfdb.rdheader(str(hea.with_suffix('')))
            events.append(profile_for_signal_names(hdr.sig_name))
        tmp=meta[['idx','record']].copy()
        tmp['event_profile']=events
        major={}
        for rec,group in tmp.groupby('record',sort=False):
            ctr=Counter(group['event_profile'])
            maxn=max(ctr.values());winners=[v for v in ctr if ctr[v]==maxn]
            major[rec]=winners[0] if len(winners)==1 else 'P3'
        tmp['profile']=tmp['record'].map(major)
        out[split]=tmp[['idx','profile']].copy()
        records_by_split[split]=major
        nrecord=Counter(major.values())
        nevent=Counter(tmp['profile'])
        for typ,dct in [('records',nrecord),('events',nevent)]:
            protected=safe_profile_counts(dct)
            for p in ('P1','P2','P3'):
                results.append({'split':split,'unit':typ,'profile':p,'n':protected[p],
                                'total':red(len(major) if typ=='records' else n)})
        mask=np.load(proc/f'{split}_W10_D0_mask.npy',allow_pickle=False)
        must(mask.shape==(n,4),'mask shape differs')
        for p in ('P1','P2','P3'):
            use=tmp['profile'].to_numpy()==p
            if use.sum()<10:
                preval.append({'split':split,'profile':p,'n':red(use.sum()),
                               'ECG2':None,'PPG':None,'ABP':None,'status':'WITHHELD_SMALL_GROUP'})
            else:
                preval.append({'split':split,'profile':p,'n':red(use.sum()),
                               'ECG2':float(mask[use,1].mean()),
                               'PPG':float(mask[use,2].mean()),
                               'ABP':float(mask[use,3].mean()),'status':'OK'})
        print(f'T06B_HEADER_PROFILES=PASS; split={split}; event_count={n}',flush=True)
    total_records=sum(len(v) for v in records_by_split.values())
    allprofile=Counter(v for group in records_by_split.values() for v in group.values())
    shares={p:float(allprofile[p]/total_records) for p in ('P1','P2','P3')}
    must(all(abs(shares[p]-expect['expected'])<=expect['tolerance']
             for p,expect in [(p,{'expected':e,'tolerance':.02}) for p,e in [('P1',.68),('P2',.28)]]),
         'Director P1/P2 expected record fractions differ by more than 2pp; STOP')
    return out,results,preval,shares


def load_approved_data(project,c,lock,conformal,metrics):
    import t06_branch_a as old
    return old.source_data(project,{'outputs_subdir':'outputs/t06',
            'director_final_lock_sha256':c['source_lock_sha256'],
            'expected_auroc':{'test':.946049,'c2015':.942304}},lock,conformal,metrics)


def metric_at(probs,y,lam,metrics,alpha,cohort,train_on,assessed,method):
    y=np.asarray(y,dtype=np.int8);p=np.asarray(probs,dtype=float)
    if lam is None:
        return {'method':method,'source_profile':train_on,'evaluated_profile':assessed,
                'cohort':cohort,'evaluation':'POST-HOC' if cohort in ('test','c2015') else 'VALIDATION',
                'alpha':alpha,'lambda':None,'n_true':red(np.sum(y==1)),
                'n_false':red(np.sum(y==0)),'auroc':None,'suppression':None,'fnr':None,
                'fnr_ci_lo':None,'fnr_ci_hi':None,'status':'INFEASIBLE_CRC'}
    if len(y)==0 or np.sum(y==1)<10 or np.sum(y==0)<10:
        return {'method':method,'source_profile':train_on,'evaluated_profile':assessed,
                'cohort':cohort,'evaluation':'POST-HOC' if cohort in ('test','c2015') else 'VALIDATION',
                'alpha':alpha,'lambda':lam,'n_true':red(np.sum(y==1)),
                'n_false':red(np.sum(y==0)),'auroc':None,'suppression':None,'fnr':None,
                'fnr_ci_lo':None,'fnr_ci_hi':None,'status':'WITHHELD_SMALL_GROUP'}
    miss=p<lam
    k=int(np.count_nonzero(miss&(y==1)));n=int(np.count_nonzero(y==1))
    low,high=metrics.clopper_pearson(k,n)
    return {'method':method,'source_profile':train_on,'evaluated_profile':assessed,
            'cohort':cohort,'evaluation':'POST-HOC' if cohort in ('test','c2015') else 'VALIDATION',
            'alpha':alpha,'lambda':lam,'n_true':red(n),'n_false':red(np.sum(y==0)),
            'auroc':float(metrics.auroc(p,y)),
            'suppression':float(miss[y==0].mean()),'fnr':float(k/n),
            'fnr_ci_lo':float(low),'fnr_ci_hi':float(high),'status':'OK'}


def threshold(prob,y,rec,alpha,conformal):
    n=len(set(np.asarray(rec,dtype=str)[np.asarray(y)==1]))
    if n==0 or 1/(n+1)>alpha:return None,n
    return float(conformal.crc_fnr_threshold(prob,y,rec,alpha)),n


def assess_ab(project,c,lock,profiles,data,conformal,metrics):
    rows=[];thresholds=[]
    calib=data['calib']
    thresholds_by={}
    for donor in ('P1','P2'):
        sel=profiles['calib']['profile'].to_numpy()==donor
        d={k:np.asarray(calib[k])[sel] for k in ('prob','y','records')}
        for alpha in c['alphas']:
            lam,n=threshold(d['prob'],d['y'],d['records'],alpha,conformal)
            thresholds_by[(donor,alpha)]=lam
            thresholds.append({'profile':donor,'alpha':alpha,'calib_pos_records':red(n),
                               'lambda':lam,'status':'OK' if lam is not None else 'INFEASIBLE'})
    for cohort in ('val','test'):
        y=data[cohort]['y'];p=data[cohort]['prob']
        pflat=profiles[cohort]['profile'].to_numpy()
        for donor,target in (('P1','P2'),('P2','P1')):
            sel=pflat==target
            for alpha in c['alphas']:
                rows.append(metric_at(p[sel],y[sel],thresholds_by[(donor,alpha)],metrics,
                            alpha,cohort,donor,target,'cross_profile_crc'))
                rows.append(metric_at(p[sel],y[sel],float(lock['lambda']['A'][alpha]),metrics,
                            alpha,cohort,'A_LOCKED',target,'locked_marginal_crc'))
        for group in ('P1','P2'):
            sel=pflat==group
            for alpha in c['alphas']:
                rows.append(metric_at(p[sel],y[sel],thresholds_by[(group,alpha)],metrics,
                            alpha,cohort,group,group,'profile_mondrian_crc'))
    for alpha in c['alphas']:
        rows.append({'method':'profile_mondrian_crc','source_profile':'NONE',
                     'evaluated_profile':'C2015_UNSEEN','cohort':'c2015',
                     'evaluation':'POST-HOC','alpha':alpha,'lambda':None,
                     'n_true':'WITHHELD','n_false':'WITHHELD','auroc':None,'suppression':None,
                     'fnr':None,'fnr_ci_lo':None,'fnr_ci_hi':None,
                     'status':'N/A_C2015_NO_CALIBRATED_PROFILE_THRESHOLD'})
    return rows,thresholds


def audit(project,c,lock,proc,conformal,metrics):
    out=project/c['output_dir'];must(not out.exists(),'existing T06b outputs; no overwrites')
    profiles,counts,pres,shares=build_profiles(project,proc)
    data=load_approved_data(project,c,lock,conformal,metrics)
    metrics_rows,ths=assess_ab(project,c,lock,profiles,data,conformal,metrics)
    out.mkdir(parents=True,exist_ok=False);files=[]
    for split,frame in profiles.items():
        path=out/f'profiles_{split}.parquet'
        frame.to_parquet(path,index=False)
        files.append((path,len(frame)))
    for name,rows in [('profiles_counts',counts),('slot_availability',pres),
                      ('profile_crc',metrics_rows),('thresholds',ths)]:
        dest=out/f'{name}.parquet';pd.DataFrame(rows).to_parquet(dest,index=False)
        files.append((dest,len(rows)))
    audit_json=out/'audit_manifest.json'
    manifest={'status':'PASS','profile_shares':{k:round(v,2) for k,v in shares.items()},'files':{p.name:sha(p) for p,n in files},
              'lock_sha256':c['source_lock_sha256'],'t06b_cfg_sha256':sha(project/'configs'/'t06b.yaml')}
    audit_json.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    files.append((audit_json,1));append_manifest(project,files,'P05-T06b-AUDIT')
    print('T06B_AUDIT=PASS; profile_fractions='+json.dumps(shares),flush=True)
    print('T06B_A_B_D=PASS; GPU=NONE; TEST_C2015_POST_HOC',flush=True)


def load_profile_tables(project,c,proc):
    out=project/c['output_dir'];path=out/'audit_manifest.json'
    must(path.is_file(),'run CPU audit stage first')
    obj=json.loads(path.read_text(encoding='utf-8'))
    must(obj['status']=='PASS' and obj['lock_sha256']==c['source_lock_sha256']
         and obj['t06b_cfg_sha256']==sha(project/'configs'/'t06b.yaml'),
         'audit manifest changed')
    for file,h in obj['files'].items():must(sha(out/file)==h,'profile audit file changed: '+file)
    tables={}
    for split in ('fit','calib','val','test'):
        f=pd.read_parquet(out/f'profiles_{split}.parquet')
        meta=pd.read_parquet(proc/f'{split}_meta.parquet',columns=['idx','record']) if split!='test' else pd.read_parquet(proc/'test_meta.parquet',columns=['idx','record'])
        must(np.array_equal(meta['idx'].to_numpy(),f['idx'].to_numpy()),'profile index misalignment')
        must(f['profile'].isin(['P1','P2','P3']).all(),'invalid profile name')
        # Re-check each record has exactly one profile; no reclassification allowed.
        must((pd.concat([meta[['record']],f[['profile']]],axis=1)
              .groupby('record')['profile'].nunique()==1).all(),'profile inconsistent within record')
        tables[split]=f['profile'].to_numpy()
    return tables


def take(data,rows,torch):
    i=np.flatnonzero(rows).astype(np.int64)
    must(len(i)>0,'empty training/validation profile')
    ids=torch.as_tensor(i,dtype=torch.long)
    return {'x':data['x'].index_select(0,ids),
            'mask':data['mask'].index_select(0,ids),
            'y':data['y'].index_select(0,ids),
            'idx':data['idx'][i]}


def verify_existing_run(path,train_lib,model,seed,sha_map,lock):
    must(train_lib.completed_run(path),'unfinished training run output; preserve for audit')
    import torch
    ckpt=torch.load(path/'ckpt.pt',map_location='cpu')
    must(ckpt['model']==model and int(ckpt['seed'])==seed,'wrong checkpoint metadata')
    must(ckpt['script_sha256']==sha_map,'script/inputs changed since completed run')
    must(ckpt['config']['expected_locked_sha256']==lock['frozen_sha256']['locked_design_yaml'],
         'run not linked to frozen locked design')


def train(project,c,lock,proc,core,conformal,metrics):
    import torch
    import train_lib
    import models
    torch.set_num_threads(8)
    profiles=load_profile_tables(project,c,proc)
    old_cfg=yaml.safe_load((project/'configs'/'t02c.yaml').read_text(encoding='utf-8'))
    must(old_cfg['model_specs']['resnet1d']=={'optimizer':'AdamW','lr':.001,
        'weight_decay':.01,'batch_size':128,'pos_weight':1.0,'cosine_decay':True},
        'T02c resnet1d hyperparameters changed')
    must(old_cfg['max_epochs']==150 and old_cfg['patience']==20,'T02c early stopping changed')
    old_cfg['cpu_threads']=8
    fit=train_lib.load_split(proc,'fit',w=10,d=0,labels=True)
    val=train_lib.load_split(proc,'val',w=10,d=0,labels=True)
    device=train_lib.cuda_device()
    must('Quadro M6000' in torch.cuda.get_device_name(device),'physical GPU1 incorrect')
    scripts={'t06b_profile_shift_py':sha(__file__),'t06b_yaml':sha(project/'configs'/'t06b.yaml'),
             'models_py':sha(project/'src'/'models.py'),'train_lib_py':sha(project/'src'/'train_lib.py')}
    for donor in ('P1','P2'):
        train_mask=profiles['fit']==donor
        stop_mask=profiles['val']==donor
        must(len(np.unique(fit['y'].numpy()[train_mask]))==2 and
             len(np.unique(val['y'].numpy()[stop_mask]))==2,
             'fit/val profile lacks both classes')
        f=take(fit,train_mask,torch);v=take(val,stop_mask,torch)
        for seed in c['seeds']:
            dest=project/c['output_dir']/'train'/f'{donor}_s{seed}'
            if dest.exists():
                verify_existing_run(dest,train_lib,'resnet1d',seed,scripts,lock)
                print(f'T06B_TRAIN_SKIP_VERIFIED=PASS; profile={donor}; seed={seed}',flush=True)
                continue
            result=train_lib.train_model('resnet1d',seed,f,v,old_cfg,device,metrics,
                                         output_dir=dest,sha=scripts)
            must(train_lib.completed_run(dest),'training artifacts incomplete')
            verify_existing_run(dest,train_lib,'resnet1d',seed,scripts,lock)
            generated=[(dest/f,1) for f in ('ckpt.pt','scores_val.parquet','scores_calib.parquet','log.csv')]
            append_manifest(project,generated,'P05-T06b-TRAIN')
            print(f'T06B_TRAIN=PASS; profile={donor}; seed={seed}; epochs={result["epochs_run"]}',flush=True)
    core.verify_nvidia_process_is_gpu1()
    print('T06B_TRAIN_ALL_10=PASS; NO_EXISTING_T03_MODEL_MODIFIED',flush=True)


def eval_new_models(project,c,lock,proc,core,conformal,metrics):
    import torch
    import train_lib
    import models
    import wfdb
    from scipy import signal
    import preprocess_lib as prep
    import t05b_external_c2015 as ext
    torch.set_num_threads(8)
    profiles=load_profile_tables(project,c,proc)
    runroot=project/c['output_dir']/'train'
    for donor in ('P1','P2'):
        for seed in range(5):
            must(train_lib.completed_run(runroot/f'{donor}_s{seed}'),'10 required training runs incomplete')
    val=train_lib.load_split(proc,'val',w=10,d=0,labels=True)
    valmeta=pd.read_parquet(proc/'val_meta.parquet')
    calmeta=pd.read_parquet(proc/'calib_meta.parquet')
    cal_profile=profiles['calib']
    testmeta=pd.read_parquet(proc/'test_meta.parquet')
    labels=pd.read_parquet(proc/'test_labels.parquet')
    must(np.array_equal(labels['idx'].to_numpy(),testmeta['idx'].to_numpy()),'test idx incorrect')
    tx=np.load(proc/'test_W10_D0_x.npy',allow_pickle=False)
    tm=np.load(proc/'test_W10_D0_mask.npy',allow_pickle=False)
    test={'x':torch.from_numpy(np.array(tx,copy=True)),
          'mask':torch.from_numpy(np.array(tm,copy=True)),
          'idx':testmeta['idx'].to_numpy(),'y':None}
    frozen_cfg=core.validate_frozen(project/'configs'/'t02c.yaml')
    ext_folder=ext.external_root(frozen_cfg)
    vt=ext.audit_training(ext_folder)
    must(len(vt)==341,'Challenge 2015 VT cohort changed')
    external=ext.prepare_public_vt(frozen_cfg,vt,ext_folder,prep,np,wfdb,signal)
    extdata={'x':torch.from_numpy(np.array(external['x'],dtype=np.float32,copy=True)),
             'mask':torch.from_numpy(np.array(external['mask'],dtype=np.bool_,copy=True)),
             'idx':np.arange(len(vt)),'y':None}
    device=train_lib.cuda_device()
    logits={};files=[]
    sha_map={'t06b_profile_shift_py':sha(__file__),'t06b_yaml':sha(project/'configs'/'t06b.yaml'),
             'models_py':sha(project/'src'/'models.py'),'train_lib_py':sha(project/'src'/'train_lib.py')}
    for donor in ('P1','P2'):
        logits[donor]={'calib':[],'val':[],'test':[],'c2015':[]}
        for seed in range(5):
            folder=runroot/f'{donor}_s{seed}'
            verify_existing_run(folder,train_lib,'resnet1d',seed,sha_map,lock)
            ckpt=torch.load(folder/'ckpt.pt',map_location='cpu')
            net,nparams=models.make_model('resnet1d')
            must(nparams==ckpt['parameters'],'model architecture differs')
            net.load_state_dict(ckpt['state_dict'],strict=True);net.to(device)
            cal=pd.read_parquet(folder/'scores_calib.parquet')
            must(len(cal)==len(calmeta) and cal['idx'].is_unique,'train calib scores malformed')
            zc=cal.set_index('idx')['logit'].reindex(pd.Index(calmeta['idx'])).to_numpy(dtype=float)
            must(np.isfinite(zc).all(),'nonfinite calib logit')
            logits[donor]['calib'].append(zc)
            for name,data in [('val',val),('test',test),('c2015',extdata)]:
                z=train_lib.predict_logits(net,data,'resnet1d',device,128)
                must(np.isfinite(z).all(),'nonfinite inference')
                logits[donor][name].append(z)
            del net,ckpt;torch.cuda.empty_cache()
            print(f'T06B_EXTERNAL_PROFILE_INFERENCE=PASS; profile={donor}; seed={seed}',flush=True)
    core.verify_nvidia_process_is_gpu1()
    sig=lambda z: np.exp(-np.logaddexp(0,-np.asarray(z,dtype=float)))
    outs=[]
    for donor,target in (('P1','P2'),('P2','P1')):
        # threshold is calibrated strictly on positive-record events in same donor profile.
        idx=cal_profile==donor
        for model in ['ensemble']+[f'seed{x}' for x in range(5)]:
            selected=range(5) if model=='ensemble' else [int(model[-1])]
            probs={name:sig(np.mean([logits[donor][name][i] for i in selected],axis=0))
                   for name in ('calib','val','test','c2015')}
            for alpha in c['alphas']:
                lam,n=threshold(probs['calib'][idx],calmeta['y'].to_numpy(dtype=np.int8)[idx],
                                calmeta['record'].to_numpy(dtype=str)[idx],alpha,conformal)
                for cohort,mask,y in (
                    ('val',profiles['val']==target,valmeta['y'].to_numpy(dtype=np.int8)),
                    ('test',profiles['test']==target,labels['y'].to_numpy(dtype=np.int8)),
                    ('c2015',np.ones(len(vt),dtype=bool),np.asarray([v[1] for v in vt],dtype=np.int8))):
                    row=metric_at(probs[cohort][mask],y[mask],lam,metrics,alpha,
                                  cohort,donor,target if cohort!='c2015' else 'C2015_EXTERNAL',
                                  'profile_only_train')
                    row['model']=model;row['donor_calib_pos_records']=red(n)
                    outs.append(row)
    result=pd.DataFrame(outs)
    return result


def report(project,c,lock,proc,core,conformal,metrics):
    dest=project/c['output_dir'];reportpath=project/c['report_file']
    must(not reportpath.exists(),'T06b report exists; refuse overwrite')
    result=eval_new_models(project,c,lock,proc,core,conformal,metrics)
    newpath=dest/'trained_cross_profile.parquet'
    must(not newpath.exists(),'existing profile training evaluation; refuse overwrite')
    files=[(newpath,len(result))]
    audit_manifest=json.loads((dest/'audit_manifest.json').read_text(encoding='utf-8'))
    lines=['# P05-T06b: frozen-profile shift (12 November 2026)','',
           'Status: COMPLETE; all test/C2015 analyses are **POST-HOC**.',
           'Monitor profiles are header-derived proxies, NOT verified vendor labels.',
           'P1: aVR present. P2: I and III present with no aVR. P3: other. Record majority; ties -> P3.',
           'For C2015, P1/P2 Mondrian has no calibrated threshold; mark not applicable.',
           'CRC control is calibrated within the source profile. It does NOT guarantee the same target-profile FNR under domain shift.',
           'No T05b/T06 threshold, checkpoint, or archived output is modified.',
           '',f"Director lock SHA256: `{c['source_lock_sha256']}`",'']
    lines+=['## Profile distribution (header-only)','',
            '| Split | Unit | Profile | N | Total |','|---|---|---|---:|---:|']
    df=pd.read_parquet(dest/'profiles_counts.parquet')
    for z in df.itertuples(index=False):lines.append(f'| {z.split} | {z.unit} | {z.profile} | {z.n} | {z.total} |')
    lines+=['','Record-profile fractions: '+str(audit_manifest['profile_shares']),
             '', '## Modality availability (mask; header-derived profiles)','',
             '| Split | Profile | Events | ECG2 | PPG | ABP |',
             '|---|---|---:|---:|---:|---:|']
    df=pd.read_parquet(dest/'slot_availability.parquet')
    for z in df.itertuples(index=False):
        v=[getattr(z,k) for k in ('ECG2','PPG','ABP')]
        lines.append(f'| {z.split} | {z.profile} | {z.n} | '+' | '.join('N/A' if x is None or pd.isna(x) else f'{x:.6f}' for x in v)+' |')
    lines+=['','## Profile-calibrated thresholds','',
            '| Donor | Alpha | Positive calib records | Lambda | Status |',
            '|---|---:|---:|---:|---|']
    df=pd.read_parquet(dest/'thresholds.parquet')
    for _,z in df.iterrows():
        lam=z['lambda']
        lines.append(f"| {z['profile']} | {z['alpha']:.2f} | {z['calib_pos_records']} | "+
                     ('N/A' if pd.isna(lam) else f'{lam:.9g}')+f" | {z['status']} |")
    def metrics_table(df,heading):
        lines.extend(['',heading,'',
          '| Cohort | Scope | Method | Source | Target | Model | Alpha | AUROC | Suppression | FNR | 95% CP CI | Status |',
          '|---|---|---|---|---|---|---:|---:|---:|---:|---|---|'])
        for _,z in df.iterrows():
            def f(x):return 'N/A' if pd.isna(x) else f'{float(x):.6f}'
            interval=f(z['fnr_ci_lo'])+' / '+f(z['fnr_ci_hi'])
            lines.append(f"| {z['cohort']} | {z['evaluation']} | {z['method']} | {z['source_profile']} | "+
              f"{z['evaluated_profile']} | {z.get('model','ensemble')} | {z['alpha']:.2f} | "+
              f"{f(z['auroc'])} | {f(z['suppression'])} | {f(z['fnr'])} | {interval} | {z['status']} |")
    metrics_table(pd.read_parquet(dest/'profile_crc.parquet'),'## (a) Cross-profile CRC and (b) Mondrian versus frozen marginal')
    metrics_table(result,'## (c) Single-profile supervised training, five seeds plus ensemble')
    lines.extend(['','All result cells with small class counts are withheld.',
                  'No data-based model, profile, alpha or threshold reselection was performed.',
                  'Test and C2015 are POST-HOC; no validated vendor/site labels are available.',
                  'WAIT_FOR_DIRECTOR.',''])
    # Build complete report in RAM before writing either final artifact.
    result.to_parquet(newpath,index=False)
    with reportpath.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    files.append((reportpath,0));append_manifest(project,files,'P05-T06b')
    print('=== REPORT P05-T06b ===',flush=True)
    print('status: COMPLETE; WAIT_FOR_DIRECTOR; pc=reference-workstation; GPU1 used for train/inference',flush=True)
    print(f'script_sha256_8: {sha(__file__)[:8]}; frozen sources SHA PASS',flush=True)
    print('P1_P2_FRACTION_GATE=PASS; PROFILE_HEADER_ONLY=PASS',flush=True)
    print('a: cross-profile CRC PASS; b: Mondrian profile PASS',flush=True)
    print('c: 10 profile-only training runs plus POST-HOC cross-profile evaluation PASS',flush=True)
    print('d: profile-level modality availability PASS',flush=True)
    print('outputs: '+str(dest)+'; report: '+str(reportpath),flush=True)
    print('warnings: test/C2015 POST-HOC; T05b/T06 original files untouched',flush=True)
    print('=== END ===',flush=True)


def main():
    started=time.monotonic()
    a=argparse.ArgumentParser();a.add_argument('--stage',required=True,choices=['audit','train','report'])
    args=a.parse_args();project=Path(__file__).resolve().parents[1]
    c,lock,proc,core,conformal,metrics=context(project)
    if args.stage=='audit':
        # t02c_pilot imported in context sets GPU visibility; reset for CPU-only audit.
        os.environ['CUDA_VISIBLE_DEVICES']=''
        must('torch' not in sys.modules,'CPU audit unexpectedly imported torch')
        audit(project,c,lock,proc,conformal,metrics)
    elif args.stage=='train':
        # The module sets GPU1 prior to importing torch, regardless of caller shell state.
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        train(project,c,lock,proc,core,conformal,metrics)
    else:
        os.environ['CUDA_VISIBLE_DEVICES']='1'
        report(project,c,lock,proc,core,conformal,metrics)
    print(f'P05_T06B_{args.stage.upper()}_EXIT=PASS; runtime_s={time.monotonic()-started:.3f}',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('P05_T06B_ERROR='+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        raise
