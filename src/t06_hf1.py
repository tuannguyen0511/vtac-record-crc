"""P05-T06-HF1: preregistered methodological correction. CPU-only.

Reads T03 calibration/validation scores, T06 approved summary sources and
T05b aggregate results. Never reloads sealed test labels, writes to new paths.
"""
import release_paths
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for env in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[env] = '8'
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta

EXPECTED_T06_BRANCH_SHA = 'bfec07cbaba13428a0771655f35265057743ba4c8c6dc290010c1df40a7d7d56'
EXPECTED_FINAL_LOCK_SHA = 'c09b6cd952319950f65b57e64086ab691946ccabac84adb106efb2bc765299c8'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()


def require(ok,msg):
    if not ok: raise RuntimeError('T06_HF1_STOP: '+msg)


def q(a):
    return [float(x) for x in np.percentile(np.asarray(a,dtype=float),[2.5,50,97.5])]


def cp_upper_975(k,n):
    require(n>0 and 0<=k<=n,'invalid k/n')
    return 1.0 if k==n else float(beta.ppf(.975,k+1,n-k))


def aggregate_counts(n,rate):
    n=int(n); r=float(rate); k=int(round(n*r))
    require(0<=k<=n and abs(k/n-r)<1e-8,'T05b aggregate FNR cannot recover integral count')
    return k,n


def record_members(y,records):
    _,inverse=np.unique(np.asarray(records,dtype=str),return_inverse=True)
    members=[np.flatnonzero(inverse==j) for j in range(int(inverse.max())+1)]
    positive=[indices for indices in members if np.any(y[indices]==1)]
    negative=[indices for indices in members if not np.any(y[indices]==1)]
    return positive,negative


def resampling(calib,val,conformal,old):
    cp,cy=calib['prob'],calib['y']
    vp,vy=val['prob'],val['y']
    pos,neg=record_members(cy,calib['records'])
    require(len(pos)==188 and len(pos)+len(neg)==543,'calibration positive/total record mismatch')
    rng=np.random.default_rng(42)
    models=['ensemble']+[f'seed{i}' for i in range(5)]
    rows=[]
    for npos in (50,100,150,188):
        nneg=int(round(npos*len(neg)/len(pos)))
        for rep in range(1000):
            selected=[pos[j] for j in rng.integers(len(pos),size=npos)]
            selected += [neg[j] for j in rng.integers(len(neg),size=nneg)]
            ii=np.concatenate(selected)
            # Each independent with-replacement copy must have its OWN cluster ID.
            cloned_group=np.repeat(np.arange(len(selected),dtype=np.int64),
                                   [len(z) for z in selected])
            for model in models:
                seed=None if model=='ensemble' else int(model[-1])
                pcal=cp if seed is None else calib['seeds'][seed]
                pval=vp if seed is None else val['seeds'][seed]
                lam=float(conformal.crc_fnr_threshold(pcal[ii],cy[ii],cloned_group,.02))
                suppress,fnr=old.rates(pval,vy,lam)
                rows.append({'n_pos':npos,'n_neg':nneg,'model':model,'rep':rep,
                             'lambda':lam,'suppression':suppress,'fnr':fnr})
        print(f'HF1_A1_BOOTSTRAP=PASS; n_pos={npos}; replicates=1000; models=6',flush=True)
    frame=pd.DataFrame(rows)
    summary=[]
    for (n,model),g in frame.groupby(['n_pos','model'],sort=True):
        summary.append({'n_pos':int(n),'model':model,'replicates':int(len(g)),
                        'lambda_q025_q50_q975':q(g['lambda']),
                        'supp_q025_q50_q975':q(g['suppression']),
                        'fnr_q025_q50_q975':q(g['fnr']),
                        'p_supp_lt_030':float((g['suppression']<.30).mean())})
    return frame,summary


def a3_fixed(val,project,lock,old):
    # Do not load annotation decisions for ANY test events.
    keys=set(zip(val['meta']['record'].astype(str),val['meta']['event'].astype(str)))
    votes=old.gated_votes(release_paths.get("data_raw")/'vtac'/'1.1'/'event_label_per_annotator.csv',keys)
    rng=np.random.default_rng(42)
    lam=float(lock['lambda']['A'][.02])
    require(abs(lam-.0139079025)<1e-12,'fixed A lambda modified')
    rows=[]
    for rep in range(1000):
        yd=old.sample_annotation_labels(val['meta'],votes,rng)
        supp,fnr=old.rates(val['prob'],yd,lam)
        rows.append({'rep':rep,'fnr_sampled':fnr,'suppression_sampled':supp})
    f=pd.DataFrame(rows)
    require(f['fnr_sampled'].notna().all() and f['suppression_sampled'].notna().all(),
            'a sampled validation class is absent')
    return f,{'n_replicates':1000,'seed':42,'locked_lambda':lam,
              'fnr_q025_q50_q975':q(f['fnr_sampled']),
              'supp_q025_q50_q975':q(f['suppression_sampled']),
              'p_fnr_gt_002':float((f['fnr_sampled']>.02).mean())}


def a2_multi_positive(calib):
    # Positive RECORDS are defined by at least one y=1, but their modality
    # diversity is counted over ALL events in each such record.
    positive=set(np.asarray(calib['records'])[np.asarray(calib['y'])==1])
    record_strata={}
    for rec,s in zip(calib['records'],calib['strata']):
        if rec in positive: record_strata.setdefault(rec,set()).add(str(s))
    n=sum(len(s)>=2 for s in record_strata.values())
    return '<10' if n<10 else str(n)


def a5_aggregate(project,lock):
    src=project/'outputs'/'t05b'/'summary.json'
    require(src.is_file(),'frozen T05b summary.json is absent')
    report=json.loads(src.read_text(encoding='utf-8'))
    require(report['final_lock_sha256']==EXPECTED_FINAL_LOCK_SHA,'T05b final lock differs')
    records=[]
    methods=[('0.01','CRC 0.01'),('0.02','CRC 0.02'),('0.05','CRC 0.05')]
    cohorts=(('test',.286),('c2015',.261))
    for cohort,prev in cohorts:
        if cohort=='test':
            variants=report['test']['variants']['A']['alphas']
            n=report['test']['ltt_at_tau_0_09']['n_true']
            packed=[(display,variants[k]['fnr'],variants[k]['n_true'])
                    for k,display in methods]
            packed.append(('LTT',report['test']['ltt_at_tau_0_09']['fnr'],n))
        else:
            vt=[v for v in report['external']['results'] if v['subset']=='all']
            packed=[]
            for k,display in methods:
                found=[v for v in vt if v['alpha'] == float(k)]
                require(len(found)==1 and found[0]['status']=='OK','C2015 method missing')
                packed.append((display,found[0]['fnr'],found[0]['true']))
            found=[v for v in vt if v['alpha']=='LTT_tau_0.09']
            require(len(found)==1 and found[0]['status']=='OK','C2015 LTT absent')
            packed.append(('LTT',found[0]['fnr'],report['external']['positive_vt']))
        for method,fnr,denom in packed:
            k,n=aggregate_counts(denom,fnr)
            upper=cp_upper_975(k,n)
            records.append({'cohort':cohort,'evaluation':'POST-HOC','method':method,
                            'k_true_suppressed':k,'n_true':n,'fnr':k/n,
                            'one_sided_cp_upper_975':upper,
                            'upper_true_blocked_per_1000':1000*prev*upper,
                            'assumed_prevalence':prev})
    return records


def make_figure(summary,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5.0),dpi=160)
    cmap=plt.get_cmap('tab10')
    for i,model in enumerate(['ensemble']+[f'seed{i}' for i in range(5)]):
        a=sorted([z for z in summary if z['model']==model],key=lambda z:z['n_pos'])
        x=np.asarray([v['n_pos'] for v in a])
        med=np.asarray([v['supp_q025_q50_q975'][1] for v in a])
        lo=np.asarray([v['supp_q025_q50_q975'][0] for v in a])
        hi=np.asarray([v['supp_q025_q50_q975'][2] for v in a])
        color=cmap(i)
        ax.plot(x,med,marker='o',lw=2 if i==0 else 1,alpha=1 if i==0 else .6,label=model,color=color)
        ax.fill_between(x,lo,hi,alpha=.18 if i==0 else .04,color=color)
    ax.axhline(.30,ls=':',color='gray',label='Suppression 0.30')
    ax.set(xlabel='Positive calibration records, with-replacement bootstrap',
           ylabel='Validation false-alarm suppression',ylim=(-.02,1.02),
           title='P05-T06-HF1: m-out-of-n record bootstrap')
    ax.grid(alpha=.15);ax.legend(ncol=2,fontsize=8)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def main():
    start=time.monotonic();project=Path(__file__).resolve().parents[1]
    reportpath=project/'reports'/'P05-T06-HF1.md'
    out=project/'outputs'/'t06_hf1'
    require(not out.exists() and not reportpath.exists(),'existing HF1 outputs; refuse overwrite')
    require((project/'reports'/'TEST_OPENED.flag').is_file(),'accepted T05b flag absent')
    require((project/'reports'/'P05-T06.md').is_file(),'accepted T06 report absent')
    require(sha(project/'configs'/'final_lock.yaml')==EXPECTED_FINAL_LOCK_SHA,'final lock SHA mismatch')
    require(sha(project/'src'/'t06_branch_a.py')==EXPECTED_T06_BRANCH_SHA,'T06 accepted code SHA mismatch')
    lock=yaml.safe_load((project/'configs'/'final_lock.yaml').read_text(encoding='utf-8'))
    sys.path.insert(0,str(project/'src'))
    import t06_branch_a as old
    import t02c_pilot as gpu_pilot
    os.environ['CUDA_VISIBLE_DEVICES']=''
    gpu_pilot.below_normal_priority()
    require('torch' not in sys.modules,'HF1 imported torch')
    common=release_paths.get("repo") / "common"
    require(gpu_pilot.common_source_sha256(common)==lock['frozen_sha256']['common_tree'],
            'common SHA changed')
    sys.path.insert(0,str(project.parent))
    from common import conformal
    # This only reads calibration and validation. Test is sourced from T05b aggregates.
    data={};proc=release_paths.get("data_proc")
    for split in ('calib','val'):
        meta=pd.read_parquet(proc/f'{split}_meta.parquet')
        mask=np.load(proc/f'{split}_W10_D0_mask.npy',allow_pickle=False)
        require(mask.shape==(len(meta),4),'bad modality mask')
        require({'idx','record','event','y','votes_missing'}.issubset(meta.columns),'missing meta')
        ix=pd.Index(meta['idx'])
        logits=[]
        for seed in range(5):
            f=project/'outputs'/'t03'/f'A_s{seed}'/f'scores_{split}.parquet'
            t=pd.read_parquet(f).set_index('idx')['logit']
            require(t.index.is_unique and ix.isin(t.index).all(),'T03 idx mismatch')
            logits.append(t.reindex(ix).to_numpy(dtype=float))
        z=np.stack(logits)
        require(np.isfinite(z).all(),'nonfinite T03 scores')
        data[split]={'meta':meta,'y':meta['y'].to_numpy(dtype=np.int8),
                     'records':meta['record'].astype(str).to_numpy(),
                     'strata':old.stratum_from_mask(mask),
                     'prob':old.sigmoid(z.mean(axis=0)),
                     'seeds':[old.sigmoid(z[i]) for i in range(5)]}
    frame,summary=resampling(data['calib'],data['val'],conformal,old)
    annot,annot_summary=a3_fixed(data['val'],project,lock,old)
    multi=a2_multi_positive(data['calib'])
    a5=a5_aggregate(project,lock)
    require(len(a5)==8,'A5 expects exactly eight post-hoc rows')
    row=[next(z for z in a5 if z['cohort']==cohort and z['method']=='CRC 0.01')
         for cohort in ('test','c2015')]
    require(row[0]['n_true']==137 and row[1]['n_true']==89,
            'Director expected test/C2015 positive count does not match frozen T05b')
    print('HF1_A5_CP_UPPER_POSTHOC_TEST_0_01=',row[0]['upper_true_blocked_per_1000'],flush=True)
    print('HF1_A5_CP_UPPER_POSTHOC_C2015_0_01=',row[1]['upper_true_blocked_per_1000'],flush=True)
    out.mkdir(parents=True,exist_ok=False)
    files=[]
    for fname,records in (('A1_bootstrap.parquet',frame),('A3_fixed_lambda.parquet',annot),
                          ('A5_one_sided_cp.parquet',pd.DataFrame(a5))):
        path=out/fname;records.to_parquet(path,index=False);files.append((path,len(records)))
    fpath=out/'figs_A1_bootstrap.png';make_figure(summary,fpath);files.append((fpath,1))
    text=['# P05-T06-HF1: Director-authorized correction (12 November 2026)','',
          'Status: COMPLETE; T06 and T05b original artifacts UNCHANGED.','',
          '## A1b. m-out-of-n positive-record bootstrap with replacement','',
          'Records are selected independently with replacement within positive and negative calibration classes.',
          'Every duplicated positive/negative record receives a distinct bootstrap group ID.',
          'Alpha=0.02, 1000 replicates per n_pos, seed=42; validation metrics only.','',
          '| Positive n | Model | Lambda q2.5/50/97.5 | Suppression q2.5/50/97.5 | FNR q2.5/50/97.5 | P(supp<0.30) |',
          '|---:|---|---|---|---|---:|']
    for s in summary:
        text.append(f"| {s['n_pos']} | {s['model']} | {s['lambda_q025_q50_q975']} | {s['supp_q025_q50_q975']} | {s['fnr_q025_q50_q975']} | {s['p_supp_lt_030']:.5f} |")
    pts=[z for z in summary if z['model']=='ensemble']
    pts.sort(key=lambda r:r['n_pos']);p=[z['p_supp_lt_030'] for z in pts]
    text+=['','Director 300-draw reference is descriptive only: 0.37/0.21/0.18/0.13.','Observed 1000-draw probabilities: '+str(p)+'.',
           'Non-monotonic probability fluctuations are report-only, not a failure.','',
           '## A3. Fixed lambda and randomly sampled validation labels','',
           'Validation-only; lambda=0.0139079025 frozen; each draw samples one non-adjudicating vote.','',
           'FNR q2.5/50/97.5: '+str(annot_summary['fnr_q025_q50_q975']),
           'P(FNR>0.02): '+str(annot_summary['p_fnr_gt_002']),
           'Suppression q2.5/50/97.5: '+str(annot_summary['supp_q025_q50_q975']),
           '', '## A2. Record-level multi-stratum positives','',
           'Calib positive records spanning at least two modality strata: '+multi,
           '', '## A5. One-sided Clopper-Pearson bounds (POST-HOC)','',
           'The estimates below are derived exclusively from the approved T05b aggregate FNR and denominators.',
           'Upper limits use a one-sided 97.5% Clopper-Pearson bound, scaled by the prespecified prevalence.',
           '', '| Cohort | Method | k/n true VT suppressed | FNR | CP upper 97.5% | At most X per 1000 VT alarms |',
           '|---|---|---:|---:|---:|---:|']
    for z in a5:
        text.append(f"| {z['cohort']} POST-HOC | {z['method']} | {z['k_true_suppressed']}/{z['n_true']} | {z['fnr']:.6f} | {z['one_sided_cp_upper_975']:.7f} | {z['upper_true_blocked_per_1000']:.2f} |")
    text +=['','Privacy: aggregate k/n are the Director-approved exception. All other small counts are redacted.',
            'No patient identifiers or waveforms are reported. Test and C2015 are POST-HOC.',
            'Original T06/T05b outputs remain unchanged.','']
    with reportpath.open('x',encoding='utf-8') as f:f.write('\n'.join(text))
    files.append((reportpath,0))
    with (project/'manifest.csv').open('a',encoding='utf-8',newline='') as f:
        w=csv.writer(f);now=datetime.now(timezone.utc).isoformat()
        for path,n in files:w.writerow([str(path),sha(path),n,now,'P05-T06-HF1'])
    print('=== REPORT P05-T06-HF1 ===',flush=True)
    print('status: COMPLETE; WAIT_FOR_DIRECTOR; reference-workstation; CPU<=8; BelowNormal; GPU=NONE',flush=True)
    print(f'runtime_s: {time.monotonic()-start:.3f}; script_sha256_8: {sha(__file__)[:8]}',flush=True)
    print('A1b: m-out-of-n with replacement; models=6; replicates=1000 per n; PASS',flush=True)
    print('A1_P_SUPP_LT_0_30_ENSEMBLE='+str(p),flush=True)
    print('A3: fixed lambda label sampling on validation only; PASS',flush=True)
    print('A2: multi-stratum positive calib record counts reported; PASS',flush=True)
    print('A5: eight POST-HOC one-sided CP bounds from frozen T05b aggregate; PASS',flush=True)
    print('outputs: '+str(out)+'; report: '+str(reportpath),flush=True)
    print('warnings: T06 originals preserved; T05b preserved; test/C2015 POST-HOC',flush=True)
    print('=== END ===',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('T06_HF1_STOP='+type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True)
        raise
