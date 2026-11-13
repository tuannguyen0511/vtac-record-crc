"""P05-T07. Director-approved presubmission POST-HOC analyses E3/E5/E7/E8; E4 optional.

One script, independent --only sections. Never train, alter locked thresholds,
re-run T05b, or write outside outputs/t07 and reports/P05-T07.md.
Test/C2015 results always carry evaluation=POST-HOC.
"""
import release_paths
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # used exclusively by E7
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'outputs', 't07', 'mplconfig')
os.environ['CUDA_CACHE_PATH'] = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'outputs', 't07', 'cuda_cache')

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
sys.dont_write_bytecode = True
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import beta

LOCK_SHA256 = 'c09b6cd952319950f65b57e64086ab691946ccabac84adb106efb2bc765299c8'
T06_A_SHA256 = 'bfec07cbaba13428a0771655f35265057743ba4c8c6dc290010c1df40a7d7d56'
ALPHAS = (0.01,0.02,0.05)
REQUIRED = ('E3','E5','E7','E8')
SPLITS = ('calib','val','test','c2015')
DIR = Path(__file__).resolve().parents[1]
OUT = DIR/'outputs'/'t07'
REPORT = DIR/'reports'/'P05-T07.md'
ROOT = release_paths.get("repo")
PROC = release_paths.get("data_proc")
RAW = release_paths.get("data_raw")/'vtac'/'1.1'/'annotated_waveforms'


def require(ok, reason):
    if not ok:
        raise RuntimeError('P05_T07_STOP: '+str(reason))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save_json(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)


def redact(n):
    return '<10' if int(n)<10 else str(int(n))


def number(x,digits=6):
    if x is None or not np.isfinite(x):return 'N/A'
    return f'{float(x):.{digits}f}'


def safe_fraction(numer,denom):
    return None if int(denom)==0 else float(numer)/int(denom)


def cp_95(k,n):
    if n==0:return (None,None)
    return (float(beta.ppf(.025,k,n-k+1)) if k>0 else 0.0,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.0)


def blocked_rates(prob,y,lam,records=None):
    prob=np.asarray(prob,float);y=np.asarray(y,dtype=np.int8)
    require(len(prob)==len(y) and np.isfinite(prob).all(),'rates input mismatched')
    suppressed=prob<float(lam)
    pos=y==1;neg=y==0
    k=int((suppressed&pos).sum());n=int(pos.sum());nneg=int(neg.sum())
    ans={'n_true':n,'n_false':nneg,'fnr_k':k,'fnr_n':n,
         'fnr':safe_fraction(k,n), 'fnr_ci95_lo':cp_95(k,n)[0],
         'fnr_ci95_hi':cp_95(k,n)[1],
         'false_suppressed':int((suppressed&neg).sum()),
         'suppression':safe_fraction(int((suppressed&neg).sum()),nneg)}
    if records is not None:
        rec=np.asarray(records,dtype=str)
        require(len(rec)==len(y),'record group misaligned')
        per=[]
        for record in np.unique(rec[pos]):
            indices=pos&(rec==record)
            per.append(float(suppressed[indices].mean()))
        ans['positive_records']=len(per)
        ans['record_mean_fnr']=float(np.mean(per)) if per else None
    return ans


def summary_path(section):return OUT/f'{section}_summary.json'


def all_files():
    return sorted((p for p in OUT.rglob('*') if p.is_file()),key=lambda p:p.as_posix())


def preflight():
    require((DIR/'reports'/'TEST_OPENED.flag').is_file(),'T05b flag missing')
    require((DIR/'reports'/'P05-T06.md').is_file(),'accepted T06 report missing')
    require((DIR/'reports'/'P05-T06b.md').is_file(),'accepted T06b report missing')
    require(sha(DIR/'configs'/'final_lock.yaml')==LOCK_SHA256,'final lock SHA mismatch')
    require(sha(DIR/'src'/'t06_branch_a.py')==T06_A_SHA256,'accepted T06 source SHA mismatch')
    lock=yaml.safe_load((DIR/'configs'/'final_lock.yaml').read_text(encoding='utf-8'))
    require(lock['primary']['variant']=='A' and lock['primary']['alpha']==.02,
            'primary lock variant/alpha mismatch')
    for key,file in [('models_py','models.py'),('train_lib_py','train_lib.py'),
                     ('t03_label_variants_py','t03_label_variants.py'),
                     ('t05a_conformal_ablation_py','t05a_conformal_ablation.py')]:
        require(sha(DIR/'src'/file)==lock['frozen_sha256'][key],
                'frozen file changed: '+file)
    require(sha(PROC/'test_labels.parquet')==lock['frozen_sha256']['test_labels_parquet'],
            'test labels SHA mismatch')
    for s in range(5):
        ck=DIR/'outputs'/'t03'/f'A_s{s}'/'ckpt.pt'
        require(sha(ck)==lock['checkpoints_sha256'][f'A_s{s}'],
                'A checkpoint changed')
    sys.path.insert(0,str(DIR/'src'))
    sys.path.insert(0,str(ROOT))
    import t02c_pilot as pilot
    pilot.below_normal_priority()
    require(pilot.common_source_sha256(ROOT/'common')==lock['frozen_sha256']['common_tree'],
            'common SHA mismatch')
    from common import conformal,metrics
    require((DIR/'outputs'/'t06'/'step0_manifest.json').is_file(),
            'accepted step0 score manifest missing')
    print('P05_T07_PREFLIGHT=PASS; WINDOWS_PROCESS_PRIORITY=BelowNormal',flush=True)
    return lock,pilot,conformal,metrics


def get_data(lock,conformal,metrics):
    import t06_branch_a as source
    data=source.source_data(DIR,{'outputs_subdir':'outputs/t06',
            'director_final_lock_sha256':LOCK_SHA256,
            'expected_auroc':{'test':.946049,'c2015':.942304}},
            lock,conformal,metrics)
    for split in ('calib','val','test'):
        meta=pd.read_parquet(PROC/f'{split}_meta.parquet')
        require(len(meta)==len(data[split]['y']) and meta['idx'].is_unique,
                split+' metadata invalid')
        data[split]['meta']=meta
    return data


def e3(data,lock,conformal):
    print('T07_E3_START',flush=True)
    cal=data['calib']; n=int(np.count_nonzero(cal['y']==1))
    require(n>0,'no positive calibration events')
    groups=np.arange(len(cal['y']),dtype=np.int64)
    rows=[];lambdas=[]
    for alpha in ALPHAS:
        lam_event=float(conformal.crc_fnr_threshold(cal['prob'],cal['y'],groups,alpha))
        lam_record=float(lock['lambda']['A'][alpha])
        record_calc=float(conformal.crc_fnr_threshold(cal['prob'],cal['y'],cal['records'],alpha))
        require(abs(record_calc-lam_record)<=1e-9,'locked record-level CRC mismatch')
        # The +1 correction is for positive events, not negative alarms.
        losses=int(np.count_nonzero((cal['y']==1)&(cal['prob']<lam_event)))
        require((losses+1)/(n+1)<=alpha+1e-12,
                'event-level CRC inequality violated')
        lambdas.append({'alpha':alpha,'n_positive_events_calib':n,
                        'lambda_event':lam_event,'lambda_record':lam_record,
                        'event_calib_misses':losses})
        for method,lam in (('event_CRC',lam_event),('record_CRC_locked',lam_record)):
            for split in ('val','test','c2015'):
                d=data[split];r=blocked_rates(d['prob'],d['y'],lam,d['records'])
                rows.append({'cohort':split,
                             'evaluation':'POST-HOC' if split in ('test','c2015') else 'VALIDATION',
                             'method':method,'alpha':alpha,'lambda':lam,**r})
    pval=next(z for z in rows if z['method']=='record_CRC_locked'
              and z['cohort']=='val' and z['alpha']==.02)
    require(abs(pval['suppression']-.596)<=.001,
            f'E3 val suppression gate FAILED: {pval["suppression"]:.9f} vs 0.596')
    require(pval['fnr_k']==2 and pval['fnr_n']==141,
            'E3 val primary FNR differs from Director 2/141')
    pd.DataFrame(rows).to_parquet(OUT/'E3_crc_event_vs_record.parquet',index=False)
    pd.DataFrame(lambdas).to_parquet(OUT/'E3_thresholds.parquet',index=False)
    ans={'status':'PASS','n_positive_calib_events':n,'lambdas':lambdas,
         'val_record_suppression_at_002':pval['suppression'],
         'val_record_fnr_at_002':'2/141'}
    save_json(summary_path('E3'),ans)
    print('T07_E3=PASS; val_record_suppression='+number(pval['suppression'])+
          '; val_record_FNR=2/141',flush=True)
    return ans


def external_vt():
    from t05b_external_c2015 import external_root
    # No external model inference: only use the frozen public ALARMS ordering.
    cfg={'project_dir':str(DIR)}
    folder=external_root(cfg)
    with (folder/'ALARMS').open('r',encoding='utf-8',newline='') as f:
        rows=list(csv.reader(f))
    vt=[(r[0],int(r[2])) for r in rows if len(r)==3 and r[1]=='Ventricular_Tachycardia']
    require(len(vt)==341 and sum(y for _,y in vt)==89,
            'external VT order/count differs')
    return folder,vt


def normalized(prep,name):
    return str(prep.normalized_channel(name)).strip().upper()


def selected_channels(names,prep):
    names=[normalized(prep,s) for s in names]
    first=next((names.index(q) for q in ('II','I','III','AVR','AVL','AVF')
                if q in names),None)
    other_pref=('V','MCL','V1','V2','V3','V4','V5','V6','II','I','III','AVR','AVL','AVF')
    second=next((names.index(q) for q in other_pref if q in names and names.index(q)!=first),None)
    return first,second


def read_raw60(base,wfdb):
    hdr=wfdb.rdheader(str(base))
    require(abs(float(hdr.fs)-250)<1e-9 and int(hdr.sig_len)>=75000,
            'WFDB header fs/length differs')
    x,meta=wfdb.rdsamp(str(base),sampfrom=60000,sampto=75000)
    require(x.shape[0]==15000 and len(meta['sig_name'])==x.shape[1],
            'WFDB prealarm 60s sampling failure')
    return x,list(meta['sig_name']),hdr


def beats_150ms(one,two,tolerance=37):
    # One-to-one sorted peak matching, 37 samples=148 ms at 250 Hz.
    a=np.sort(np.asarray(one,dtype=int));b=np.sort(np.asarray(two,dtype=int))
    i=j=match=0
    while i<len(a) and j<len(b):
        delta=int(a[i])-int(b[j])
        if abs(delta)<=tolerance:
            match+=1;i+=1;j+=1
        elif delta<0:i+=1
        else:j+=1
    return match


def bsqi_one(wave,wfproc):
    wave=np.asarray(wave,dtype=np.float64)
    if wave.shape!=(2500,) or not np.isfinite(wave).all() or float(wave.std())<1e-6:
        return 0.0,'INVALID_WAVEFORM'
    try:
        q=wfproc.xqrs_detect(sig=wave,fs=250,verbose=False)
        g=wfproc.gqrs_detect(sig=wave,fs=250)
        q=np.asarray(q,dtype=int);g=np.asarray(g,dtype=int)
        if len(q)==0 or len(g)==0:
            return 0.0,'NO_QRS_BOTH_DETECTORS'
        matched=beats_150ms(q,g)
        return float(np.clip(2*matched/(len(q)+len(g)),0.0,1.0)),'OK'
    except (ValueError,IndexError,RuntimeError,FloatingPointError) as exc:
        return 0.0,'QRS_DETECTOR_FAILURE'


def sqi_from_signal(raw,names,cached_x,mask,prep,wfproc):
    # Raw NaN and flat fraction refer to the original [-10s,0) ECG1 samples.
    first,second=selected_channels(names,prep)
    if first is None:
        nan_fraction=1.0;flat_fraction=1.0
    else:
        s=np.asarray(raw[-2500:,first],dtype=np.float64)
        nan_fraction=float((~np.isfinite(s)).mean())
        adjacent=np.isfinite(s[1:])&np.isfinite(s[:-1])
        flat_fraction=float((adjacent&(np.abs(np.diff(s))<1e-6)).sum()/(len(s)-1))
    quality=[];reasons=[]
    for k,idx in ((0,first),(1,second)):
        if idx is None or not bool(mask[k]):continue
        q,status=bsqi_one(cached_x[k],wfproc)
        quality.append(q);reasons.append(status)
    score=max(quality) if quality else 0.0
    return {'nan_ecg1':nan_fraction,'flat_ecg1':flat_fraction,
            'bsqi':float(score),'qrs_status':('OK' if 'OK' in reasons else
                (reasons[0] if reasons else 'NO_USABLE_ECG'))}


def compute_sqi(data,only=('calib','val','test','c2015')):
    import wfdb
    from wfdb import processing as wfproc
    import preprocess_lib as prep
    folder=None;vt=None
    if 'c2015' in only:folder,vt=external_vt()
    rawout=[];items={}
    for split in only:
        d=data[split]
        meta=d.get('meta')
        if split!='c2015':
            require(meta is not None and {'record','event','idx'}.issubset(meta.columns),
                    split+' meta schema')
            paths=[RAW/str(rec)/str(ev) for rec,ev in meta[['record','event']].itertuples(index=False,name=None)]
            x_all=np.load(PROC/f'{split}_W10_D0_x.npy',mmap_mode='r',allow_pickle=False)
            m_all=np.load(PROC/f'{split}_W10_D0_mask.npy',mmap_mode='r',allow_pickle=False)
            require(x_all.shape==(len(meta),4,2500) and m_all.shape==(len(meta),4),
                    'cached preprocessing shapes mismatch: '+split)
        else:
            paths=[folder/name for name,y in vt]
            x_all=m_all=None
        per=[]
        for i,base in enumerate(paths):
            raw,names,_=read_raw60(base,wfdb)
            if split=='c2015':
                x,mask=prep.preprocess_event(raw,names,10,0,fs=250,t0=75000,start_sample=60000)
            else:
                x,mask=x_all[i],m_all[i]
            q=sqi_from_signal(raw,names,np.asarray(x),np.asarray(mask),prep,wfproc)
            per.append({'idx':int(i),'cohort':split,
                        'evaluation':'POST-HOC' if split in ('test','c2015') else
                                     ('CALIBRATION' if split=='calib' else 'VALIDATION'),**q})
            if (i+1)%250==0:
                print(f'T07_E5_SQI_PROGRESS={split}; completed={i+1}/{len(paths)}',flush=True)
        frame=pd.DataFrame(per)
        require(len(frame)==len(d['y']) and frame['idx'].is_unique,
                'SQI alignment error')
        items[split]=frame
        print(f'T07_E5_SQI_PASS={split}; events={len(frame)}',flush=True)
    return items


def tertile_code(bsqi,edges):
    # Fixed calibration thresholds; ties assigned to the lower tertile.
    return np.searchsorted(np.asarray(edges,dtype=float),np.asarray(bsqi,dtype=float),side='left')+1


def e5(data,lock):
    print('T07_E5_START',flush=True)
    scores=compute_sqi(data)
    cuts=np.quantile(scores['calib']['bsqi'].to_numpy(dtype=float),[1/3,2/3])
    require(np.isfinite(cuts).all() and 0<=cuts[0]<=cuts[1]<=1,
            'calibration bSQI thresholds invalid')
    rows=[];qualities=[]
    lam=float(lock['lambda']['A'][.02])
    for split in SPLITS:
        q=scores[split].copy()
        q['tertile']=tertile_code(q['bsqi'],cuts)
        q['y']=data[split]['y']
        q['suppressed']=np.asarray(data[split]['prob'])<lam
        qualities.extend(q[['cohort','evaluation','idx','nan_ecg1','flat_ecg1',
                            'bsqi','qrs_status','tertile']].to_dict('records'))
        for tertile in (1,2,3):
            use=(q['tertile']==tertile).to_numpy()
            row=blocked_rates(data[split]['prob'][use],data[split]['y'][use],lam)
            rows.append({'cohort':split,'evaluation':'POST-HOC' if split in ('test','c2015') else
                         ('CALIBRATION' if split=='calib' else 'VALIDATION'),
                         'tertile':tertile,'bsqi_cut_1':float(cuts[0]),
                         'bsqi_cut_2':float(cuts[1]),'n_events':int(use.sum()),**row})
        require(sum(r['n_events'] for r in rows if r['cohort']==split)==len(q),
                'E5 tertile counts do not sum to dataset size: '+split)
        if split=='val':
            full=blocked_rates(data['val']['prob'],data['val']['y'],lam)
            require((full['fnr_k'],full['fnr_n'])==(2,141),
                    'E5 val FNR differs from Director 2/141')
    # Logistic regression is fit ONLY on true VT, with no test- or external-driven fitting.
    logistic=[]
    from sklearn.linear_model import LogisticRegression
    for split in ('val','test','c2015'):
        q=scores[split]
        y=np.asarray(data[split]['y'],dtype=np.int8)
        caught=(np.asarray(data[split]['prob'])<lam)&(y==1)
        count=int(caught.sum())
        if count<3:
            logistic.append({'cohort':split,
                             'evaluation':'POST-HOC' if split!='val' else 'VALIDATION',
                             'missed_true_vt':count,'status':'INSUFFICIENT_EVENTS_LT3',
                             'beta_bsqi':None,'odds_ratio_bsqi':None})
            continue
        yy=caught[y==1].astype(int)
        x=q.loc[y==1,'bsqi'].to_numpy(dtype=float).reshape(-1,1)
        if len(np.unique(yy))<2 or len(np.unique(x))<2:
            logistic.append({'cohort':split,
                             'evaluation':'POST-HOC' if split!='val' else 'VALIDATION',
                             'missed_true_vt':count,'status':'DEGENERATE_DATA',
                             'beta_bsqi':None,'odds_ratio_bsqi':None})
            continue
        lr=LogisticRegression(penalty=None,solver='lbfgs',max_iter=1000,random_state=42)
        lr.fit(x,yy)
        coefficient=float(lr.coef_[0,0])
        logistic.append({'cohort':split,
                         'evaluation':'POST-HOC' if split!='val' else 'VALIDATION',
                         'missed_true_vt':count,'status':'DESCRIPTIVE_ONLY',
                         'beta_bsqi':coefficient,
                         'odds_ratio_bsqi':float(np.exp(np.clip(coefficient,-700,700)))})
    pd.DataFrame(qualities).to_parquet(OUT/'E5_signal_quality.parquet',index=False)
    pd.DataFrame(rows).to_parquet(OUT/'E5_tertiles.parquet',index=False)
    pd.DataFrame(logistic).to_parquet(OUT/'E5_logistic.parquet',index=False)
    ans={'status':'PASS','bsqi_cutpoints':[float(v) for v in cuts],
         'n_by_split':{k:int(len(scores[k])) for k in SPLITS},
         'tertile_rows':len(rows),'logistic':logistic,
         'val_total_fnr':'2/141',
         'ties_policy':'bSQI exactly equal to a tertile threshold goes into lower tertile',
         'unscorable_bsqi_policy':'bSQI=0, QRS status recorded; never excluded',
         'method':'bSQI=2*matched/(n_xqrs+n_gqrs), 150ms one-to-one matching; ECG2 max'}
    save_json(summary_path('E5'),ans)
    print('T07_E5=PASS; tertile_totals=PASS; val_fnr=2/141; cuts='+str(ans['bsqi_cutpoints']),flush=True)
    return ans


def cpu_name():
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            name=winreg.QueryValueEx(key,'ProcessorNameString')[0]
            if str(name).strip():return str(name).strip()
    except (OSError,ImportError,ValueError):
        pass
    return os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor() or 'NOT_REPORTED'


def model_logit(net,x,mask):
    """Replicate frozen train_lib ResNet1D input geometry for latency only.

    The locked network's first Conv1d expects 8 channels: four waveform slots
    followed by four per-slot presence-mask channels broadcast along time.
    The E7 caller validates all five resulting logits against archived T03
    validation logits before any latency measurement is accepted.
    """
    import inspect
    import torch
    require(x.ndim==3 and tuple(x.shape[1:])==(4,2500),
            'E7 waveform input must be [B,4,2500]')
    require(mask.ndim==2 and tuple(mask.shape)==(x.shape[0],4),
            'E7 slot mask input must be [B,4]')
    parameter_count=len(inspect.signature(net.forward).parameters)
    if parameter_count==2:
        out=net(x,mask)
    elif parameter_count==1:
        # Existing E7 incorrectly called net(x), providing only 4 channels.
        # Frozen ResNet1D takes the mask as four additional temporal channels.
        mask_channels=mask.to(device=x.device,dtype=x.dtype).unsqueeze(-1)
        mask_channels=mask_channels.expand(-1,-1,x.shape[-1])
        model_input=torch.cat((x,mask_channels),dim=1)
        require(model_input.shape[1]==8,'E7 model input must have 8 channels')
        out=net(model_input)
    else:
        raise RuntimeError('P05_T07_STOP: unsupported locked model.forward signature')
    require(hasattr(out,'numel') and out.numel()==1,
            'locked binary backbone must return one logit per alarm')
    return out.reshape(())


def e7(lock,pilot):
    print('T07_E7_START',flush=True)
    import torch
    import models
    import preprocess_lib as prep
    import wfdb
    torch.manual_seed(42)
    torch.set_num_threads(8)
    require(torch.__version__.startswith('2.'),'unexpected torch major version')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1 and
            'Quadro M6000' in torch.cuda.get_device_name(0),
            'GPU1 must be the only CUDA-visible Quadro M6000')
    meta=pd.read_parquet(PROC/'val_meta.parquet',columns=['idx','record','event'])
    xcache=np.load(PROC/'val_W10_D0_x.npy',mmap_mode='r',allow_pickle=False)
    mcache=np.load(PROC/'val_W10_D0_mask.npy',mmap_mode='r',allow_pickle=False)
    require(xcache.shape==(len(meta),4,2500) and mcache.shape==(len(meta),4),
            'E7 benchmark input geometry mismatch')
    all_present=np.flatnonzero(np.all(mcache,axis=1))
    row=int(all_present[0]) if len(all_present) else int(np.argmax(np.sum(mcache,axis=1)))
    x=np.array(xcache[row:row+1],dtype=np.float32,copy=True)
    mask=np.array(mcache[row:row+1],dtype=np.bool_,copy=True)
    expected=[]
    for seed in range(5):
        score=pd.read_parquet(DIR/'outputs'/'t03'/f'A_s{seed}'/'scores_val.parquet')
        v=score.set_index('idx')['logit']
        require(meta['idx'].iloc[row] in v.index,'validation score idx missing')
        expected.append(float(v.loc[meta['idx'].iloc[row]]))
    # Check a real source waveform against the locked 10 s cache (no post-alarm samples).
    raw,names,_=read_raw60(RAW/str(meta['record'].iloc[row])/str(meta['event'].iloc[row]),wfdb)
    testx,testm=prep.preprocess_event(raw,names,10,0,fs=250,t0=75000,
                                     start_sample=60000)
    require(np.allclose(testx,x[0],atol=2e-5,rtol=1e-5)
            and np.array_equal(testm,mask[0]),
            'E7 preprocessing differs from accepted cached alarm')
    rows=[]
    for name,nthreads,device in (('CPU_1_THREAD',1,'cpu'),('CPU_8_THREADS',8,'cpu'),
                                 ('GPU1',8,'cuda:0')):
        torch.set_num_threads(nthreads)
        d=torch.device(device)
        xx=torch.from_numpy(x).to(d)
        mm=torch.from_numpy(mask).to(d)
        nets=[]
        for seed in range(5):
            checkpoint=torch.load(DIR/'outputs'/'t03'/f'A_s{seed}'/'ckpt.pt',
                                  map_location='cpu')
            require(checkpoint['model']=='resnet1d' and int(checkpoint['seed'])==seed,
                    'benchmark checkpoint identity mismatch')
            net,nparams=models.make_model('resnet1d')
            require(nparams==checkpoint['parameters'],'benchmark model parameters mismatch')
            net.load_state_dict(checkpoint['state_dict'],strict=True)
            net.eval().to(d)
            nets.append(net)
        def forward_ensemble():
            logits=[model_logit(net,xx,mm) for net in nets]
            return torch.sigmoid(torch.stack(logits).mean())
        with torch.inference_mode():
            observed=[]
            for net in nets:
                z=model_logit(net,xx,mm)
                observed.append(float(z.item()))
            require(np.allclose(observed,expected,rtol=1e-3,atol=1e-3),
                    'locked model output differs from archived T03 val logits')
            for _ in range(20):forward_ensemble()
            if d.type=='cuda':torch.cuda.synchronize(d)
            elapsed=[]
            for _ in range(500):
                if d.type=='cuda':torch.cuda.synchronize(d)
                st=time.perf_counter_ns()
                z=forward_ensemble()
                if d.type=='cuda':torch.cuda.synchronize(d)
                elapsed.append((time.perf_counter_ns()-st)*1e-6)
            require(np.isfinite(elapsed).all() and (np.asarray(elapsed)>0).all(),
                    'E7 invalid timer readings')
        rows.append({'configuration':name,'threads':nthreads,
                     'physical_gpu_index':1 if d.type=='cuda' else None,
                     'warmup_runs':20,'measured_runs':500,
                     'median_ms':float(np.median(elapsed)),
                     'p95_ms':float(np.percentile(elapsed,95)),
                     'torch_version':str(torch.__version__),
                     'cpu_model':cpu_name(),
                     'gpu_model':torch.cuda.get_device_name(0) if d.type=='cuda' else None})
        print(f'T07_E7_LATENCY_PASS={name}; median_ms={rows[-1]["median_ms"]:.4f}; '
              f'p95_ms={rows[-1]["p95_ms"]:.4f}',flush=True)
        del nets,xx,mm
        if d.type=='cuda':
            pilot.verify_nvidia_process_is_gpu1()
            torch.cuda.empty_cache()
    # Preprocessing benchmark is separate, 100 independent repeats of frozen preprocessing.
    torch.set_num_threads(1)
    pre_times=[]
    for _ in range(20):prep.preprocess_event(raw,names,10,0,fs=250,t0=75000,
                                            start_sample=60000)
    for _ in range(100):
        st=time.perf_counter_ns()
        px,pm=prep.preprocess_event(raw,names,10,0,fs=250,t0=75000,
                                    start_sample=60000)
        pre_times.append((time.perf_counter_ns()-st)*1e-6)
    require(np.allclose(px,x[0],atol=2e-5,rtol=1e-5)
            and np.array_equal(pm,mask[0]),'frozen preprocessing output mismatch')
    pre={'configuration':'PREPROCESSING_CPU_1_THREAD','warmup_runs':20,'measured_runs':100,
         'median_ms':float(np.median(pre_times)),
         'p95_ms':float(np.percentile(pre_times,95)),
         'scope':'60s prealarm filter and normalize; extract 10s before alarm; no waveform I/O'}
    require(len(rows)==3,'E7 requires exactly three benchmark configurations')
    pd.DataFrame(rows).to_parquet(OUT/'E7_latency.parquet',index=False)
    pd.DataFrame([pre]).to_parquet(OUT/'E7_preprocessing_latency.parquet',index=False)
    ans={'status':'PASS','rows':rows,'preprocessing':pre,
         'ensemble':'5 frozen A checkpoints; sigmoid(mean logits); no model fitting',
         'time_unit':'ms','batch_size':1,'includes_host_dispatch':True,
         'raw_wfdb_read_excluded':True}
    save_json(summary_path('E7'),ans)
    print('T07_E7=PASS; models=5; batch=1; n=500/config; preprocess_separate=PASS',flush=True)
    return ans


def annotator_votes(record,event,raw_root):
    """Return count of usable nonadjudicating votes if the official CSV exists."""
    file=raw_root/'event_label_per_annotator.csv'
    if not file.is_file():return None
    votes=pd.read_csv(file,dtype=str,keep_default_na=False)
    required={'record','event','decision','is_adjudication'}
    require(required.issubset(votes.columns),'raw annotator CSV schema mismatch')
    p=votes.loc[(votes['record']==str(record))&(votes['event']==str(event))]
    p=p.loc[p['is_adjudication'].str.strip().str.lower().isin(('false','0','no'))]
    p=p.loc[p['decision'].str.strip().str.lower().isin(('true','false','uncertain'))]
    return int(len(p))


def future_display(base,hdr,prep,wfdb):
    """Post-alarm samples are for display ONLY; never passed to models or SQI."""
    if int(hdr.sig_len)<76250:
        return None,'POST_ALARM_NOT_AVAILABLE'
    sig,meta=wfdb.rdsamp(str(base),sampfrom=60000,sampto=76250)
    require(sig.shape[0]==16250,'postalarm 5s waveform truncated')
    x,mask=prep.preprocess_event(sig,list(meta['sig_name']),10,5,
                                 fs=250,t0=75000,start_sample=60000)
    require(x.shape==(4,3750),'future DISPLAY-ONLY preprocessing shape mismatch')
    return x[:,2500:],'POST_ALARM_DISPLAY_ONLY'


def plot_case(case_no,cohort,x,mask,future,status,out_png,out_pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.arange(2500)/250.0-10.0
    tf=np.arange(1250)/250.0
    fig,axes=plt.subplots(4,1,figsize=(11,7),sharex=True,dpi=160)
    for ch,(ax,name) in enumerate(zip(axes,('ECG1','ECG2','PPG','ABP'))):
        ax.axvline(0,color='black',lw=.9,ls='--')
        if bool(mask[ch]):
            ax.plot(t,x[ch],lw=.7,color='#1f5281')
            if future is not None:
                ax.plot(tf,future[ch],lw=.55,color='gray',alpha=.45)
        else:
            ax.text(.05,.68,'MISSING SLOT',transform=ax.transAxes,fontsize=8)
        ax.set_ylabel(name+'\n(norm.)',fontsize=9)
        ax.grid(alpha=.12)
    axes[-1].set_xlim(-10,5)
    axes[-1].set_xlabel('Time relative to alarm, s; model input [-10,0) only')
    fig.suptitle(f'Case {case_no:02d} ({cohort}) | '
                 'gray: future observation only, NOT model input',fontsize=11)
    if status=='POST_ALARM_NOT_AVAILABLE':
        axes[0].text(.68,.83,'Post-alarm signal not available',
                     transform=axes[0].transAxes,color='gray',fontsize=8)
    fig.tight_layout(rect=[0,0,1,.965])
    fig.savefig(out_png,dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)


def e8(data,lock):
    print('T07_E8_START',flush=True)
    import wfdb
    from wfdb import processing as wfproc
    import preprocess_lib as prep
    folder,vt=external_vt()
    lam=float(lock['lambda']['A'][.02])
    picks=[]
    for cohort,nexpected in (('test',1),('c2015',2)):
        y=np.asarray(data[cohort]['y'],dtype=int)
        p=np.asarray(data[cohort]['prob'],dtype=float)
        found=np.flatnonzero((y==1)&(p<lam))
        require(len(found)==nexpected,
                f'E8 missed true VT gate differs: {cohort}, got={len(found)} expected={nexpected}')
        picks.extend((cohort,int(i)) for i in found)
    require(len(picks)==3,'E8 must contain exactly 3 missed cases')
    sqi_cache={}
    path=OUT/'E5_signal_quality.parquet'
    if path.is_file():
        tbl=pd.read_parquet(path)
        for cohort in ('test','c2015'):
            q=tbl.loc[tbl['cohort']==cohort]
            sqi_cache[cohort]=q.set_index('idx')['bsqi']
    folderout=OUT/'missed_cases'
    require(not folderout.exists(),'existing E8 missed_cases directory; preserve originals')
    # Pre-compute ALL case summaries and figures in-memory until three checks pass.
    # No record/event ID or signal vector is serialized into REPORT/JSON/parquet.
    cases=[];figures=[]
    for ordinal,(cohort,i) in enumerate(picks,1):
        if cohort=='test':
            meta=data[cohort]['meta'].iloc[i]
            base=RAW/str(meta['record'])/str(meta['event'])
            cached_x=np.load(PROC/'test_W10_D0_x.npy',mmap_mode='r')[i]
            cached_m=np.load(PROC/'test_W10_D0_mask.npy',mmap_mode='r')[i]
            votes=annotator_votes(str(meta['record']),str(meta['event']),
                                  release_paths.get("data_raw")/'vtac'/'1.1')
        else:
            base=folder/vt[i][0]
            cached_x=None;cached_m=None
            votes=None
        raw,names,hdr=read_raw60(base,wfdb)
        if cohort=='c2015':
            cached_x,cached_m=prep.preprocess_event(raw,names,10,0,
                                  fs=250,t0=75000,start_sample=60000)
        else:
            recalc_x,recalc_m=prep.preprocess_event(raw,names,10,0,
                                  fs=250,t0=75000,start_sample=60000)
            require(np.allclose(recalc_x,cached_x,atol=2e-5,rtol=1e-5)
                    and np.array_equal(recalc_m,cached_m),
                    'E8 input disagrees with frozen test cache')
        quality=sqi_from_signal(raw,names,np.asarray(cached_x),
                                np.asarray(cached_m),prep,wfproc)
        if cohort in sqi_cache:
            previous=float(sqi_cache[cohort].loc[i])
            require(abs(previous-quality['bsqi'])<1e-9,'E8/E5 bSQI not reproducible')
        future,status=future_display(base,hdr,prep,wfdb)
        p=float(data[cohort]['prob'][i]);seed=[float(d[i]) for d in data[cohort]['seeds']]
        require(p<lam and np.isfinite(seed).all(),
                'E8 probability/seed validity gate failed')
        cases.append({'case':ordinal,'cohort':cohort,'evaluation':'POST-HOC',
                      'p_ensemble':p,'p_seed_0':seed[0],'p_seed_1':seed[1],
                      'p_seed_2':seed[2],'p_seed_3':seed[3],'p_seed_4':seed[4],
                      'lambda_locked':lam,
                      'missing_slots':','.join(z for z,b in zip(('ECG1','ECG2','PPG','ABP'),cached_m) if not b) or 'NONE',
                      'bsqi':quality['bsqi'],'votes':votes,
                      'post_alarm_status':status})
        figures.append((ordinal,cohort,np.array(cached_x,copy=True),
                        np.array(cached_m,copy=True),future,status))
        print('T07_E8_CASE_CHECK=PASS; ordinal='+str(ordinal)+'; cohort='+cohort+
              '; p_below_locked_lambda=TRUE; future='+status,flush=True)
    folderout.mkdir(parents=True,exist_ok=False)
    for ordinal,cohort,x,mask,future,status in figures:
        name=f'case_{ordinal:02d}_{cohort}'
        plot_case(ordinal,cohort,x,mask,future,status,
                  folderout/(name+'.png'),folderout/(name+'.pdf'))
    pd.DataFrame(cases).to_parquet(OUT/'E8_missed_summary.parquet',index=False)
    ans={'status':'PASS','n_cases':3,'test_missed':1,'c2015_missed':2,
         'post_alarm_unavailable':int(sum(c['post_alarm_status']=='POST_ALARM_NOT_AVAILABLE'
                                          for c in cases)),
         'figures_saved_local_only':6,
         'cases':[{'case':c['case'],'cohort':c['cohort'],'evaluation':c['evaluation'],
                   'p_ensemble':c['p_ensemble'],'p_seeds':[c[f'p_seed_{s}'] for s in range(5)],
                   'missing_slots':c['missing_slots'],'bsqi':c['bsqi'],
                   'votes':('<10' if c['votes'] is not None and c['votes']<10 else c['votes']),
                   'post_alarm_status':c['post_alarm_status']} for c in cases]}
    save_json(summary_path('E8'),ans)
    print('T07_E8=PASS; test=1; C2015=2; all_p_less_than_lambda=PASS',flush=True)
    return ans


def e4(data,lock,conformal):
    print('T07_E4_START; optional=TRUE; deadline_s=7200',flush=True)
    import wfdb
    from scipy import signal
    import preprocess_lib as prep
    import t05b_external_c2015 as ext
    start=time.monotonic();deadline=start+7200.0
    frame=[]
    for split in ('calib','val'):
        prof=DIR/'outputs'/'t06b'/f'profiles_{split}.parquet'
        require(prof.is_file(),'accepted T06b profile table missing')
        q=pd.read_parquet(prof)
        meta=data[split]['meta']
        require(np.array_equal(q['idx'].to_numpy(),meta['idx'].to_numpy()),
                'T06b event profile index misaligned')
        m=meta[['record','event']].copy();m['profile']=q['profile']
        m['split']=split
        frame.append(m)
    allrows=pd.concat(frame,ignore_index=True)
    grouped=allrows.groupby('record',sort=False)
    require((grouped['profile'].nunique()==1).all()
            and (grouped['split'].nunique()==1).all(),
            'locked record-level profile or partition invalid')
    recordings=[];timedout=False;read_events=0
    for record,g in grouped:
        if time.monotonic()>deadline:
            timedout=True;break
        fps=[]
        for row in g.itertuples(index=False):
            if time.monotonic()>deadline:
                timedout=True;break
            raw,names=ext.read_pre_alarm(RAW/str(record)/str(row.event),wfdb,np)
            fp=ext.fingerprint_ii(raw,names,np,signal,prep)
            if fp is not None:fps.append(fp)
            read_events+=1
        if timedout:break
        if fps:
            recordings.append({'record':str(record),'profile':str(g['profile'].iloc[0]),
                               'split':str(g['split'].iloc[0]),
                               'fingerprints':fps})
    duplicate=set();related=set();suspect_calib=set();evaluated=0
    for profile in ('P1','P2','P3'):
        group=[z for z in recordings if z['profile']==profile]
        if not group:continue
        refs=[];owners=[]
        for owner,z in enumerate(group):
            for fp in z['fingerprints']:
                refs.append(fp);owners.append(owner)
        refs=np.stack(refs,dtype=np.float32)
        owners=np.asarray(owners,dtype=np.int32)
        for i,z in enumerate(group):
            if time.monotonic()>deadline:
                timedout=True;break
            targets=owners>i
            if not targets.any():continue
            others=refs[targets]
            other_owner=owners[targets]
            for fp in z['fingerprints']:
                if time.monotonic()>deadline:
                    timedout=True;break
                similarities=ext.max_normalized_xcorr(fp,others,np)
                hits=np.unique(other_owner[similarities>0.95])
                for j in hits:
                    duplicate.add((z['record'],group[int(j)]['record']))
                    related.add(z['record']);related.add(group[int(j)]['record'])
                    if z['split']!=group[int(j)]['split']:
                        suspect_calib.add(z['record'] if z['split']=='calib' else
                                          group[int(j)]['record'])
            evaluated+=1
            if evaluated%100==0:
                print(f'T07_E4_PROGRESS=profile_{profile}; groups_done={evaluated}; '
                      f'time_s={time.monotonic()-start:.1f}',flush=True)
            if timedout:break
        if timedout:break
    # Descriptive sensitivity, never changes frozen lock.
    sensitive={'status':'NOT_APPLICABLE_NO_CROSS_SPLIT_MATCHES'}
    if suspect_calib and not timedout:
        use=~np.isin(data['calib']['records'],list(suspect_calib))
        cc=data['calib'];vv=data['val'];y=cc['y'][use]
        nr=len(np.unique(cc['records'][use&(cc['y']==1)]))
        if nr==0 or 1/(nr+1)>.02:
            sensitive={'status':'INFEASIBLE_CRC_AFTER_EXCLUSION',
                       'positive_calibration_records':redact(nr)}
        else:
            lam=float(conformal.crc_fnr_threshold(cc['prob'][use],y,
                                    cc['records'][use],.02))
            res=blocked_rates(vv['prob'],vv['y'],lam,vv['records'])
            sensitive={'status':'POSTHOC_SENSITIVITY_ONLY',
                       'positive_calibration_records':redact(nr),
                       'lambda_recalibrated':lam,
                       'val_suppression':res['suppression'],
                       'val_fnr_k':res['fnr_k'],'val_fnr_n':res['fnr_n'],
                       'val_fnr_ci95_lo':res['fnr_ci95_lo'],
                       'val_fnr_ci95_hi':res['fnr_ci95_hi'],
                       'lambda_locked_unchanged':float(lock['lambda']['A'][.02])}
    elif timedout:
        sensitive={'status':'NOT_COMPUTED_PARTIAL_SEARCH_TIME_LIMIT'}
    ans={'status':'PARTIAL_2H_LIMIT' if timedout else 'PASS',
         'fingerprint_threshold':.95,'method':'absolute full-lag L2-normalized FFT correlation of lead II 60s at 25Hz',
         'same_profile_only':True,'distinct_recordings_only':True,
         'time_elapsed_s':time.monotonic()-start,
         'events_scanned':read_events,'recordings_with_fingerprints':len(recordings),
         'recording_groups_compared':evaluated,'suspected_pairs':len(duplicate),
         'recordings_related':len(related),
         'suspected_calib_records_cross_val':len(suspect_calib),
         'sensitivity':sensitive}
    save_json(summary_path('E4'),ans)
    print('T07_E4='+ans['status']+'; pairs='+redact(len(duplicate))+
          '; recordings_related='+redact(len(related)),flush=True)
    return ans


def render_report(lock_before,lock_after):
    if not all(summary_path(s).is_file() for s in REQUIRED):
        print('T07_REPORT_DEFERRED=MISSING_REQUIRED_SECTION',flush=True)
        return False
    summaries={s:json.loads(summary_path(s).read_text(encoding='utf-8'))
               for s in REQUIRED}
    require(all(z['status']=='PASS' for z in summaries.values()),
            'not all required T07 sections passed')
    if summary_path('E4').is_file():
        summaries['E4']=json.loads(summary_path('E4').read_text(encoding='utf-8'))
    e3tab=pd.read_parquet(OUT/'E3_crc_event_vs_record.parquet')
    e5tab=pd.read_parquet(OUT/'E5_tertiles.parquet')
    etab=pd.read_parquet(OUT/'E7_latency.parquet')
    e8tab=pd.read_parquet(OUT/'E8_missed_summary.parquet')
    lams=summaries['E3']['lambdas']; e7pre=summaries['E7']['preprocessing']
    lines=[
      '# P05-T07 presubmission POST-HOC analysis',
      '',
      'Status: COMPLETE; WAIT_FOR_DIRECTOR. reference-workstation, CPU <=8 threads, BelowNormal; GPU1 E7 only.',
      'T05b was not rerun. All primary model thresholds and frozen artifacts remain unchanged.',
      'TEST and Challenge 2015 are POST-HOC, not independent preregistered confirmatory evidence.',
      '',
      '## E3. Event-level CRC versus frozen record-level CRC',
      '',
      'Positive event is the CRC group for event-level comparison. '
      'Suppression is p < lambda (strict). Exact Clopper-Pearson 95% two-sided CI.',
      'Record-mean FNR averages within-positive-record missed-VT fractions, then across positive records.',
      '',
      '| Alpha | Lambda event | Lambda record locked | Positive calib events |',
      '|---:|---:|---:|---:|'
    ]
    for z in lams:
        lines.append(f"| {z['alpha']:.2f} | {z['lambda_event']:.10g} | "
                     f"{z['lambda_record']:.10g} | {z['n_positive_events_calib']} |")
    lines+=['',
        '| Dataset / evaluation | CRC unit | Alpha | Suppression | FNR k/n | FNR 95% CP | Record-mean FNR |',
        '|---|---|---:|---:|---:|---|---:|']
    for z in e3tab.to_dict('records'):
        lines.append(f"| {z['cohort']} / {z['evaluation']} | {z['method']} | "
          f"{z['alpha']:.2f} | {number(z['suppression'])} | {z['fnr_k']}/{z['fnr_n']} | "
          f"[{number(z['fnr_ci95_lo'])}, {number(z['fnr_ci95_hi'])}] | "
          f"{number(z['record_mean_fnr'])} |")
    lines += ['', 'Validation gate: frozen record CRC alpha=0.02 false suppression '
              f"{number(summaries['E3']['val_record_suppression_at_002'])}; "
              'FNR=2/141. PASS.', '',
              '## E5. Prealarm signal-quality strata', '',
              'ECG1 raw nonfinite proportion; raw ECG1 adjacent flat samples (|difference|<1e-6); '
              'bSQI from 150 ms one-to-one agreement of XQRS and GQRS, '
              'max of ECG1/ECG2 when available. Signals derive from locked 10 s prealarm input.',
              'Calibration bSQI tertile cutpoints fixed before validation/test/external evaluation: ' +
              str(summaries['E5']['bsqi_cutpoints'])+'.',
              'Ties assigned to lower tertile; unscorable bSQI set to 0 and retained. '
              'Discrete-score ties may make tertile sizes unequal. The underlying tertile event counts '
              'sum exactly to the frozen split sizes (automated PASS).',
              '',
              '| Dataset / evaluation | bSQI tertile | Total | True | False | Suppression | FNR k/n | FNR 95% CP |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for cohort in SPLITS:
        group=e5tab.loc[e5tab['cohort']==cohort].to_dict('records')
        # Complementary masking within a split when any label-stratum count is <10.
        vulnerable=any(z['n_true']<10 or z['n_false']<10 for z in group)
        protect=set()
        if vulnerable:
            for z in group:
                if z['n_true']<10 or z['n_false']<10:protect.add(int(z['tertile']))
            if len(protect)==1:
                larger=max((z for z in group if int(z['tertile']) not in protect),
                           key=lambda a:a['n_events'])
                protect.add(int(larger['tertile']))
        for z in group:
            hide=int(z['tertile']) in protect
            a='WITHHELD' if hide and z['n_true']>=10 else redact(z['n_true'])
            b='WITHHELD' if hide and z['n_false']>=10 else redact(z['n_false'])
            # Do not expose group rates when denominator <10.
            suppression=number(z['suppression']) if z['n_false']>=10 and not hide else 'N/A'
            fnr=f"{z['fnr_k']}/{z['fnr_n']}" if z['n_true']>=10 and not hide else 'N/A'
            ci=(f"[{number(z['fnr_ci95_lo'])}, {number(z['fnr_ci95_hi'])}]"
                if z['n_true']>=10 and not hide else 'N/A')
            lines.append(f"| {z['cohort']} / {z['evaluation']} | {z['tertile']} | "
                         f"{redact(z['n_events'])} | {a} | {b} | {suppression} | {fnr} | {ci} |")
    lines+=['', 'E5 logistic regression of suppressed | true VT on continuous bSQI. '
             'Analyses with fewer than 3 missed VT are not fit.', '',
             '| Dataset / evaluation | Suppressed true VT | Status | Logistic coefficient | Odds ratio (per unit bSQI) |',
             '|---|---:|---|---:|---:|']
    for z in summaries['E5']['logistic']:
        lines.append(f"| {z['cohort']} / {z['evaluation']} | "
                     f"{redact(z['missed_true_vt'])} | {z['status']} | "
                     f"{number(z['beta_bsqi'])} | {number(z['odds_ratio_bsqi'])} |")
    lines+=['', 'Validation total FNR at locked alpha=0.02: 2/141. PASS.', '',
             '## E7. Ensemble inference latency (batch 1)', '',
             '500 timed runs following 20 warmup runs per configuration. '
             'GPU1 runs synchronized before and after each iteration. '
             'Latency includes five frozen forward passes and logit-mean sigmoid, '
             'but excludes waveform disk I/O and preprocessing.',
             '', '| Hardware | Torch | Threads | Median (ms) | p95 (ms) |',
             '|---|---|---:|---:|---:|']
    for z in etab.to_dict('records'):
        lines.append(f"| {z['configuration']} | {z['torch_version']} | "
                     f"{z['threads']} | {z['median_ms']:.4f} | {z['p95_ms']:.4f} |")
    lines += ['', 'CPU model: '+str(summaries['E7']['rows'][0]['cpu_model'])+'.',
              f"Separate preprocessor (100 runs; CPU 1 thread): median "
              f"{e7pre['median_ms']:.4f} ms, p95 {e7pre['p95_ms']:.4f} ms. "
              'Filter and normalization of prealarm 60 s; waveform I/O excluded.', '',
              '## E8. Missed true VT under frozen alpha=0.02', '',
              'Three cases: 1 test and 2 Challenge 2015; all p < frozen lambda. '
              'Individual waveform images (PNG/PDF) stored ONLY under outputs/t07/missed_cases. '
              'No patient/record/event identifiers or case figures are included in this report.',
              'Post-alarm signal is displayed in gray only when actual source samples exist. '
              'Post-alarm samples never enter the model.', '',
              '| Case | Cohort / evaluation | p ensemble | p seeds 0..4 | Locked lambda | Missing slots | bSQI | Annotator votes | Postalarm |',
              '|---:|---|---:|---|---:|---|---:|---|---|']
    for z in e8tab.to_dict('records'):
        votes=('N/A' if pd.isna(z['votes']) else redact(int(z['votes'])))
        seeds=', '.join(number(z[f'p_seed_{i}'],7) for i in range(5))
        lines.append(f"| {z['case']} | {z['cohort']} / {z['evaluation']} | "
                     f"{number(z['p_ensemble'],8)} | {seeds} | "
                     f"{number(z['lambda_locked'],9)} | {z['missing_slots']} | "
                     f"{number(z['bsqi'])} | {votes} | {z['post_alarm_status']} |")
    lines+=['', '## E4. Optional within-VTaC lead II recording-overlap screening', '']
    if 'E4' in summaries:
        z=summaries['E4']
        lines+=['Search status: '+z['status']+'.; limit 7200 s.',
                '| Pairs suspected | Recordings related | Calib recordings possibly overlapping val | Records screened |',
                '|---:|---:|---:|---:|',
                f"| {redact(z['suspected_pairs'])} | {redact(z['recordings_related'])} | "
                f"{redact(z['suspected_calib_records_cross_val'])} | {z['recordings_with_fingerprints']} |",
                'Sensitivity (post-hoc; locked lambda unchanged): '+json.dumps(
                    z['sensitivity'],ensure_ascii=False)+'.']
    else:
        lines.append('NOT RUN (optional; no claims from E4).')
    lines+=['', '## Integrity and provenance', '',
            'final_lock.yaml SHA256 before: `'+lock_before+'`',
            'final_lock.yaml SHA256 after: `'+lock_after+'`',
            'Final lock unchanged: '+('PASS' if lock_before==lock_after==LOCK_SHA256 else 'FAIL')+'.',
            'Frozen checkpoints, configurations, common, T05b, T06, T06b and TEST_OPENED.flag '
            'were not written by this script.',
            'Per-record IDs are never published; counts below 10 are masked except '
            'the explicitly permitted FNR k/n counts.',
            '', '## Generated output files, SHA-256', '',
            '| File (relative to project) | SHA-256 |', '|---|---|']
    lines.append('| `src/t07_presubmission.py` | `'+sha(__file__)+'` |')
    for path in all_files():
        if path.name.endswith('.log') or path.name=='file_manifest.sha256':
            continue  # active tee output is still changing; hash after process exit
        lines.append('| `'+path.relative_to(DIR).as_posix()+'` | `'+sha(path)+'` |')
    lines+=['', 'The active execution log SHA is printed by the shell after it closes. '
             'The report itself is SHA-256 verified after writing, since self-hashing '
             'inside its own contents is mathematically circular.', '']
    require(sha(DIR/'configs'/'final_lock.yaml')==LOCK_SHA256,
            'final lock changed while formatting report')
    # Updating only P05-T07.md is permitted when optional E4 runs after required sections.
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    return True


def main():
    p=argparse.ArgumentParser(description='P05-T07 posthoc presubmission sections')
    p.add_argument('--only',default='E3,E5,E7,E8',
                   help='comma-separated E3,E5,E7,E8,E4 or REPORT')
    args=p.parse_args()
    requested=[z.strip().upper() for z in args.only.split(',') if z.strip()]
    require(requested and len(set(requested))==len(requested) and
            set(requested).issubset({'E3','E5','E7','E8','E4','REPORT'}),
            'bad --only list')
    require(not ('E4' in requested and any(s in requested for s in REQUIRED)),
            'E4 is optional and must run separately after priority analyses')
    start=time.monotonic()
    before=sha(DIR/'configs'/'final_lock.yaml')
    lock,pilot,conformal,metrics=preflight()
    OUT.mkdir(parents=True,exist_ok=True)
    data=None
    run_status={}
    for section in requested:
        if section=='REPORT':continue
        summ=summary_path(section)
        if summ.is_file():
            payload=json.loads(summ.read_text(encoding='utf-8'))
            require(payload['status'] in ('PASS','PARTIAL_2H_LIMIT'),
                    'existing section output invalid; do not overwrite')
            run_status[section]='PRESERVED_PREVIOUS_'+payload['status']
            print(f'T07_{section}=PRESERVED_PREVIOUS_OUTPUT',flush=True)
            continue
        # A previously interrupted section must be inspected, not overwritten.
        partial=list(OUT.glob(section+'_*.parquet'))+list(OUT.glob(section+'_*.json'))
        if section=='E8' and (OUT/'missed_cases').exists():
            partial.append(OUT/'missed_cases')
        require(not partial, section+' partial output exists; preserve for forensic audit')
        if section!='E7' and data is None:
            data=get_data(lock,conformal,metrics)
        if section=='E3':e3(data,lock,conformal)
        elif section=='E5':e5(data,lock)
        elif section=='E7':e7(lock,pilot)
        elif section=='E8':e8(data,lock)
        elif section=='E4':e4(data,lock,conformal)
        run_status[section]='PASS'
    after=sha(DIR/'configs'/'final_lock.yaml')
    require(before==after==LOCK_SHA256,'final lock changed during T07')
    report_ready=render_report(before,after)
    print('=== REPORT P05-T07 ===',flush=True)
    print('status: '+('COMPLETE; WAIT_FOR_DIRECTOR' if report_ready else
                       'PARTIAL; REQUIRED_SECTIONS_PENDING')+'; reference-workstation',flush=True)
    print('cpu: <=8; priority: BelowNormal; GPU1: E7 ONLY',flush=True)
    print('runtime_s: %.3f'% (time.monotonic()-start),flush=True)
    print('script_sha256_8: '+sha(__file__)[:8],flush=True)
    print('final_lock_sha256_before: '+before,flush=True)
    print('final_lock_sha256_after: '+after,flush=True)
    for sec in ('E3','E5','E7','E8','E4'):
        s=summary_path(sec)
        print(sec+': '+(json.loads(s.read_text(encoding='utf-8'))['status']
                           if s.is_file() else 'NOT_RUN'),flush=True)
    print('outputs: '+str(OUT),flush=True)
    print('report: '+(str(REPORT) if report_ready else 'DEFERRED'),flush=True)
    print('report_sha256: '+(sha(REPORT) if report_ready else 'N/A'),flush=True)
    print('file_count_no_live_log: '+str(sum(not p.name.endswith('.log') for p in all_files())),flush=True)
    print('warnings: test/C2015 POST-HOC; E8 case images LOCAL ONLY; E4 optional',flush=True)
    print('=== END ===',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('P05_T07_STOP='+type(e).__name__+': '+str(e),file=sys.stderr,flush=True)
        raise
