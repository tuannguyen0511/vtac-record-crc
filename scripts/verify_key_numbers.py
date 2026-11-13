"""Independent verification from frozen logits, with no training/reinference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
PTH=yaml.safe_load((ROOT/'configs'/'paths.yaml').read_text(encoding='utf-8'))
def local(key):return (ROOT/str(PTH[key])).resolve()
PROC=local('data_proc'); OUT=local('outputs'); RAW=local('data_raw')
LOCK=yaml.safe_load((ROOT/'configs'/'final_lock.yaml').read_text(encoding='utf-8'))
LAMBDA=float(LOCK['lambda']['A'][.02])

def scores(split,expected):
    if split in ('calib','val'):
        meta=pd.read_parquet(PROC/f'{split}_meta.parquet')
        assert len(meta)==expected and meta['idx'].is_unique
        idx=meta['idx'].to_numpy()
        arr=[]
        for seed in range(5):
            df=pd.read_parquet(OUT/'t03'/f'A_s{seed}'/f'scores_{split}.parquet')
            assert list(df.columns)==['idx','logit'] and df['idx'].is_unique
            arr.append(df.set_index('idx')['logit'].reindex(idx).to_numpy(float))
        y=meta['y'].to_numpy(int)
    else:
        n=expected
        df=pd.read_parquet(OUT/'t06'/f'scores_{split}.parquet')
        assert list(df.columns)==['idx','seed','logit'] and len(df)==n*6
        arr=[]
        for seed in range(5):
            s=df.loc[df['seed'].astype(str)==str(seed)]
            assert len(s)==n and np.array_equal(s['idx'].to_numpy(),np.arange(n))
            arr.append(s['logit'].to_numpy(float))
        ens=df.loc[df['seed'].astype(str)=='ensemble']
        assert len(ens)==n and np.array_equal(ens['idx'].to_numpy(),np.arange(n))
        assert np.allclose(np.mean(arr,axis=0),ens['logit'].to_numpy(float),atol=1e-12,rtol=0)
        if split=='test':
            meta=pd.read_parquet(PROC/'test_meta.parquet'); lab=pd.read_parquet(PROC/'test_labels.parquet')
            assert len(meta)==len(lab)==n and np.array_equal(meta['idx'].to_numpy(),lab['idx'].to_numpy())
            y=lab['y'].to_numpy(int)
        else:
            root=RAW/'challenge-2015'/'1.0.0'
            f=root/'training'/'ALARMS' if (root/'training'/'ALARMS').is_file() else root/'ALARMS'
            import csv
            with f.open('r',newline='',encoding='utf-8') as h:
                vt=[r for r in csv.reader(h) if len(r)==3 and r[1]=='Ventricular_Tachycardia']
            y=np.asarray([int(r[2]) for r in vt],int)
            assert len(y)==n
    logits=np.stack(arr)
    assert np.isfinite(logits).all()
    prob=np.exp(-np.logaddexp(0,-logits.mean(axis=0)))
    return prob,y

CHECKS=[]
def emit(name,actual,expected,ok):
    CHECKS.append((name,bool(ok)))
    print('VERIFY '+name+' actual='+str(actual)+' expected='+str(expected)+' '+('PASS' if ok else 'FAIL'),flush=True)


def main():
    emit('LOCKED_LAMBDA_002',f'{LAMBDA:.10f}','0.0139079025',abs(LAMBDA-.0139079025)<5e-11)
    val,yv=scores('val',495)
    test,yt=scores('test',482)
    ext,ye=scores('c2015',341)
    def result(name,p,y):
        false=y==0;true=y==1
        sf=int(np.sum((p<LAMBDA)&false)); nf=int(false.sum())
        st=int(np.sum((p<LAMBDA)&true)); nt=int(true.sum())
        emit(name+'_SUPP_COUNT',f'{sf}/{nf}',
             {'VAL':'211/354','TEST':'191/345','C2015':'156/252'}[name],
             (sf,nf)=={'VAL':(211,354),'TEST':(191,345),'C2015':(156,252)}[name])
        rate=float(sf/nf)
        exp={'VAL':.596045,'TEST':191/345,'C2015':.619048}[name]
        emit(name+'_SUPP',f'{rate:.6f}',f'{exp:.6f}',abs(rate-exp)<1e-6)
        expk,expn={'VAL':(2,141),'TEST':(1,137),'C2015':(2,89)}[name]
        emit(name+'_FNR',f'{st}/{nt}',f'{expk}/{expn}',(st,nt)==(expk,expn))
        return float(roc_auc_score(y,p))
    result('VAL',val,yv)
    auc_t=result('TEST',test,yt)
    auc_e=result('C2015',ext,ye)
    emit('TEST_AUROC',f'{auc_t:.6f}','0.946049',abs(auc_t-.946049)<1e-6)
    emit('C2015_AUROC',f'{auc_e:.6f}','0.942304',abs(auc_e-.942304)<1e-6)
    n_ok=sum(x[1] for x in CHECKS)
    print(f'VERIFY_FINAL={"PASS" if n_ok==len(CHECKS) else "FAIL"} checks={n_ok}/{len(CHECKS)}',flush=True)
    return 0 if n_ok==len(CHECKS) else 2
if __name__=='__main__':raise SystemExit(main())
