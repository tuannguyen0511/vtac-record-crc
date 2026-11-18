"""P05-T11 POST-HOC correction of T10/N5 calibration row (original inputs read-only).

Do not apply the truncated/display lambda at the calibration boundary. This
script is intended for the audited P05 workspace, not for a fresh clone.
It creates only outputs/t11/N5_calib_corrected.json, never touches T10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '8'

from t10_shared import Context, must, save_json, set_priority, sha, sigmoid


def main(project: Path):
    import numpy as np
    set_priority()
    ctx=Context(project)
    source=ctx.out/'N5_record_multiplicity.csv'
    must(source.is_file(),'N5_T10_ORIGINAL_PRESENT')
    before=sha(source)
    calib=ctx.cohort('calib','A')
    y=calib['y']; p=calib['p']; rec=calib['rec']
    positives=np.asarray(y==1)
    recs=np.unique(rec[positives]); n=len(recs)
    must(n==188 and int(positives.sum())==353,'N5_CALIB_COUNTS_LOCKED')
    # The exact function returns an existing positive-score float64 boundary;
    # no rounding, nextafter, decimal formatting, or calibration reselection.
    lam=float(ctx.conformal.crc_fnr_threshold(p,y,rec,0.02))
    must(abs(lam-0.0139079025)<=5e-9,'N5_EXACT_CRC_ANCHOR')
    losses=np.asarray([float(np.mean(p[(rec==r)&positives]<lam)) for r in recs],dtype=np.float64)
    missed=int(np.sum((p<lam)&positives))
    total=float(np.sum(losses,dtype=np.float64))
    upper=float((total+1)/(n+1))
    record_mean=float(np.mean(losses))
    must(upper<=0.02+1e-14,'N5_EXACT_CRC_RISK_BOUND')
    must(abs(total-2.5)<1e-8,'N5_EXPECTED_FRACTIONAL_LOSS_2P5')
    must(missed==3,'N5_EXPECTED_BOUNDARY_CORRECTED_3_OF_353')
    must(0<=missed<=int(positives.sum()) and abs(record_mean-total/n)<1e-14,
         'N5_INTERNAL_RECORD_LOSS')
    unchanged={}
    for cohort,expected in [('val',(2,141)),('test',(1,137)),('c2015',(2,89))]:
        d=ctx.cohort(cohort,'A')
        count=int(np.sum((d['p']<float(ctx.lock['lambda']['A'][.02]))&(d['y']==1)))
        denom=int(np.sum(d['y']==1))
        must((count,denom)==expected,'N5_UNCHANGED_'+cohort.upper())
        unchanged[cohort]=f'{count}/{denom}'
    must(sha(source)==before,'N5_T10_ORIGINAL_UNCHANGED')
    ctx.check_unchanged()
    # The full-cohort FNR numerator is permitted; no per-record information
    # or small subgroup counts are included in the public corrected summary.
    record={'status':'POST-HOC_CORRECTION','target':'N5 calib all_positive_records',
            'alpha':0.02,'lambda_full_precision':lam,
            'lambda_full_float_hex':lam.hex(),
            'lambda_display_original':float(ctx.lock['lambda']['A'][.02]),
            'positive_calibration_records':n,'true_calibration_alarms':int(positives.sum()),
            'suppressed_true_alarms':missed,'fnr_k_n':f'{missed}/{int(positives.sum())}',
            'fractional_record_loss_sum':total,
            'crc_upper_bound':upper,'record_mean_fnr':record_mean,
            'unchanged_cohort_fnr':unchanged,
            'original_n5_t10_sha256':before,
            'note':'POST-HOC: strict p < exact CRC lambda; original outputs/t10/N5_record_multiplicity.csv preserved. '
                   'The displayed locked decimal lambda is rounded and is not used at the calibration boundary.'}
    target=ctx.project/'outputs'/'t11'/'N5_calib_corrected.json'
    save_json(target,record)
    must(sha(source)==before,'N5_T10_ORIGINAL_POSTWRITE_UNCHANGED')
    ctx.check_unchanged()
    print('=== P05-T11 N5 FIX ===',flush=True)
    print('N5_CALIB_EXACT_LAMBDA='+format(lam,'.17g'),flush=True)
    print('N5_CALIB_SUPPRESSED_TRUE='+str(missed)+'/'+str(int(positives.sum())),flush=True)
    print('N5_CALIB_FRACTIONAL_LOSS_SUM='+format(total,'.16g'),flush=True)
    print('N5_CALIB_CRC_BOUND='+format(upper,'.16g'),flush=True)
    print('N5_CALIB_RECORD_FNR='+format(record_mean,'.16g'),flush=True)
    print('N5_VAL_TEST_C2015_ANCHORS=PASS',flush=True)
    print('N5_T10_ORIGINAL_UNCHANGED=PASS',flush=True)
    print('T11_N5_FIX=PASS; POST-HOC',flush=True)
    print('N5_CORRECTED_SHA256='+sha(target),flush=True)
    print('=== END ===',flush=True)
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    a=parser.parse_args()
    try:raise SystemExit(main(a.project))
    except Exception as exc:
        print('T11_N5_FIX=FAIL; '+type(exc).__name__+'; AUDIT_HOLD; NO_PUSH',flush=True)
        raise SystemExit(2)
