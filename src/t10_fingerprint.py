"""P05-T10 POST-HOC N6 cross-cohort lead-II overlap threshold sensitivity.

The frozen T05b FFT correlation function is reused exactly. Original T06 does
not persist per-alarm correlation scores, so raw signals are read in memory.
No fingerprint, patient ID, waveform, or per-event correlation is exported.
"""
from __future__ import annotations

import numpy as np

from t10_shared import must,sha,save_json,red,safe_count_fraction


def run_n6(ctx):
    import wfdb
    from scipy import signal
    import preprocess_lib as prep
    import t05b_external_c2015 as ext
    folder,vt=ctx.vt()
    must((folder/"SHA256SUMS").is_file(),"N6_SOURCE_SHA_MANIFEST_PRESENT")
    checked=ext.audit_training(folder)
    must(len(checked)==341 and len(vt)==341,"N6_C2015_AUDIT")
    cfg={"project_dir":str(ctx.project),"proc_dir":str(ctx.proc),
         "raw_dir":str(ctx.workspace/"data_raw"/"vtac"/"1.1")}
    payload=ext.prepare_public_vt(cfg,vt,folder,prep,np,wfdb,signal)
    reference=ext.read_reference_fingerprints(cfg,prep,np,ctx.pd,wfdb,signal)
    # The original method is full-lag *absolute* FFT xcorr on 1500-sample
    # unit-normalized lead-II vectors, returning max over all VTaC references.
    maxcorr=np.zeros(341,dtype=np.float64)
    for i,q in enumerate(payload["fingerprints"]):
        values=[float(np.max(ext.max_normalized_xcorr(q,reference[key],np)))
                if len(reference[key]) else 0.0 for key in ("train","val","test")]
        maxcorr[i]=max(values)
        if (i+1)%50==0 or i==340:
            print(f"T10_N6_FINGERPRINT_PROGRESS={i+1}/341; no_ids_exported=YES",flush=True)
    must(np.isfinite(maxcorr).all() and np.all(maxcorr>=0),"N6_CORRELATIONS_FINITE")
    c=ctx.cohort("c2015")
    y=c["y"];p=c["p"]
    must(len(y)==341,"N6_Y_COUNT")
    rows=[]
    for tau in (.90,.95,.98):
        flagged=maxcorr>tau
        count=int(flagged.sum())
        if tau==.95:must(count==1,"N6_LOCKED_095_ANCHOR_ONE")
        remain=~flagged
        negative=(y==0)&remain;positive=(y==1)&remain
        suppressed=(p<float(ctx.lock["lambda"]["A"][.02]))
        nf=int(negative.sum());nt=int(positive.sum())
        must(nf>0 and nt>0,"N6_COHORT_CLASS_COUNTS")
        kf=int(np.sum(suppressed&negative));kt=int(np.sum(suppressed&positive))
        rows.append({"threshold":tau,"flagged_alarms":red(count),
                     "flagged_recordings":red(count),
                     "false_remaining":red(nf),"true_remaining":red(nt),
                     "suppression":float(kf/nf),"suppression_k_n":safe_count_fraction(kf,nf),
                     "fnr":float(kt/nt),"fnr_k_n":safe_count_fraction(kt,nt),
                     "status":"POST-HOC","source":"recomputed in memory with exact T05b overlap function",
                     "rule":"flag if max cross-correlation > threshold; not >= threshold"})
    save_json(ctx.out/"N6_fingerprint.json",{"status":"POST-HOC","rows":rows,
        "note":"No pairwise results/record IDs/waveforms stored. Input provenance is the T05b FFT method. "
               "Counts below 10 are redacted (including known 0.95 one-record overlap)."})
    ctx.check_unchanged()
    print("T10_N6=PASS; 3_THRESHOLDS; 095_ANCHOR=PASS",flush=True)
