"""P05-T10 POST-HOC CPU analyses N1,N4,N5,N7,N8,N9.

Run from t10_run.py after preflight. Never persists per-event predictions,
record identifiers, waveforms or sensitive group counts.
"""
from __future__ import annotations

import math
import warnings
from collections import Counter
import numpy as np
from scipy.stats import spearmanr

from t10_shared import (ALPHAS,CORE_ALPHAS,EXPECTED_LAMBDA_A,must,red,record_fnr,
                        safe_count_fraction,save_csv,save_json,out_metrics,sigmoid,forbid_overwrite_bytes)


def n1(ctx,c,v):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    summary={"label":"POST-HOC", "entropy_definition":
       "H(q_mean) = -q_mean log(q_mean)-(1-q_mean) log(1-q_mean), binary mean-vote proxy; "
       "categorical entropy of raw True/False/Uncertain votes is NOT recoverable from the T02b cache. "
       "q_mean uses Uncertain=0.5, excludes Reject/Save for Later; votes_missing excluded.",
       "cohorts":{}}
    for split,base in (("calib",c),("val",v)):
        b=ctx.cohort(split,"B")
        must(np.array_equal(base["y"],b["y"]) and np.array_equal(base["rec"],b["rec"]),"N1_AB_ALIGN_"+split.upper())
        true=base["y"]==1
        q=base["meta"]["q_mean"].to_numpy(dtype=float)
        nv=base["meta"]["n_votes"].to_numpy(dtype=int)
        missing=base["meta"]["votes_missing"].to_numpy(dtype=int)
        contested=base["meta"]["contested"].to_numpy(dtype=int)
        must(np.isfinite(q).all() and ((q>=0)&(q<=1)).all(),"N1_VOTE_CACHE_VALID_"+split.upper())
        qclip=np.clip(q,1e-14,1-1e-14)
        H=np.where((q<=0)|(q>=1),0.0,-qclip*np.log(qclip)-(1-qclip)*np.log1p(-qclip))
        pq={}
        for percent in [1,2,5,10,25]:
            pa=float(np.percentile(base["p"][true],percent))
            pb=float(np.percentile(b["p"][true],percent))
            pq[str(percent)]={"p_A":pa,"p_B":pb,"delta_B_minus_A":pb-pa}
        z_delta=base["z"]-b["z"]
        include=true&(missing==0)&(nv>0)
        x=z_delta[include];h=H[include];r=base["rec"][include]
        assert len(x)>15
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic=float(spearmanr(x,h).statistic)
        valid=np.isfinite(statistic)
        unique=np.unique(r)
        rng=np.random.default_rng(42)
        members=[np.flatnonzero(r==name) for name in unique]
        boot=[]
        for _ in range(2000):
            sample=rng.integers(0,len(members),size=len(members))
            ids=np.concatenate([members[i] for i in sample])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result=spearmanr(x[ids],h[ids]).statistic
            if np.isfinite(result): boot.append(float(result))
        ci=[float(z) for z in np.percentile(boot,[2.5,97.5])] if len(boot)>=100 else None
        summary["cohorts"][split]={
             "n_true":red(true.sum()),"n_true_vote_present":red(include.sum()),
             "n_true_contested":red((true&(contested==1)).sum()),
             "n_true_noncontested":red((true&(contested==0)).sum()),
             "n_true_vote_missing":red((true&(missing==1)).sum()),
             "quantiles_prob":pq,"lambda_A_002":float(ctx.lock["lambda"]["A"][.02]),
             "lambda_B_002":float(ctx.lock["lambda"]["B"][.02]),
             "spearman_logitA_minus_B_vs_vote_entropy":statistic if valid else None,
             "spearman_ci_record_bootstrap_95":ci,"bootstrap_valid":len(boot),
             "bootstrap_requested":2000,"seed":42}
        must(ci is not None,"N1_BOOTSTRAP_CI_"+split.upper())
    # VAL true-alarm logit CDF, black-and-white PDF vector.
    b=ctx.cohort("val","B")
    fig,ax=plt.subplots(figsize=(6.5,4.5),layout="constrained")
    sel=v["y"]==1
    for name,z,style in (("A (hard labels)",v["z"],"-"),("B (soft labels)",b["z"],"--")):
        sorted_z=np.sort(z[sel])
        yc=np.arange(1,len(sorted_z)+1)/len(sorted_z)
        ax.step(sorted_z,yc,where="post",linestyle=style,color="black",lw=1.6,label=name)
    ax.set(xlabel="Ensemble logit (true VT alarms, validation)",ylabel="Empirical cumulative probability",
           ylim=(0,1.01),title="POST-HOC: true-alarm score distribution")
    ax.grid(alpha=.18);ax.legend(loc="lower right");ax.yaxis.set_minor_locator(AutoMinorLocator())
    import io
    target=ctx.out/"N1_cdf.pdf"
    pdf=io.BytesIO()
    fig.savefig(pdf,format="pdf",metadata={"Title":"P05-T10 N1 POST-HOC",
        "Creator":"P05-T10","CreationDate":None,"ModDate":None})
    forbid_overwrite_bytes(target,pdf.getvalue())
    plt.close(fig)
    save_json(ctx.out/"N1_summary.json",summary)
    print("T10_N1=PASS; POST-HOC; entropy=H(q_mean)_PROXY; bootstrap=2000",flush=True)
    return summary


def n4(ctx,e):
    y=e["y"];p=e["p"]
    out=[]
    for alpha in (.02,.05):
        lam=float(ctx.lock["lambda"]["A"][alpha])
        yes=p>=lam
        tp=int(np.sum((y==1)&yes));tn=int(np.sum((y==0)&~yes))
        fp=int(np.sum((y==0)&yes));fn=int(np.sum((y==1)&~yes))
        val=float((tp+tn)/(tp+tn+fp+5*fn))
        out.append({"alpha":alpha,"lambda":lam,"challenge_score":val,
                    "TP":red(tp),"TN":red(tn),"FP":red(fp),"FN":red(fn),
                    "FN_FNR_WHOLE_COHORT":safe_count_fraction(fn,int(sum(y)),True),
                    "scope":"C2015_341_VT_POST-HOC", "note":"Threshold was not optimized for Challenge score"})
    save_json(ctx.out/"N4_challenge_score.json",{"status":"POST-HOC","rows":out})
    print("T10_N4=PASS; unoptimized_threshold=YES",flush=True)
    return out


def n5(ctx,cohorts):
    rows=[]
    for cohort,d in cohorts.items():
        p=d["p"];y=d["y"];rec=d["rec"]
        lam=float(ctx.lock["lambda"]["A"][.02]);supp=p<lam
        counts=Counter(rec[y==1]);groups={"one":[k for k,n in counts.items() if n==1],
                                          "two_plus":[k for k,n in counts.items() if n>=2]}
        for group,names in groups.items():
            belongs=np.isin(rec,names)&(y==1)
            n=int(belongs.sum());k=int((supp&belongs).sum())
            rows.append({"cohort":cohort,"group":group,
                         "positive_records":red(len(names)),"missed_true_k_n":safe_count_fraction(k,n),
                         "fnr":float(k/n) if n>=10 and k>=10 else None,
                         "scope":"POST-HOC","status":"OK" if n>=10 else "WITHHELD_SMALL_GROUP"})
        rec_value=record_fnr(p,y,rec,lam)
        rows.append({"cohort":cohort,"group":"all_positive_records",
                     "positive_records":red(len(counts)),
                     "missed_true_k_n":safe_count_fraction(int((supp&(y==1)).sum()),int((y==1).sum()),True),
                     "fnr":rec_value,"scope":"POST-HOC", "status":"RECORD_MEAN_FNR"})
    save_csv(ctx.out/"N5_record_multiplicity.csv",rows,
             ["cohort","group","positive_records","missed_true_k_n","fnr","scope","status"])
    print("T10_N5=PASS; groups_suppressed_below_10=YES",flush=True)
    return rows


def n7(ctx,cohorts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c=cohorts["calib"];np=ctx.np
    npos=len(set(c["rec"][c["y"]==1]));must(npos==188,"N7_POS_RECORDS_188")
    rows=[]
    for alpha in ALPHAS:
        k=int(math.floor(alpha*(npos+1)+1e-12))
        feasible=k>=1
        lam=float(ctx.conformal.crc_fnr_threshold(c["p"],c["y"],c["rec"],alpha)) if feasible else None
        for cohort in ("val","test","c2015"):
            d=cohorts[cohort]
            if not feasible:
                row={"alpha":alpha,"kstar":k,"feasible":False,"lambda":None,"cohort":cohort,
                     "suppression":None,"suppressed_false":"N/A","fnr":None,"fnr_count":"N/A","scope":"POST-HOC"}
            else:
                m=out_metrics(d["p"],d["y"],lam,ctx.metrics,cohort)
                row={"alpha":alpha,"kstar":k,"feasible":True,"lambda":lam,"cohort":cohort,
                     "suppression":m["suppression"],"suppressed_false":m["suppressed_false"],
                     "fnr":m["fnr"],"fnr_count":m["fnr_count"],"scope":"POST-HOC"}
            rows.append(row)
        if alpha in EXPECTED_LAMBDA_A:
            must(abs(lam-EXPECTED_LAMBDA_A[alpha])<=5e-9,"N7_FROZEN_ALPHA_"+str(alpha).replace(".",""))
    save_csv(ctx.out/"N7_suppression_alpha.csv",rows,
             ["alpha","kstar","feasible","lambda","cohort","suppression","suppressed_false","fnr","fnr_count","scope"])
    fig,ax=plt.subplots(figsize=(6.5,4.5),layout="constrained")
    for cohort,style,mark,label in (("val","-","o","VAL"),("test","--","^","TEST (post-hoc)"),
                                   ("c2015","-.","s","C2015 (post-hoc)")):
        data=[r for r in rows if r["cohort"]==cohort and r["feasible"]]
        ax.plot([r["alpha"] for r in data],[100*r["suppression"] for r in data],
                linestyle=style,marker=mark,ms=5,mfc="white",color="black",label=label)
    ax.set(xlabel=r"Nominal $\alpha$",ylabel="False-alarm suppression (%)",
           title="POST-HOC: suppression versus risk level")
    ax.grid(alpha=.2);ax.legend()
    import io
    dest=ctx.out/"N7_suppression_alpha.pdf"
    pdf=io.BytesIO()
    fig.savefig(pdf,format="pdf",metadata={"Title":"P05-T10 N7 POST-HOC",
        "CreationDate":None,"ModDate":None})
    forbid_overwrite_bytes(dest,pdf.getvalue())
    plt.close(fig)
    print("T10_N7=PASS; 8_ALPHAS; 3_COHORTS; STRICT_THRESHOLD",flush=True)
    return rows


def n8(ctx,cohorts):
    rows=[]
    tau=float(ctx.lock["ltt_comparator"]["tau"])
    must(abs(tau-.09)<1e-12,"N8_LTT_TAU_009")
    for split in ("test","c2015"):
        d=cohorts[split]
        for alpha in CORE_ALPHAS:
            lam=float(ctx.lock["lambda"]["A"][alpha])
            val=record_fnr(d["p"],d["y"],d["rec"],lam)
            rows.append({"cohort":split,"method":"CRC","alpha":alpha,"threshold":lam,
                         "rec_fnr":val,"n_pos_records":red(len(set(d["rec"][d["y"]==1]))),"scope":"POST-HOC"})
        rows.append({"cohort":split,"method":"LTT","alpha":.05,"threshold":tau,
                     "rec_fnr":record_fnr(d["p"],d["y"],d["rec"],tau,inclusive=True),
                     "n_pos_records":red(len(set(d["rec"][d["y"]==1]))),"scope":"POST-HOC"})
    save_csv(ctx.out/"N8_record_fnr.csv",rows,
             ["cohort","method","alpha","threshold","rec_fnr","n_pos_records","scope"])
    print("T10_N8=PASS; CRC_STRICT; LTT_INCLUSIVE",flush=True)
    return rows


def n9(ctx,c):
    p=c["p"];y=c["y"];r=c["rec"]
    positive_records=np.unique(r[y==1]); n=len(positive_records)
    must(n==188,"N9_188_POSITIVE_RECORDS")
    mins=np.sort(np.asarray([np.min(p[(r==rec)&(y==1)]) for rec in positive_records]))
    out=[]
    for alpha in CORE_ALPHAS:
        k=int(math.floor(alpha*(n+1)+1e-12))
        frac=float(ctx.conformal.crc_fnr_threshold(p,y,r,alpha))
        binary=float(mins[k-1]) if k>=1 else None
        diff=float(frac-binary) if binary is not None else None
        equal=binary is not None and math.isclose(frac,binary,rel_tol=0,abs_tol=1e-12)
        # Difference between record loss fractional (mean true alarms) and binary (any missed true alarm).
        partial_fractional=partial_binary=0
        for rec in positive_records:
            sel=(r==rec)&(y==1)
            nf=int(np.sum(p[sel]<frac))
            nb=int(np.sum(p[sel]<binary))
            if 0<nf<int(sel.sum()):partial_fractional+=1
            if 0<nb<int(sel.sum()):partial_binary+=1
        out.append({"alpha":alpha,"kstar":k,"lambda_fractional":frac,
                    "orderstat_binary_minimum":binary,"difference_fractional_minus_binary":diff,
                    "equal_1e_minus12":bool(equal),"partial_records_at_fractional_lambda":red(partial_fractional),
                    "partial_records_at_binary_orderstat":red(partial_binary),
                    "scope":"POST-HOC", "statement":
                    ("Thresholds coincide on these data" if equal else "Thresholds differ: binary Beta analysis is not exact for fractional-loss CRC")})
        must(abs(frac-EXPECTED_LAMBDA_A[alpha])<=5e-9,"N9_FROZEN_ALPHA_"+str(alpha).replace(".",""))
    save_json(ctx.out/"N9_fractional_vs_binary.json",{"status":"POST-HOC","rows":out,
        "note":"Binary record-minimum order statistic is diagnostic, not substituted into Eq.1"})
    print("T10_N9=PASS; NO_ASSUMED_EQUIVALENCE",flush=True)
    return out


def run_cpu(ctx):
    c,v,t,e=ctx.assert_locked_A()
    coh={"calib":c,"val":v,"test":t,"c2015":e}
    results={"N1":n1(ctx,c,v),"N4":n4(ctx,e),"N5":n5(ctx,coh),
             "N7":n7(ctx,coh),"N8":n8(ctx,coh),"N9":n9(ctx,c)}
    save_json(ctx.out/"T10_cpu_status.json",{"status":"COMPLETE_POST-HOC",
               "analyses":list(results),"note":"No input artifacts modified; only aggregated outputs except vector PDF"})
    ctx.check_unchanged()
    print("T10_CPU_N1_N4_N5_N7_N8_N9=PASS",flush=True)
