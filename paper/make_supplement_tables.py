import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import argparse
import yaml
ROOT = Path(__file__).resolve().parents[1]
PTH = yaml.safe_load((ROOT / "configs" / "paths.yaml").read_text(encoding="utf-8"))
OUT = (ROOT / str(PTH["outputs"])).resolve()
PROC = (ROOT / str(PTH["data_proc"])).resolve()
_parser = argparse.ArgumentParser()
_parser.add_argument("--out", default="paper_outputs/")
DEST = Path(_parser.parse_args().out)
DEST.mkdir(parents=True, exist_ok=True)
sig = lambda z: 1 / (1 + np.exp(-z))


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def frac(k, n):
    return f"{int(k)}/{int(n)}"


lines = []
w = lines.append

# ---- S2 ablations (validation, ensemble)
vm = pd.read_parquet(PROC / "val_meta.parquet", columns=["idx", "y"])
zs = []
for s in range(5):
    d = pd.read_parquet(OUT / "t03" / f"A_s{s}" / "scores_val.parquet")
    zs.append(vm[["idx"]].merge(d, on="idx", how="left", validate="1:1")["logit"].to_numpy(float))
pv = sig(np.mean(zs, 0)); yv = vm.y.to_numpy(int)
ref_auc, ref_ap = roc_auc_score(yv, pv), average_precision_score(yv, pv)
lam = 0.0139079025
ref_sup = (pv[yv == 0] < lam).mean(); ref_k = int(((pv < lam) & (yv == 1)).sum())
ab = pd.read_parquet(OUT / "t05a" / "ablation_summary.parquet")
names = {"no_ppg": "Without PPG", "no_abp": "Without ABP", "ecg_only": "ECG only",
         "w20_d0": "20 s window", "w10_d5": "10 s + 5 s after onset", "w10_d10": "10 s + 10 s after onset"}
w(r"\begin{table}[H]\centering\caption{Ablations on Validation (Five-Seed Ensembles, $\alpha=0.02$)}\label{tab:s-ablation}")
w(r"\begin{tabular}{lcccc}\toprule Configuration & AUROC & AUPRC & Supp. & FNR\\\midrule")
w(f"Locked model (all signals, 10 s) & {ref_auc:.3f} & {ref_ap:.3f} & {pct(ref_sup)} & {frac(ref_k, yv.sum())}\\\\")
for _, r in ab.iterrows():
    k = int(round(r.fnr_ensemble * yv.sum()))
    w(f"{names[r['name']]} & {r.auroc_ensemble:.3f} & {r.auprc_ensemble:.3f} & {pct(r.fa_suppression_ensemble)} & {frac(k, yv.sum())}\\\\")
w(r"\bottomrule\multicolumn{5}{p{0.9\linewidth}}{\footnotesize Each ablation has its own recalibrated CRC threshold. Samples after onset were never used by the locked model.}\end{tabular}\end{table}")
w("")

(DEST / "supp_tab_ablation.tex").write_text("\n".join(lines), encoding="utf-8")
lines = []
w = lines.append

# ---- S5 event vs record CRC
e3 = pd.read_parquet(OUT / "t07" / "E3_crc_event_vs_record.parquet")
coh = {"val": "Validation", "test": "Test$^{\\dagger}$", "c2015": "Challenge$^{\\dagger}$"}
w(r"\begin{table}[H]\centering\caption{Record-Level vs Event-Level CRC}\label{tab:s-event}")
w(r"\begin{tabular}{llcccccc}\toprule $\alpha$ & Data & \multicolumn{3}{c}{Record-level (locked)} & \multicolumn{3}{c}{Event-level}\\")
w(r" & & Supp. & FNR & Rec.\ FNR & Supp. & FNR & Rec.\ FNR\\\midrule")
for a in (0.01, 0.02, 0.05):
    for c in ("val", "test", "c2015"):
        rr = e3[(e3.alpha == a) & (e3.cohort == c) & (e3.method == "record_CRC_locked")].iloc[0]
        ee = e3[(e3.alpha == a) & (e3.cohort == c) & (e3.method == "event_CRC")].iloc[0]
        w(f"{a:.2f} & {coh[c]} & {pct(rr.suppression)} & {frac(rr.fnr_k, rr.fnr_n)} & {pct(rr.record_mean_fnr)} & "
          f"{pct(ee.suppression)} & {frac(ee.fnr_k, ee.fnr_n)} & {pct(ee.record_mean_fnr)}\\\\")
th = pd.read_parquet(OUT / "t07" / "E3_thresholds.parquet")
tt = "; ".join(f"$\\alpha={r.alpha:.2f}$: {r.lambda_record:.4g} vs {r.lambda_event:.4g}" for _, r in th.iterrows())
w(r"\bottomrule\multicolumn{8}{p{0.95\linewidth}}{\footnotesize Thresholds (record vs event): " + tt +
  r". Calibration had 353 true alarms in 188 positive recordings. Rec.\ FNR: mean over positive recordings of the share of their true alarms suppressed. $^{\dagger}$Post-hoc.}\end{tabular}\end{table}")
w("")

# ---- S6 bSQI tertiles
e5 = pd.read_parquet(OUT / "t07" / "E5_tertiles.parquet")
coh2 = {"calib": "Calibration", "val": "Validation", "test": "Test$^{\\dagger}$", "c2015": "Challenge$^{\\dagger}$"}
w(r"\begin{table}[H]\centering\caption{Suppression and FNR by ECG Signal-Quality Tertile at the Locked Threshold ($\alpha=0.02$)}\label{tab:s-bsqi}")
w(r"\begin{tabular}{llcccc}\toprule Data & bSQI tertile & Alarms (true) & Supp. & FNR & 95\% CI\\\midrule")
lab = {1: "low ($\\le 0.889$)", 2: "middle", 3: "high ($> 0.973$)"}
for c in ("calib", "val", "test", "c2015"):
    for _, r in e5[e5.cohort == c].sort_values("tertile").iterrows():
        w(f"{coh2[c]} & {lab[int(r.tertile)]} & {int(r.n_events)} ({int(r.n_true)}) & {pct(r.suppression)} & "
          f"{frac(r.fnr_k, r.fnr_n)} & {pct(r.fnr_ci95_lo)}--{pct(r.fnr_ci95_hi)}\\\\")
w(r"\bottomrule\multicolumn{6}{p{0.95\linewidth}}{\footnotesize bSQI: agreement of the xqrs and gqrs QRS detectors ($\pm$150 ms) on the 10 s input, maximum over available ECG leads; tertile cut points fixed on calibration data. 18 alarms without usable ECG were kept with bSQI $=0$. $^{\dagger}$Post-hoc.}\end{tabular}\end{table}")
w("")

# ---- S7 latency
e7 = pd.read_parquet(OUT / "t07" / "E7_latency.parquet")
pp = pd.read_parquet(OUT / "t07" / "E7_preprocessing_latency.parquet").iloc[0]
cfg = {"CPU_1_THREAD": "CPU, 1 thread", "CPU_8_THREADS": "CPU, 8 threads", "GPU1": "GPU (Quadro M6000)"}
w(r"\begin{table}[H]\centering\caption{Inference Latency per Alarm (Batch Size 1)}\label{tab:s-latency}")
w(r"\begin{tabular}{lcc}\toprule Step and hardware & Median (ms) & 95th percentile (ms)\\\midrule")
for _, r in e7.iterrows():
    w(f"Five-model ensemble, {cfg[r.configuration]} & {r.median_ms:.1f} & {r.p95_ms:.1f}\\\\")
w(f"Pre-processing, CPU, 1 thread & {pp.median_ms:.1f} & {pp.p95_ms:.1f}\\\\")
w(r"\bottomrule\multicolumn{3}{p{0.9\linewidth}}{\footnotesize Intel Xeon E5-2680 v4, PyTorch 2.0.1; 20 warm-up and 500 measured runs (100 for pre-processing). Reading WFDB files is excluded.}\end{tabular}\end{table}")
w("")

# ---- S8 missed cases
e8 = pd.read_parquet(OUT / "t07" / "E8_missed_summary.parquet")
w(r"\begin{table}[H]\centering\caption{True VT Alarms Suppressed at the Locked Threshold on Test and External Data (Post-Hoc)}\label{tab:s-missed}")
w(r"\begin{tabular}{lllccc}\toprule Case & Data & Missing slots & bSQI & Ensemble $p$ & Seeds with $p\ge\hat\lambda$\\\midrule")
for _, r in e8.iterrows():
    seeds = [r[f"p_seed_{i}"] for i in range(5)]
    nabove = sum(x >= r.lambda_locked for x in seeds)
    data = "VTaC test" if r.cohort == "test" else "Challenge 2015"
    ms = "none" if r.missing_slots == "NONE" else r.missing_slots
    w(f"{int(r.case)} & {data} & {ms} & {r.bsqi:.2f} & {r.p_ensemble:.5f} & {nabove} of 5\\\\")
w(r"\bottomrule\multicolumn{6}{p{0.95\linewidth}}{\footnotesize Locked threshold $\hat\lambda=0.0139$ ($\alpha=0.02$). The ensemble averages the five seed logits before the sigmoid, so single seeds above $\hat\lambda$ need not lift the ensemble above it.}\end{tabular}\end{table}")

(DEST / "supplement_tables.tex").write_text("\n".join(lines), encoding="utf-8")
print("written", DEST / "supplement_tables.tex", "ref AUROC", round(ref_auc, 4), "AUPRC", round(ref_ap, 4), "supp", round(ref_sup, 4), "k", ref_k)
