"""P05-T10 staged POST-HOC runner.

Stages: preflight | cpu | n6 | n3-train | n3-eval | n11-plan |
        n11-train | n11-eval | final | selftest.

All scripts MUST run in the pinned pn_b environment. Nothing here executes
T05b, changes the frozen lock, overwrites old checkpoints/scores or runs Git.
"""
from __future__ import annotations

# Set Windows and CUDA process environment before imported numerical packages.
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='8'

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from t10_shared import (Context,EXPECTED_LOCK,must,sha,save_json,
                        file_snapshot,set_priority,safe_exception)


def original_inputs(ctx):
    src=ctx.project/"outputs"/"t03"
    paths=[ctx.lock_path,ctx.project/"reports"/"TEST_OPENED.flag",
           ctx.proc/"test_labels.parquet",
           ctx.project/"outputs"/"t06"/"scores_test.parquet",
           ctx.project/"outputs"/"t06"/"scores_c2015.parquet"]
    for k,s in ctx.lock["checkpoints_sha256"].items():
        path=src/k/"ckpt.pt"
        must(path.is_file() and sha(path)==s,"FROZEN_CKPT_"+k.upper())
        paths.append(path)
        for split in ("calib","val"):
            paths.append(src/k/f"scores_{split}.parquet")
    must(all(p.is_file() for p in paths),"ORIGINAL_FILES_PRESENT")
    return paths


def guard(ctx,write=False):
    inputs=original_inputs(ctx)
    # Store relative paths, never Windows absolute paths, credentials or raw IDs.
    digest={p.relative_to(ctx.project).as_posix() if p.is_relative_to(ctx.project)
            else "external/"+p.name:sha(p) for p in inputs}
    path=ctx.out/"T10_frozen_guard.json"
    data={"frozen_lock_sha256":EXPECTED_LOCK,"input_file_sha256":digest,
          "test_flag_sha256":ctx.flag_sha,"note":"Original artifacts are read-only, snapshot before T10"}
    if write:
        save_json(path,data)
    if path.is_file():
        old=json.loads(path.read_text(encoding="utf-8"))
        must(data==old,"FROZEN_FILE_MANIFEST_UNCHANGED")
    else:
        must(not ctx.out.exists() or not any(ctx.out.glob('N*.json')),"GUARD_REQUIRED_BEFORE_T10")
    return data


def check_python_venv():
    import sys
    exe=str(Path(sys.executable).resolve()).lower().replace('\\','/')
    must('/pn_b/' in exe and sys.prefix != sys.base_prefix,"PINNED_PN_B_VENV")


def preflight(ctx):
    start=time.perf_counter()
    guard(ctx,write=False)
    ctx.assert_locked_A()
    must(not ctx.report.exists(),"T10_REPORT_NOT_PREEXISTING")
    must(not (ctx.out/"T10_frozen_guard.json").exists() or
         (ctx.out/"T10_frozen_guard.json").is_file(),"T10_GUARD_STATE")
    ctx.check_unchanged()
    print("T10_PREFLIGHT=PASS; cpu_only=YES; frozen_lock=YES; source_data_read_only=YES",flush=True)
    print("T10_PREFLIGHT_SECONDS=%.3f"%(time.perf_counter()-start),flush=True)


def stage(ctx,name):
    if name=="preflight":
        preflight(ctx);return
    if name=="final":
        finalize(ctx);return
    if name=="selftest":
        selftest(ctx);return
    guard(ctx,write=True)
    if name=="cpu":
        import t10_cpu
        t10_cpu.run_cpu(ctx)
    elif name=="n6":
        import t10_fingerprint
        t10_fingerprint.run_n6(ctx)
    elif name in ("n3-train","n3-eval","n11-plan","n11-train","n11-eval"):
        if name.startswith("n11-"):
            required=["T10_cpu_status.json","N3_summary.json","N6_fingerprint.json"]
            must(all((ctx.out/f).is_file() for f in required),"N11_ONLY_AFTER_N1_N9")
        import t10_gpu
        if name=="n3-train":t10_gpu.n3_train(ctx)
        elif name=="n3-eval":t10_gpu.n3_eval(ctx)
        elif name=="n11-plan":t10_gpu.n11_plan(ctx)
        elif name=="n11-train":t10_gpu.n11_train(ctx)
        elif name=="n11-eval":t10_gpu.n11_eval(ctx)
    else:raise RuntimeError("UNKNOWN_T10_STAGE")
    guard(ctx,write=False)
    ctx.check_unchanged()
    print("T10_STAGE_"+name.upper().replace('-','_')+"=PASS; POST-HOC",flush=True)


def run_verifier(ctx):
    import subprocess
    python=sys.executable
    script=ctx.repo/"scripts"/"verify_key_numbers.py"
    must(script.is_file(),"HF5_VERIFIER_PRESENT")
    for name,cmd in (("ORIGINAL",[]),("ASSETS",["--assets",str(ctx.assets)])):
        result=subprocess.run([python,"-B",str(script)]+cmd,cwd=str(ctx.repo),
             capture_output=True,text=True,timeout=180,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        okay=result.returncode==0 and "VERIFY_FINAL=PASS checks=12/12" in result.stdout
        must(okay,"VERIFY_12_12_"+name)
    return {"original":"PASS 12/12","assets":"PASS 12/12"}


def concise_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def paper_sentences(ctx):
    sentences=[]
    n1=concise_json(ctx.out/"N1_summary.json")
    n3=concise_json(ctx.out/"N3_fcn_crc_results.csv") if False else None
    n9=concise_json(ctx.out/"N9_fractional_vs_binary.json")
    if n1:
        d=n1['cohorts']['val'];q=d['quantiles_prob']['5'];rho=d['spearman_logitA_minus_B_vs_vote_entropy'];ci=d['spearman_ci_record_bootstrap_95']
        rho_text='not estimable' if rho is None else f'{rho:.3f}'
        ci_text='not estimable' if ci is None else f'[{ci[0]:.3f}, {ci[1]:.3f}]'
        sentences.append('N1: In a post-hoc validation analysis, the fifth percentile of true-VT scores '
          f'was {q["p_A"]:.6g} for hard labels and {q["p_B"]:.6g} for soft labels '
          f'(B minus A: {q["delta_B_minus_A"]:+.6g}); the record-bootstrap Spearman '
          f'correlation with cached binary-proxy vote entropy was {rho_text} (95% CI {ci_text}). '
          'Full categorical vote entropy was not available in the frozen cache.')
    fcn=ctx.out/'N3_fcn_crc_results.csv'
    if fcn.is_file():
        import pandas as pd
        df=pd.read_csv(fcn)
        t=df[(df['cohort']=='test')&(abs(df['alpha']-.02)<1e-12)]
        e=df[(df['cohort']=='c2015')&(abs(df['alpha']-.02)<1e-12)]
        if len(t)==1 and len(e)==1:
            x=t.iloc[0];z=e.iloc[0]
            sentences.append('N3: A post-hoc FCN trained with the frozen T02c procedure and separately '
                'calibrated using record-level CRC at alpha = 0.02 suppressed '
                f'{100*x["suppression"]:.1f}% of false alarms on VTaC test and '
                f'{100*z["suppression"]:.1f}% on Challenge 2015; corresponding '
                f'FNRs were {100*x["fnr"]:.1f}% and {100*z["fnr"]:.1f}%, respectively. '
                'These results were not preregistered.')
    if n9:
        vals=n9['rows']
        equivalent=all(row['equal_1e_minus12'] for row in vals)
        if equivalent:
            sentences.append('N9: A post-hoc calibration check found that the '
                'fractional record-level CRC thresholds coincided with the binary '
                'positive-record minimum-score order statistics at alpha = 0.01, 0.02, and 0.05; '
                'the equivalence is empirical for this dataset, not a general identity.')
        else:
            differing=', '.join(f'{r["alpha"]:.2f}' for r in vals if not r['equal_1e_minus12'])
            sentences.append('N9: A post-hoc check found that fractional record-level CRC and '
                'binary positive-record minimum-score order statistics produced different thresholds '
                f'at alpha = {differing}; thus the binary Beta approximation cannot be asserted to '
                'describe the fractional loss exactly.')
    return sentences



def report_tables(ctx):
    import csv
    tables=[]
    def fmt(value):
        if value is None or value=='':return 'N/A'
        if isinstance(value,bool):return 'yes' if value else 'no'
        if isinstance(value,float):return f'{value:.8g}'
        return str(value).replace('|','/').replace('\n',' ')
    def add(title,rows,fields):
        tables.extend(['',title,'','| '+' | '.join(fields)+' |','|'+ '|'.join(['---']*len(fields))+'|'])
        for row in rows:
            tables.append('| '+' | '.join(fmt(row.get(k)) for k in fields)+' |')
    n1=concise_json(ctx.out/'N1_summary.json')
    if n1:
        rows=[]
        for split,part in n1['cohorts'].items():
            for q,values in part['quantiles_prob'].items():
                rows.append({'split':split,'percentile':q,**values})
        add('### N1: true-alarm score percentiles, POST-HOC',rows,
            ['split','percentile','p_A','p_B','delta_B_minus_A'])
        add('### N1: record-clustered Spearman, vote entropy PROXY',
            [{'split':split,'rho':part['spearman_logitA_minus_B_vs_vote_entropy'],
              'ci_95':str(part['spearman_ci_record_bootstrap_95']),
              'valid_reps':part['bootstrap_valid']}
             for split,part in n1['cohorts'].items()],['split','rho','ci_95','valid_reps'])
        add('### N1: frozen A/B thresholds and vote strata',
            [{'split':split,'lambda_A_002':part['lambda_A_002'],
              'lambda_B_002':part['lambda_B_002'],
              'n_true_contested':part['n_true_contested'],
              'n_true_noncontested':part['n_true_noncontested'],
              'n_true_vote_missing':part['n_true_vote_missing']}
             for split,part in n1['cohorts'].items()],
            ['split','lambda_A_002','lambda_B_002','n_true_contested',
             'n_true_noncontested','n_true_vote_missing'])
        tables.append('Vote entropy is H(q_mean), a binary proxy. Categorical vote entropy is not recoverable from the cache.')
    def from_csv(name):
        with (ctx.out/name).open('r',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    add('### N3: independent FCN CRC, POST-HOC',from_csv('N3_fcn_crc_results.csv'),
        ['cohort','alpha','lambda_fcn','auroc','suppression','suppressed_false','fnr','fnr_count','fnr_ci_low','fnr_ci_high'])
    add('### N4: Challenge score at frozen thresholds, POST-HOC',
        concise_json(ctx.out/'N4_challenge_score.json')['rows'],
        ['alpha','lambda','challenge_score','TP','TN','FP','FN','FN_FNR_WHOLE_COHORT'])
    tables.append('Challenge-score thresholds were not optimized for this metric.')
    add('### N5: positive-record alarm multiplicity, POST-HOC',
        from_csv('N5_record_multiplicity.csv'),
        ['cohort','group','positive_records','missed_true_k_n','fnr','status'])
    add('### N6: C2015 overlap fingerprint threshold sensitivity, POST-HOC',
        concise_json(ctx.out/'N6_fingerprint.json')['rows'],
        ['threshold','flagged_alarms','flagged_recordings','suppression','suppression_k_n','fnr','fnr_k_n'])
    add('### N7: suppression versus alpha, POST-HOC',
        from_csv('N7_suppression_alpha.csv'),
        ['alpha','kstar','feasible','lambda','cohort','suppression','fnr','fnr_count'])
    add('### N8: mean recording-level FNR, POST-HOC',
        from_csv('N8_record_fnr.csv'),['cohort','method','alpha','threshold','rec_fnr'])
    add('### N9: fractional-loss CRC versus binary order statistic, POST-HOC',
        concise_json(ctx.out/'N9_fractional_vs_binary.json')['rows'],
        ['alpha','kstar','lambda_fractional','orderstat_binary_minimum',
         'difference_fractional_minus_binary','equal_1e_minus12',
         'partial_records_at_fractional_lambda','partial_records_at_binary_orderstat'])
    if (ctx.out/'N11_evaluation.json').exists():
        add('### N11: optional matched-size model comparison, POST-HOC',
            concise_json(ctx.out/'N11_evaluation.json')['rows'],
            ['cohort','evaluated_profile','model','alpha','lambda','suppression','fnr','status'])
    return tables


def finalize(ctx):
    must(not ctx.report.exists(),"FINAL_REPORT_ABSENT")
    required=['T10_cpu_status.json','N1_summary.json','N1_cdf.pdf','N4_challenge_score.json',
              'N5_record_multiplicity.csv','N6_fingerprint.json','N7_suppression_alpha.csv',
              'N7_suppression_alpha.pdf','N8_record_fnr.csv','N9_fractional_vs_binary.json',
              'N3_fcn_crc_results.csv','N3_summary.json']
    for name in required:must((ctx.out/name).is_file(),"FINAL_N1_N9_"+name.split('.')[0].upper())
    # N11 optional, only evaluate if a complete set of 5 checkpoints exists.
    n11_status='NOT_RUN_OPTIONAL'
    n11_eval=ctx.out/'N11_evaluation.json'
    if n11_eval.exists():n11_status='POST-HOC_COMPLETE'
    elif (ctx.out/'n11').exists():n11_status='INCOMPLETE_OR_DEFERRED'
    verification=run_verifier(ctx)
    guard(ctx,write=False)
    ctx.check_unchanged()
    files=[]
    for path in sorted(ctx.out.rglob('*')):
        if path.is_file():
            # Never print hash-list content or per-checkpoint files.
            files.append((path.relative_to(ctx.project).as_posix(),sha(path),path.stat().st_size))
    lines=["# P05-T10 post-hoc analyses", "", "**Status:** COMPLETE; WAIT_FOR_DIRECTOR; NO_PUSH", "",
           "All analyses are POST-HOC. The frozen test and Challenge 2015 labels were already opened before T10.",
           "These are exploratory results and do not update the preregistered primary endpoint, thresholds or calibration guarantee.",
           "",f"Original final_lock SHA256: `{EXPECTED_LOCK}`", "", "## Analysis inventory", "",
           "| ID | Result artifact | Scope |", "|---|---|---|",
           "| N1 | N1_summary.json, N1_cdf.pdf | Calibration and validation, true alarms; vote entropy PROXY |",
           "| N3 | N3_fcn_crc_results.csv, N3_summary.json | FCN 5-seed POST-HOC, VAL/TEST/C2015 |",
           "| N4 | N4_challenge_score.json | Challenge score, POST-HOC, thresholds not score-optimized |",
           "| N5 | N5_record_multiplicity.csv | Recording multiplicity, POST-HOC |",
           "| N6 | N6_fingerprint.json | Original T05b FFT method, thresholds 0.90/0.95/0.98; POST-HOC |",
           "| N7 | N7_suppression_alpha.csv/.pdf | 8 risk levels, VAL/TEST/C2015 POST-HOC |",
           "| N8 | N8_record_fnr.csv | Test/C2015 CRC and LTT record-level mean FNR; POST-HOC |",
           "| N9 | N9_fractional_vs_binary.json | Fractional CRC versus binary order statistic; POST-HOC |",
           f"| N11 | N11_evaluation.json if present | {n11_status}; optional POST-HOC |",
           "", "## Verification", "",
           f"- verify_original: {verification['original']}",f"- verify_assets: {verification['assets']}",
           "- final_lock.yaml SHA256: PASS", "- TEST_OPENED.flag unchanged: PASS",
           "- Input checkpoint and score SHA256 manifest unchanged: PASS", "",
           "## Restricted disclosure", "",
           "No raw waveforms, VTaC/Challenge record IDs or event IDs are written by T10.",
           "All subgroup counts below 10 are displayed as `<10`. Whole-cohort FNR k/n is permitted.",
           "Training checkpoints are new and confined to outputs/t10/.",
           "Never distribute outputs/t10/n11/N11_selected_record_hashes.csv as public data; hashes are pseudonyms.",
           "", "## Numerical source artifacts and hashes", "",
           "| Relative path | SHA-256 | Bytes |", "|---|---|---:|"]
    for p,h,n in files:lines.append(f'| `{p}` | `{h}` | {n} |')
    lines.extend(['','## Consolidated numerical results (all POST-HOC)',''])
    lines.extend(report_tables(ctx))
    runtime=concise_json(ctx.out/'T10_stage_runtime.json')
    if runtime:
        lines.extend(['','## Stage runtimes','',
                      '| Stage | Wall seconds |','|---|---:|'])
        for stage_name,seconds in runtime.items():
            lines.append(f'| {stage_name} | {seconds:.3f} |')
    lines.extend(['','## Manuscript-ready sentences (POST-HOC, Director must approve)',''])
    lines += ['- '+s for s in paper_sentences(ctx)]
    lines.extend(['','=== REPORT P05-T10 ===',
                  'N1,N3,N4,N5,N6,N7,N8,N9: PASS (all POST-HOC)',
                  'N11: '+n11_status,
                  'verify_original: PASS 12/12','verify_assets: PASS 12/12',
                  'final_lock: SHA256 UNCHANGED','TEST_OPENED.flag: UNCHANGED',
                  'status COMPLETE; WAIT_FOR_DIRECTOR; NO_PUSH','=== END ===',''])
    with ctx.report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    print('=== REPORT P05-T10 ===',flush=True)
    print('N1..N9=PASS; N11='+n11_status,flush=True)
    print('verify_original: PASS 12/12',flush=True)
    print('verify_assets: PASS 12/12',flush=True)
    print('final_lock_sha256: '+EXPECTED_LOCK,flush=True)
    print('report_sha256: '+sha(ctx.report),flush=True)
    print('status COMPLETE; WAIT_FOR_DIRECTOR; NO_PUSH',flush=True)
    print('=== END ===',flush=True)


def selftest(ctx):
    import numpy as np
    from t10_shared import sigmoid,record_fnr
    must(np.allclose(sigmoid(np.array([0.,1.])),[.5,.7310585786]),'SELFTEST_SIGMOID')
    must(abs(record_fnr([.01,.09,.07],[1,1,1],['x','x','y'],.05)-.25)<1e-12,'SELFTEST_RECORD_FNR')
    must(ctx.lock['lambda']['A'][.02]==.0139079025,'SELFTEST_FROZEN_PRIMARY')
    print('T10_SELFTEST=PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stage',required=True,
        choices=['preflight','cpu','n6','n3-train','n3-eval','n11-plan','n11-train','n11-eval','final','selftest'])
    args=parser.parse_args()
    set_priority()
    check_python_venv()
    ctx=Context(args.project)
    must(Path.cwd().resolve()==ctx.workspace,'T10_EXECUTION_CWD_WORKSPACE')
    started=time.perf_counter()
    try:
        stage(ctx,args.stage)
        if args.stage not in ('preflight','selftest','final'):
            runtime=ctx.out/'T10_stage_runtime.json'
            old=json.loads(runtime.read_text(encoding='utf-8')) if runtime.exists() else {}
            old[args.stage]=float(time.perf_counter()-started)
            runtime.parent.mkdir(parents=True,exist_ok=True)
            # Mutable T10-only stage bookkeeping. Original study artifacts stay read-only.
            tmp=runtime.with_suffix('.json.new')
            tmp.write_text(json.dumps(old,sort_keys=True,indent=2)+'\n',encoding='utf-8')
            os.replace(tmp,runtime)
    except BaseException as exc:
        safe_exception(exc)
        print('T10_STATUS=AUDIT_HOLD; NO_PUSH',flush=True)
        raise SystemExit(2)
    finally:
        print(f'T10_{args.stage.upper().replace("-","_")}_RUNTIME_SECONDS={time.perf_counter()-started:.2f}',flush=True)

if __name__=='__main__':main()
