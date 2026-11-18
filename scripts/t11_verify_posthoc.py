"""Public aggregate-only T10/T11 checks appended to the original 12 verifier gates.

No per-event or per-record data are loaded. Numerical checks are consistency
checks of released summaries; original 12 gates independently use frozen scores.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path

ALPHAS=(.01,.015,.02,.03,.04,.05,.07,.10)
PRIMARY={.01:7.61028e-5,.02:.0139079025,.05:.0314556314}
COUNTS={'val':(354,141),'test':(345,137),'c2015':(252,89)}


def check_posthoc(repo, lock, emit):
    root=Path(repo)/'results'/'t10'
    def obj(name):
        p=root/name
        assert p.is_file(), 'missing public summary'
        return json.loads(p.read_text(encoding='utf-8'))
    def rows(name):
        p=root/name
        assert p.is_file(), 'missing public table'
        with p.open('r',newline='',encoding='utf-8') as f:
            return list(csv.DictReader(f))
    def near(x,y,atol=1e-10):
        try:return math.isfinite(float(x)) and abs(float(x)-float(y))<=atol
        except (TypeError,ValueError,OverflowError):return False
    def count(text):
        a,b=str(text).split('/')
        return int(a),int(b)
    def e(name,actual,expected,ok):
        emit('T11_'+name,actual,expected,bool(ok))

    n4=obj('N4_challenge_score.json')['rows']
    e('N4_2_ROWS',len(n4),2,len(n4)==2)
    for r,alpha,expected in zip(n4,(.02,.05),(.696,.756)):
        e('N4_SCORE_'+str(alpha),f'{float(r["challenge_score"])*100:.4f}','~'+str(expected*100),
          near(float(r['challenge_score']),expected,0.00055))
        k,n=count(r['FN_FNR_WHOLE_COHORT'])
        tp=int(r['TP']);tn=int(r['TN']);fp=int(r['FP'])
        expected_score=(tp+tn)/(tp+tn+fp+5*k)
        e('N4_MATRIX_'+str(alpha),f'{tp+int(k)},{tn+fp}', '89,252',
          tp+k==89 and tn+fp==252 and near(expected_score,r['challenge_score'],1e-12))

    n7=rows('N7_suppression_alpha.csv')
    e('N7_ROWS',len(n7),24,len(n7)==24)
    groups={(float(r['alpha']),r['cohort']):r for r in n7}
    e('N7_GRID',len(groups),'8x3=24',len(groups)==24 and
      set(groups)=={(a,c) for a in ALPHAS for c in COUNTS})
    previous={c:(-1.,-1.) for c in COUNTS}
    consistent=True; anchors=True
    for alpha in ALPHAS:
        for cohort,(neg,pos) in COUNTS.items():
            r=groups.get((alpha,cohort))
            if r is None:consistent=False;continue
            try:
                kstar=math.floor(alpha*189+1e-12)
                lam=float(r['lambda']);supp=float(r['suppression']);fnr=float(r['fnr'])
                n=(pos if cohort!='val' else 141)
                cnt=count(r['fnr_count'])
                sf=r['suppressed_false']
                sf_ok=True if sf=='<10' else (count(sf)[1]==neg and near(count(sf)[0]/neg,supp,1e-12))
                common=(str(r['feasible']).lower()=='true' and int(r['kstar'])==kstar and
                        0<=supp<=1 and 0<=fnr<=1 and cnt[1]==n and
                        near(cnt[0]/n,fnr,1e-12) and sf_ok)
                if supp+1e-12<previous[cohort][0] or fnr+1e-12<previous[cohort][1]: common=False
                previous[cohort]=(supp,fnr)
                consistent &= common
                if near(alpha,.02,1e-12):
                    targets={'val':('211/354',(2,141)),'test':('191/345',(1,137)),
                             'c2015':('156/252',(2,89))}
                    ts,tf=targets[cohort]
                    anchors &= (sf==ts and cnt==tf)
                if alpha in PRIMARY: anchors &= near(lam,PRIMARY[alpha],5e-9)
            except (KeyError,ValueError,TypeError,ZeroDivisionError):consistent=False
    e('N7_NUMERIC_MONOTONIC','PASS' if consistent else 'FAIL','PASS',consistent)
    e('N7_FROZEN_ANCHORS','PASS' if anchors else 'FAIL','PASS',anchors)

    n8=rows('N8_record_fnr.csv')
    e('N8_ROWS',len(n8),8,len(n8)==8)
    n8_valid=True
    for cohort in ('test','c2015'):
        cr=[r for r in n8 if r['cohort']==cohort and r['method']=='CRC']
        lt=[r for r in n8 if r['cohort']==cohort and r['method']=='LTT']
        cr.sort(key=lambda z:float(z['alpha']))
        n8_valid &= (len(cr)==3 and len(lt)==1 and
                     [float(r['alpha']) for r in cr]==[.01,.02,.05])
        try:
            n8_valid &= all(0<=float(r['rec_fnr'])<=1 for r in cr+lt)
            n8_valid &= all(near(float(r['threshold']),PRIMARY[float(r['alpha'])],5e-9) for r in cr)
            n8_valid &= near(float(lt[0]['threshold']),.09)
            n8_valid &= all(float(cr[i]['rec_fnr'])<=float(cr[i+1]['rec_fnr'])+1e-12 for i in range(2))
        except (ValueError,KeyError,IndexError):n8_valid=False
    e('N8_BOTH_COHORTS','PASS' if n8_valid else 'FAIL','PASS',n8_valid)

    n9=obj('N9_fractional_vs_binary.json')['rows']
    e('N9_ROWS',len(n9),3,len(n9)==3)
    n9_valid=True
    for r,a,k in zip(n9,(.01,.02,.05),(1,3,9)):
        try:
            f=float(r['lambda_fractional']);b=float(r['orderstat_binary_minimum'])
            equal=abs(f-b)<=1e-12
            n9_valid &= near(r['alpha'],a) and int(r['kstar'])==k
            n9_valid &= near(f,PRIMARY[a],5e-9)
            n9_valid &= near(r['difference_fractional_minus_binary'],f-b,1e-15)
            n9_valid &= (r['equal_1e_minus12'] is equal)
        except (KeyError,TypeError,ValueError):n9_valid=False
    e('N9_FRACTIONAL_BINARY','PASS' if n9_valid else 'FAIL','PASS',n9_valid)

    n3=rows('N3_fcn_crc_results.csv')
    summary=obj('N3_summary.json')
    e('N3_9_ROWS',len(n3),9,len(n3)==9)
    e('N3_5_SEEDS',summary.get('seeds'),'[0..4]',summary.get('seeds')==[0,1,2,3,4])
    n3_valid=set(); counts_ok=True
    for r in n3:
        try:
            a=float(r['alpha']);cohort=r['cohort'];n3_valid.add((a,cohort))
            nf,nt=COUNTS[cohort]
            lam=float(summary['lambda_fcn'][str(a)])
            counts_ok &= near(r['lambda_fcn'],lam)
            sup=float(r['suppression']);fnr=float(r['fnr']);auc=float(r['auroc'])
            counts_ok &= all(0<=v<=1 for v in (sup,fnr,auc))
            ck,cn=count(r['fnr_count']);counts_ok &= cn==nt and near(ck/nt,fnr,1e-12)
            sf=r['suppressed_false']
            if sf!='<10':
                sk,sn=count(sf);counts_ok &= sn==nf and near(sk/nf,sup,1e-12)
        except (KeyError,TypeError,ValueError,ZeroDivisionError): counts_ok=False
    e('N3_GRID',len(n3_valid),'9',n3_valid=={(a,c) for a in (.01,.02,.05) for c in COUNTS})
    e('N3_METRIC_ARITHMETIC','PASS' if counts_ok else 'FAIL','PASS',counts_ok)

    n6=obj('N6_fingerprint.json')['rows']
    e('N6_ROWS',len(n6),3,len(n6)==3)
    n6_ok=True
    for r,a,anchor in zip(n6,(.90,.95,.98),('154/249','155/251','156/252')):
        try:
            n6_ok &= near(r['threshold'],a) and r['suppression_k_n']==anchor
            k,n=count(anchor)
            n6_ok &= near(float(r['suppression']),k/n,1e-12)
            n6_ok &= r['flagged_alarms']=='<10' and r['flagged_recordings']=='<10'
        except (KeyError,TypeError,ValueError):n6_ok=False
    e('N6_FINGERPRINT_ANCHORS','PASS' if n6_ok else 'FAIL','PASS',n6_ok)

    n5=obj('N5_calib_corrected.json')
    n5_ok=(n5.get('status')=='POST-HOC_CORRECTION' and
           n5.get('positive_calibration_records')==188 and
           n5.get('true_calibration_alarms')==353 and
           near(n5.get('fractional_record_loss_sum'),2.5,1e-8) and
           near(n5.get('crc_upper_bound'),
                (float(n5.get('fractional_record_loss_sum'))+1)/189,1e-12) and
           float(n5.get('crc_upper_bound',1))<=.02+1e-12 and
           near(n5.get('record_mean_fnr'),
                float(n5.get('fractional_record_loss_sum'))/188,1e-12) and
           n5.get('unchanged_cohort_fnr')=={'val':'2/141','test':'1/137','c2015':'2/89'})
    try:
        mk,mn=count(n5['fnr_k_n']); n5_ok &= mk==n5['suppressed_true_alarms'] and mn==353
        n5_ok &= near(n5['lambda_full_precision'],PRIMARY[.02],5e-9)
    except (KeyError,ValueError,TypeError):n5_ok=False
    e('N5_CALIB_CORRECTION','PASS' if n5_ok else 'FAIL','PASS',n5_ok)

    n1=obj('N1_summary.json')
    n1_ok=n1.get('entropy_definition','').startswith('H(q_mean)')
    for cohort in ('calib','val'):
        try:
            c=n1['cohorts'][cohort]
            n1_ok &= c['bootstrap_requested']==2000 and c['bootstrap_valid']>=100
            n1_ok &= set(c['quantiles_prob'])=={'1','2','5','10','25'}
            for v in c['quantiles_prob'].values():
                n1_ok &= 0<=float(v['p_A'])<=1 and 0<=float(v['p_B'])<=1
                n1_ok &= near(v['delta_B_minus_A'],float(v['p_B'])-float(v['p_A']),1e-12)
        except (KeyError,TypeError,ValueError):n1_ok=False
    e('N1_VOTE_ENTROPY_PROXY','PASS' if n1_ok else 'FAIL','PASS',n1_ok)

    n11=obj('N11_evaluation.json');plan=obj('N11_plan.json')
    n11_ok=(n11.get('status')=='POST-HOC' and len(n11.get('rows',[]))==9 and
            plan.get('status')=='POST-HOC' and plan.get('seed')==42 and
            plan.get('selected_record_count')==plan.get('matched_record_count') and
            int(plan.get('matched_record_count',0))>=10)
    e('N11_MATCHED_SUMMARIES','PASS' if n11_ok else 'FAIL','PASS',n11_ok)
    print('VERIFY_T11_AGGREGATE_CHECKS=COMPLETE; POST-HOC; NO_PATIENT_DATA',flush=True)
