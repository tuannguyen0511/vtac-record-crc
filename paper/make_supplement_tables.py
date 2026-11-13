"""Export archived supplementary tables. No new analysis."""
import argparse
import re
import shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'configs'/'paths.yaml').read_text(encoding='utf-8'))
OUT=(ROOT/CFG['outputs']).resolve()
PAT=re.compile(r'(?i)(?:supp(?:lementary)?[-_ ]*table|table[-_ ]*s)[-_ ]?([1-8])(?:[^0-9]|$)')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    found={}
    for base in (ROOT/'paper'/'original_sources',OUT):
        if not base.exists():continue
        for f in sorted(base.rglob('*')):
            if f.is_file() and f.suffix.lower() in ('.csv','.tex','.tsv','.xlsx'):
                m=PAT.search(f.stem)
                if m and int(m.group(1)) not in found:found[int(m.group(1))]=f
    for i in range(1,9):
        if i not in found:
            print(f'SUPPLEMENT_TABLE_S{i}=FAIL; no archived table artifact',flush=True)
            return 2
        shutil.copyfile(found[i],dest/f'Table_S{i}{found[i].suffix.lower()}')
    print('SUPPLEMENT_TABLES_FINAL=PASS; count=8',flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
