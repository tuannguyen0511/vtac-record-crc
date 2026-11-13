"""Export the five exact archived figures identified in the manuscript build.

No figure data are invented. The input may be a scientific figure archived under
paper/source_figures or a previously-rendered output reached via paths.yaml.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'configs'/'paths.yaml').read_text(encoding='utf-8'))
OUT=(ROOT/CFG['outputs']).resolve()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'paper'/'figure_manifest.json').read_text(encoding='utf-8'))
    if len(manifest.get('figures',[]))!=5:
        print('FIGURES_FINAL=FAIL; exactly five manuscript figures not verified',flush=True)
        return 2
    for i,item in enumerate(manifest['figures'],1):
        if item['item']!=f'Fig{i}':return 2
        base=ROOT if item['location']=='repo' else OUT
        f=(base/item['path']).resolve()
        if not f.is_file() or sha(f)!=item['sha256']:
            print(f'FIGURE_{i}=FAIL; source SHA or path mismatch',flush=True)
            return 2
        target=dest/f'Fig{i}{f.suffix.lower()}'
        shutil.copyfile(f,target)
        if sha(target)!=item['sha256']:return 2
        print(f'FIGURE_{i}=PASS; archived source verified',flush=True)
    print('FIGURES_FINAL=PASS; count=5; export-only',flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
