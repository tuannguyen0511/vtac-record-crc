"""Paths for publicly distributed stage scripts (local paths.yaml only)."""
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]

def get(key):
    if key=='repo': return ROOT
    path=ROOT/'configs'/'paths.yaml'
    if not path.is_file():
        raise FileNotFoundError('Copy configs/paths.example.yaml to configs/paths.yaml')
    cfg=yaml.safe_load(path.read_text(encoding='utf-8'))
    if key not in cfg: raise KeyError('missing path key '+key)
    value=Path(str(cfg[key]))
    if value.is_absolute():
        raise ValueError('paths.yaml must contain relative paths: '+key)
    return (ROOT/value).resolve()
