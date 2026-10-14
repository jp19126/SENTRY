"""Validate permitted prepared file bytes against the pinned fold manifest."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'results/cnn_study_v1/prepared'
ALLOWED={'screen_fit','fit','inner_calibration','inner_selection','full_train',
         'search_temporary_threshold','search_candidate_scoring','diagnostics'}

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def validate_prepared(rep,splits):
    if rep not in {'token','byte'} or not set(splits)<=ALLOWED:
        raise ValueError('Unapproved prepared representation/fold')
    summary=json.loads((PREPARED/rep/'summary.json').read_text(encoding='utf-8'))
    checked={}
    for split in splits:
        folder=PREPARED/rep/split
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        # summary has extra preparation provenance fields, so compare fold fields.
        if any(summary[split].get(k)!=v for k,v in manifest.items()):
            raise ValueError('Prepared manifest differs from pinned summary: '+rep+'/'+split)
        hashes=manifest['artifact_sha256']
        for name,expected in hashes.items():
            if Path(name).name!=name or digest(folder/name)!=expected:
                raise ValueError('Prepared artifact differs: '+rep+'/'+split+'/'+name)
        checked[split]=dict(hashes)
    return checked
