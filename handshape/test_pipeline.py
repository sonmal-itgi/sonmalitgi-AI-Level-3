import csv,json,math
from pathlib import Path
import numpy as np
import pytest
from features import features
from train import load_manifest

def test_angles_translation_and_mirror():
    points=[{'x':.3+i*.01,'y':.3+math.sin(i)*.03} for i in range(21)]
    assert np.allclose(features(points,640,480),features([{'x':1-p['x'],'y':p['y']} for p in points],640,480))
    with pytest.raises(ValueError): features([],640,480)

def test_reject_unreviewed_and_signer_leakage(tmp_path):
    path=tmp_path/'manifest.csv'
    fields=['path','signer','split','label','hand','start_ms','end_ms','reviewed']
    for reviewed in ['no','yes']:
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for split in ['train','val']:
                w.writerow(dict(path='same.json',signer='A',split=split,label='normal',hand='left_hand',start_ms=0,end_ms=1,reviewed=reviewed))
        with pytest.raises(ValueError): load_manifest(path)
