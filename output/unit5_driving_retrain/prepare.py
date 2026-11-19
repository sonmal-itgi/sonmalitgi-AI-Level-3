"""Local-only experiment: retain signer split and add one confirmed webcam training clip.
Raw labels/landmarks and webcam inputs must never be committed to GitHub.
"""
import csv,json,hashlib,os
from pathlib import Path
from zipfile import ZipFile
import numpy as np

root=Path(__file__).resolve().parent
workspace=Path(os.environ.get('SONMAL_WORKSPACE', str(root.parents[1])))
annotations={}
for path in (workspace/'output/unit5_driving_audit').glob('*morpheme_selected.zip'):
    with ZipFile(path) as z:
        for n in z.namelist():
            annotations[Path(n).name.removesuffix('_morpheme.json')]=json.loads(z.read(n))
rows=[]; audit=[]
(root/'clips').mkdir(exist_ok=True)
for source,allowed in [(workspace/'output/unit5_lstm_real01_to_real12/data', {f'REAL{i:02}' for i in range(1,13)}),
                       (workspace/'output/unit5_eval', {'REAL13','REAL17'})]:
    with (source/'manifest.csv').open() as f:
        source_rows=list(csv.DictReader(f))
    for r in source_rows:
        if r['signer'] not in allowed or r['word_id'] not in ('SEN0009','SEN0011'):
            continue
        # Explicit signer allowlist avoids opening any final test landmark array.
        assert (r['split']=='train') == (r['signer'] not in {'REAL13','REAL17'})
        annotation=annotations[r['clip_id']]
        words=[x['attributes'][0]['name'] for x in annotation['data']]
        if r['signer']=='REAL10' and r['word_id']=='SEN0009':
            words=['운전','천천히'] if words==['운전','느리다'] else words
        assert words==(['운전','천천히'] if r['word_id']=='SEN0009' else ['운전','빨리']), (r['clip_id'],words)
        arr=np.load(source/r['path'])
        fps=len(arr)/float(annotation['metaData']['duration'])
        for segment,word in zip(annotation['data'],words):
            start,end=[round(float(segment[k])*fps) for k in ('start','end')]
            assert 0<=start<end<=len(arr)
            rel=f"clips/{r['clip_id']}_{word}.npy"
            np.save(root/rel,arr[start:end])
            rows.append(dict(path=rel,word_id=word,label=word,split=r['split'],signer=r['signer']))
            previous=workspace/f"output/unit5_word_sequence_real01_to_real10/clips/{r['clip_id']}_{word}.npy"
            match=bool(np.array_equal(np.load(previous),arr[start:end],equal_nan=True)) if previous.exists() else None
            audit.append(dict(clip=r['clip_id'],word=word,split=r['split'],start=start,end=end,
                              fps=fps,old_clip_exact_match=match))
def manifest(name,items):
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','word_id','label','split','signer']);w.writeheader();w.writerows(items)
manifest('baseline.csv',rows)
# Only the user's confirmed driving-only log is admitted as training.
source=Path.home()/'Downloads/손말잇기-웹캠평가-SEN0009-2026-11-18T09-11-14.042Z.json'
frames=json.loads(source.read_text())['frames']
visible=[i for i,f in enumerate(frames) if f['left_hand'] or f['right_hand']]
frames=frames[min(visible):max(visible)+1]
a=np.full((len(frames),42,3),np.nan,np.float32);a[...,2]=0
for i,f in enumerate(frames):
    for base,key in ((0,'left_hand'),(21,'right_hand')):
        for j,p in enumerate(f[key]):
            a[i,base+j]=[p['x']*f['image_width'],p['y']*f['image_height'],1.]
rel='clips/webcam_driving_training.npy';np.save(root/rel,a)
rows.append(dict(path=rel,word_id='운전',label='운전',split='train',signer='WEBCAM_TRAIN'))
manifest('candidate.csv',rows)
(root/'audit.json').write_text(json.dumps(dict(segments=audit,webcam_train_frames=len(a),
    training_webcam_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    final_test_opened=False),ensure_ascii=False,indent=2))
print('baseline train',sum(r['split']=='train' for r in rows)-1,'candidate train',sum(r['split']=='train' for r in rows),
      'val',sum(r['split']=='val' for r in rows),'previous matches',sum(x['old_clip_exact_match'] is True for x in audit),
      'webcam train frames',len(a))
