"""Compare only validation signers and known webcam diagnostics; never opens final test."""
import csv,json,sys,os
from pathlib import Path
from collections import Counter,deque
import numpy as np
import torch
root=Path(__file__).resolve().parent
workspace=Path(os.environ.get('SONMAL_WORKSPACE', str(root.parents[1])))
sys.path[:0]=[str(workspace/'word-model'),str(workspace/'word-model/deployment')]
from web_adapter import HandsWordAdapter
from hands_dataset import make_features
from ai_server.recognition import RecognitionState
# Reuse the local runtime dependency environment for Pillow during this experiment.
sys.path.append(str(workspace/'.venv/lib/python3.14/site-packages'))

sources={'production':workspace/'word-model/unit5_words.pt','baseline':root/'baseline/best.pt','candidate':root/'candidate/best.pt'}
with (root/'candidate.csv').open() as f: val=[r for r in csv.DictReader(f) if r['split']=='val']
webcam_train=json.loads((Path.home()/'Downloads/손말잇기-웹캠평가-SEN0009-2026-11-18T09-11-14.042Z.json').read_text())['frames']
negative=json.loads((Path.home()/'Downloads/손말잇기-웹캠평가-SEN0009-2026-11-19T03-42-12.086Z.json').read_text())['frames']
mixed=json.loads((Path.home()/'Downloads/손말잇기-웹캠평가-SEN0009-2026-11-19T03-44-04.800Z.json').read_text())['frames']
report={}
for name,path in sources.items():
    model=HandsWordAdapter(str(path))
    total=0;correct=0;confusion=Counter();windows=Counter()
    with torch.inference_mode():
        for r in val:
            a=np.load(root/r['path'])
            p=model.model(torch.from_numpy(make_features(a,model.cfg))[None]).softmax(-1)[0]
            predicted=model.supported_ids[int(p.argmax())]
            total+=1;correct+=int(predicted==r['word_id']);confusion[f"{r['word_id']}->{predicted}"]+=1
            # Approximate 8fps reception from ~30fps source. Only labeled word interior.
            a=a[::4]
            for end in range(8,len(a)+1,2):
                p=model.model(torch.from_numpy(make_features(a[max(0,end-12):end],model.cfg))[None]).softmax(-1)[0]
                predicted=model.supported_ids[int(p.argmax())]
                windows[f"{r['word_id']}->{predicted}"]+=1
    def replay(frames):
        buffer=deque(maxlen=12);state=RecognitionState(model.supported_ids);count=0
        raw=Counter();stable=Counter()
        for f in frames:
            if not f['left_hand'] and not f['right_hand']:
                buffer.clear();state.reset();count=0;continue
            buffer.append(f);count+=1
            if len(buffer)>=8 and (count-8)%2==0:
                p=model.predict_landmarks(list(buffer));raw[p.predicted_id]+=1
                decision=state.update(p)
                if decision['predicted_id']:stable[decision['predicted_id']]+=1
        return dict(raw=dict(raw),stable=dict(stable))
    groups=[];group=[]
    for f in mixed:
        if f['left_hand'] or f['right_hand']:group.append(f)
        elif group:groups.append(group);group=[]
    if group:groups.append(group)
    assert len(groups)==2
    report[name]=dict(validation_correct=correct,validation_total=total,validation_confusion=dict(confusion),
        validation_approx8fps_windows=dict(windows),webcam_training_replay=replay(webcam_train),
        webcam_no_driving_diagnostic=replay(negative),
        webcam_diagnostic_slow=replay(groups[0]),webcam_diagnostic_fast=replay(groups[1]))
    print(name,json.dumps(report[name],ensure_ascii=False))
(root/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
