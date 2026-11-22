"""명시적으로 검토한 정상/오류 라벨만 학습한다. 합성 오류·자동 라벨 생성 없음."""
import argparse,csv,json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from features import features,SCHEMA

CLASSES = ['normal', 'under_curled']
def load_manifest(path):
    rows=list(csv.DictReader(path.open()))
    if not rows or any(r['split'] not in ('train','val') or r['label'] not in CLASSES or r.get('reviewed') != 'yes' for r in rows):
        raise ValueError('검토 완료한 train/val 정상·오류 라벨만 허용합니다')
    groups={split:{r['signer'] for r in rows if r['split']==split} for split in ('train','val')}
    if not all(groups.values()) or groups['train'] & groups['val']:
        raise ValueError('사이너가 겹치지 않는 별도 train/val이 필요합니다')
    clips={split:{r['path'] for r in rows if r['split']==split} for split in ('train','val')}
    if clips['train'] & clips['val']:
        raise ValueError('동일 원본 파일을 train/val에 나눌 수 없습니다')
    data={s:([],[],[]) for s in ('train','val')}
    for r in rows:
        document=json.loads((path.parent/r['path']).read_text())
        start,end=float(r['start_ms']),float(r['end_ms'])
        samples=[]
        for f in document['frames']:
            if start <= f['timestamp_ms'] <= end:
                try: samples.append(features(f[r['hand']],f['image_width'],f['image_height']))
                except (ValueError,KeyError): pass
        if not samples: raise ValueError(f"사용 가능한 프레임 없음: {r['path']}")
        x,y,w=data[r['split']]
        x.extend(samples);y.extend([CLASSES.index(r['label'])]*len(samples));w.extend([1/len(samples)]*len(samples))
    for split,(x,y,w) in data.items():
        if set(y) != {0,1}: raise ValueError(f'{split}에 두 클래스가 모두 필요합니다')
    return data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    data=load_manifest(args.manifest)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(2)
    x,y,w=data['train'];x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y);w=torch.tensor(w);w=w/w.sum()
    model=nn.Sequential(nn.Linear(15,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2))
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    for _ in range(150):
        model.train();opt.zero_grad();loss=(nn.functional.cross_entropy(model(x),y,reduction='none')*w).sum();loss.backward();opt.step()
    model.eval()
    vx,vy,vw=data['val'];vx=torch.tensor(vx,dtype=torch.float32);vy=torch.tensor(vy);vw=torch.tensor(vw)
    with torch.no_grad():
        probs=model(vx).softmax(-1);pred=probs.argmax(-1)
    normal=vy==0;false_correction=(pred==1)&(probs[:,1]>=.95)&normal
    metrics={'val_frame_accuracy':float((pred==vy).float().mean()),'val_clip_balanced_accuracy':float(((pred==vy)*vw).sum()/vw.sum()),'normal_false_correction_rate_at_095':float((false_correction*vw).sum()/(normal*vw).sum()),'note':'학습에 없는 사이너 검증. 최종 테스트 미사용.'}
    layers=[{'weight':m.weight.detach().tolist(),'bias':m.bias.detach().tolist(),'activation':'linear' if i==4 else 'relu'} for i,m in enumerate(model) if isinstance(m,nn.Linear)]
    export={'schema':SCHEMA,'enabled':False,'reviewed':False,'scope':'driving-fist','classes':CLASSES,'threshold':.95,'layers':layers,'metrics':metrics}
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),args.out/'candidate.pt');(args.out/'candidate.json').write_text(json.dumps(export,ensure_ascii=False,indent=2))
    print(json.dumps(metrics,ensure_ascii=False));print('후보를 저장했습니다. 실연 검증·기준 확인 전에는 자동 활성화하지 않습니다.')
if __name__=='__main__':main()
