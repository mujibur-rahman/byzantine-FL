"""
Divergence detection vs Byzantine ratio (label-flip, NYC).
For each Byzantine % and each measure (kappa, KL, Renyi2), report detection
AUC and recall@FPR=0.20 (Byzantine vs benign, threshold-free + fixed-FPR).
"""
import os, sys
os.environ.setdefault('DATASET', sys.argv[sys.argv.index('--dataset')+1] if '--dataset' in sys.argv else 'nyc-taxi')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, roc_auc_score, roc_curve
from dataset_config import get_dataset, DATASET_NAME
from fl_model import FLNeuralNet
from fl_base import partition_noniid, make_validation_set, attack_label_flip

SEED=42; N=100; ROUNDS=30; LR=0.01; VAL=2000; EPS=1e-6
BYZ_GRID=[0.10,0.20,0.30,0.40,0.50]

def kap(a,b):
    if len(np.unique(a))<2 or len(np.unique(b))<2: return 1.0 if np.array_equal(a,b) else 0.0
    return float(cohen_kappa_score(a,b))
def kl(pg,pc):
    p=np.clip(pc,EPS,1-EPS); q=np.clip(pg,EPS,1-EPS)
    return float((p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))).mean())
def renyi(pg,pc,a=2.0):
    p=np.clip(pc,EPS,1-EPS); q=np.clip(pg,EPS,1-EPS)
    return float((1/(a-1)*np.log(np.clip(p**a*q**(1-a)+(1-p)**a*(1-q)**(1-a),EPS,None))).mean())

def recall_at_fpr(y,score,target=0.20):
    fpr,tpr,_=roc_curve(y,score)
    ok=fpr<=target
    return float(tpr[ok].max()) if ok.any() else 0.0

def run(byz_ratio, clients, Xv, byz, ben, nf, per_round=20):
    gp=FLNeuralNet(nf).get_params()
    acc={m:{i:[] for i in range(N)} for m in ['kappa','kl','renyi']}
    nb=max(1,int(per_round*byz_ratio)); ng=per_round-nb
    for rnd in range(ROUNDS):
        rng=np.random.RandomState(SEED+rnd)
        s=rng.choice(ben,min(ng,len(ben)),replace=False).tolist()+rng.choice(list(byz),min(nb,len(byz)),replace=False).tolist()
        gm=FLNeuralNet(nf); gm.set_params(gp.copy()); gh=gm.predict(Xv); gpr=gm.predict_proba(Xv)
        ups=[]
        for cid in s:
            Xc,yc=clients[cid]
            if len(Xc)==0: continue
            if cid in byz: yc=attack_label_flip(yc,0.30)
            m=FLNeuralNet(nf); m.set_params(gp.copy()); m.train(Xc,yc,lr=LR,epochs=5,batch_size=32)
            ch=m.predict(Xv); cp=m.predict_proba(Xv)
            acc['kappa'][cid].append(-kap(gh,ch)); acc['kl'][cid].append(kl(gpr,cp)); acc['renyi'][cid].append(renyi(gpr,cp))
            ups.append(m.get_params()-gp)
        if ups: gp=gp+LR*np.mean(ups,axis=0)
    ids=[i for i in range(N) if acc['kappa'][i]]
    y=np.array([1 if i in byz else 0 for i in ids])
    out={}
    for m in ['kappa','kl','renyi']:
        sc=np.array([np.mean(acc[m][i]) for i in ids])
        out[m]=(roc_auc_score(y,sc) if len(np.unique(y))>1 else np.nan, recall_at_fpr(y,sc))
    return out

if __name__=='__main__':
    X,y=get_dataset(seed=SEED); sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xv,yv=make_validation_set(Xs,y,VAL,seed=SEED)
    clients=partition_noniid(Xs,y,N,alpha=0.5,seed=SEED); nf=Xs.shape[1]
    print(f"\n##### {DATASET_NAME} — label-flip: detection vs Byzantine % #####")
    print(f"{'byz%':>5s} | {'kappa AUC':>9s} {'rec@.2':>6s} | {'KL AUC':>7s} {'rec@.2':>6s} | {'Renyi AUC':>9s} {'rec@.2':>6s}")
    for br in BYZ_GRID:
        nby=int(N*br); byz=set(range(N-nby,N)); ben=list(range(N-nby))
        r=run(br,clients,Xv,byz,ben,nf)
        print(f"{int(br*100):4d}% | {r['kappa'][0]:9.3f} {r['kappa'][1]:6.2f} | "
              f"{r['kl'][0]:7.3f} {r['kl'][1]:6.2f} | {r['renyi'][0]:9.3f} {r['renyi'][1]:6.2f}")
