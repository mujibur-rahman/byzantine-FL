"""
Divergence-measure comparison for semantic Byzantine detection (NYC Taxi).

Compares three ways to score how far a client's model diverges from the
global model on D_val, as a Byzantine-detection signal:

  - Cohen's kappa   : agreement on HARD predictions (this paper's measure).
                      Lower kappa => more suspicious.
  - KL divergence   : mean per-sample D_KL( Bern(p_client) || Bern(p_global) )
                      over D_val (SOFT predictive distributions).
  - Renyi (a=2)     : mean per-sample D_2( Bern(p_client) || Bern(p_global) ).
                      Generalises KL (a->1 == KL); a=2 emphasises modes.
                      Higher KL/Renyi => more suspicious.

All three use the FULL client model (global + update) vs the global model on
the SAME server-held D_val. Detection quality is reported as AUC (Byzantine
vs benign, threshold-free) per attack type, plus mean byz/benign separation.

Run:  python3 divergence_compare.py --dataset nyc-taxi
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from dataset_config import get_dataset, DATASET_NAME
from fl_model import FLNeuralNet
from fl_base import (partition_noniid, make_validation_set,
                     attack_label_flip, attack_gps_spoof,
                     attack_gradient_poison, is_malicious_round)

SEED=42; N_CLIENTS=100; N_ROUNDS=30; LR=0.01; VAL=2000; EPS=1e-6
ATTACKS=['lfp','spf','ooa','grad-p']

def kappa_sig(pg_hard, pc_hard):
    if len(np.unique(pg_hard))<2 or len(np.unique(pc_hard))<2:
        k = 1.0 if np.array_equal(pg_hard, pc_hard) else 0.0
    else:
        k = float(cohen_kappa_score(pg_hard, pc_hard))
    return -k                                   # higher => more suspicious

def kl_sig(pg, pc):
    p=np.clip(pc,EPS,1-EPS); q=np.clip(pg,EPS,1-EPS)   # D_KL(client||global)
    kl = p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    return float(kl.mean())

def renyi_sig(pg, pc, a=2.0):
    p=np.clip(pc,EPS,1-EPS); q=np.clip(pg,EPS,1-EPS)   # D_a(client||global)
    inner = p**a * q**(1-a) + (1-p)**a * (1-q)**(1-a)
    return float((1.0/(a-1.0) * np.log(np.clip(inner,EPS,None))).mean())

def run(attack, clients, Xv, byz, ben, nf):
    gp=FLNeuralNet(nf).get_params()
    acc={'kappa':{i:[] for i in range(N_CLIENTS)},
         'kl':{i:[] for i in range(N_CLIENTS)},
         'renyi':{i:[] for i in range(N_CLIENTS)}}
    for rnd in range(N_ROUNDS):
        rng=np.random.RandomState(SEED+rnd)
        s=rng.choice(ben,16,replace=False).tolist()+rng.choice(list(byz),4,replace=False).tolist()
        gm=FLNeuralNet(nf); gm.set_params(gp.copy())
        pg_hard=gm.predict(Xv); pg_prob=gm.predict_proba(Xv)
        ups=[]
        for cid in s:
            Xc,yc=clients[cid]
            if len(Xc)==0: continue
            isb=cid in byz
            if isb and attack=='lfp': yc=attack_label_flip(yc,0.30)
            if isb and attack=='spf': Xc=attack_gps_spoof(Xc,0.30)
            if isb and attack=='ooa' and not is_malicious_round(rnd): isb=False
            m=FLNeuralNet(nf); m.set_params(gp.copy())
            m.train(Xc,yc,lr=LR,epochs=5,batch_size=32)
            upd=m.get_params()-gp
            if (cid in byz) and attack=='grad-p':
                dw,db=attack_gradient_poison(upd[:-1],upd[-1],scale=3.0)
                upd=np.append(dw,db); m.set_params(gp+upd)
            pc_hard=m.predict(Xv); pc_prob=m.predict_proba(Xv)
            acc['kappa'][cid].append(kappa_sig(pg_hard,pc_hard))
            acc['kl'][cid].append(kl_sig(pg_prob,pc_prob))
            acc['renyi'][cid].append(renyi_sig(pg_prob,pc_prob))
            ups.append(upd)
        if ups: gp=gp+LR*np.mean(ups,axis=0)
    ids=[i for i in range(N_CLIENTS) if acc['kappa'][i]]
    y=np.array([1 if i in byz else 0 for i in ids])
    out={}
    for name in ['kappa','kl','renyi']:
        sc=np.array([np.mean(acc[name][i]) for i in ids])
        auc=roc_auc_score(y,sc) if len(np.unique(y))>1 else float('nan')
        out[name]=(auc, sc[y==1].mean(), sc[y==0].mean())
    return out

if __name__=='__main__':
    X,y=get_dataset(seed=SEED); sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xv,yv=make_validation_set(Xs,y,VAL,seed=SEED)
    clients=partition_noniid(Xs,y,N_CLIENTS,alpha=0.5,seed=SEED)
    nbe=N_CLIENTS-int(N_CLIENTS*0.2)
    byz=set(range(nbe,N_CLIENTS)); ben=list(range(nbe)); nf=Xs.shape[1]
    print(f"\n##### Divergence comparison — {DATASET_NAME} (20% Byzantine) #####")
    print(f"{'attack':7s} | {'measure':7s} {'AUC':>6s}  {'byz(mean)':>10s} {'ben(mean)':>10s}")
    rows={}
    for a in ATTACKS:
        r=run(a, clients, Xv, byz, ben, nf); rows[a]=r
        for i,name in enumerate(['kappa','kl','renyi']):
            auc,bm,gm_=r[name]
            tag=a if i==0 else ''
            print(f"{tag:7s} | {name:7s} {auc:6.3f}  {bm:10.3f} {gm_:10.3f}")
        print(f"{'':7s} |")
    print("Winner per attack (highest AUC):")
    for a in ATTACKS:
        best=max(rows[a].items(), key=lambda kv: (kv[1][0] if kv[1][0]==kv[1][0] else -1))
        print(f"  {a:7s}: {best[0]} (AUC={best[1][0]:.3f})")
