"""TURNO 0 - round 4: una misura CONTINUA (quota di spostamento attribuibile a Ĉ) discrimina? SOLA LETTURA."""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
CLEAN=Path("/Users/lucaaliberti/Downloads/xsage-clean")
SCRATCH=Path("/private/tmp/claude-501/-Users-lucaaliberti-Downloads-xsage-clean/5a2b49a6-047b-489b-ac20-87da8428826a/scratchpad")
sys.path.insert(0,str(CLEAN/"scripts"/"mind")); sys.path.insert(0,str(CLEAN))
sys.path.insert(0,"/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z
D0=build_descriptor("ml1m",splits=("train","val","test"))
ds=D0["ds"];nmac=D0["n_macros"];icm=D0["icm"];excl=D0["excl"];sb=D0["sb"];cmt=D0["cmt"]
vtr,vva,vte=D0["vs"]["train"],D0["vs"]["val"],D0["vs"]["test"]
rng=np.random.default_rng(SEED);K=select_K(vtr,int(D0["attractors"].sum())+2,rng);eps=select_eps(vtr,vva,K)
fit=fit_rough_kmeans(vtr,K=K,eps=eps,seed=SEED,max_iter=80)
b_z=fit_situation_biases_z(fit.core_label.astype(np.int64),cmt,K,nmac,alpha=ALPHA)
_,kte,compte,isbte=_assign(vte,fit.prototypes,eps)
gam=1.0/np.maximum(compte.sum(1),1).astype(np.float32)
mem=membership_from_assign(kte.astype(np.int64),compte,isbte,K);nudge=(mem.astype(np.float32)@b_z)
csv=pd.read_csv(CLEAN/"outputs_results"/"battery_bfull_ml1m.csv")
kap=float(csv[(csv.seed==42)&(csv.backbone=="B_blind")&(csv.method=="SIT")]["kstar"].iloc[0])
dft=ds["df_test"];ute=dft["u_idx"].values.astype(np.int64);ite=dft["i_idx"].values.astype(np.int64);tcat=icm[ite]
rg=np.random.default_rng(11);SAMPLE=rg.choice(len(dft),2000,replace=False)
top3=np.argsort(-b_z,axis=1)[:,:3];bot3=np.argsort(b_z,axis=1)[:,:3]
def rk(r,nv):
    u=int(ute[r]);cc=excl.indices[excl.indptr[u]:excl.indptr[u+1]]
    s=np.asarray(sb[u],dtype=np.float64)+kap*float(gam[r])*nv[icm];s[cc]=-np.inf
    t=int(ite[r]);mt=icm==tcat[r]
    return int((s>s[t]).sum())+1, int((s>s[mt].max()).sum())+1
out={}
t0=time.perf_counter()
for tag,gets in (("Ctrue",lambda r:top3[int(kte[r])].tolist()),("Cfalse",lambda r:bot3[int(kte[r])].tolist()),
                 ("Crandom",lambda r:rg.choice(nmac,3,replace=False).tolist())):
    Ri,Rc=[],[]
    for r in SAMPLE:
        nv=nudge[r].astype(np.float64);C=gets(r)
        i_f,c_f=rk(r,nv); i_0,c_0=rk(r,np.zeros_like(nv))
        na=nv.copy();na[C]=0.0; i_a,c_a=rk(r,na)
        ti=i_0-i_f; tc=c_0-c_f              # spostamento TOTALE dovuto al nudge (>0 = il nudge ha aiutato)
        si=i_a-i_f; sc=c_a-c_f              # spostamento perso azzerando C
        if abs(ti)>=1: Ri.append(si/ti)
        if abs(tc)>=1: Rc.append(sc/tc)
    Ri=np.array(Ri);Rc=np.array(Rc)
    out[tag]=dict(n_item=len(Ri),n_cat=len(Rc),
        item_median=round(float(np.median(Ri)),4),item_mean=round(float(Ri.mean()),4),
        item_q25=round(float(np.percentile(Ri,25)),4),item_q75=round(float(np.percentile(Ri,75)),4),
        cat_median=round(float(np.median(Rc)),4),cat_mean=round(float(Rc.mean()),4),
        cat_q25=round(float(np.percentile(Rc,25)),4),cat_q75=round(float(np.percentile(Rc,75)),4))
    print(tag,json.dumps(out[tag]),flush=True)
out["time_s"]=round(time.perf_counter()-t0,1)
json.dump(out,open(SCRATCH/"t0_measure4.json","w"),indent=2)
