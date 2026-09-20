"""TURNO 0 - round 3: PN/PS discrimina Ĉ vero da Ĉ falso? item-level vs category-level. SOLA LETTURA."""
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
CITY="ml1m"; t00=time.perf_counter()
D0=build_descriptor(CITY,splits=("train","val","test"))
ds=D0["ds"];nmac=D0["n_macros"];icm=D0["icm"];excl=D0["excl"];sb=D0["sb"];cmt=D0["cmt"]
vtr,vva,vte=D0["vs"]["train"],D0["vs"]["val"],D0["vs"]["test"]
rng=np.random.default_rng(SEED); K=select_K(vtr,int(D0["attractors"].sum())+2,rng); eps=select_eps(vtr,vva,K)
fit=fit_rough_kmeans(vtr,K=K,eps=eps,seed=SEED,max_iter=80)
b_z=fit_situation_biases_z(fit.core_label.astype(np.int64),cmt,K,nmac,alpha=ALPHA)
_,kte,compte,isbte=_assign(vte,fit.prototypes,eps)
gam=1.0/np.maximum(compte.sum(1),1).astype(np.float32)
mem=membership_from_assign(kte.astype(np.int64),compte,isbte,K); nudge=(mem.astype(np.float32)@b_z)
print(f"[setup] {time.perf_counter()-t00:.1f}s",flush=True)
csv=pd.read_csv(CLEAN/"outputs_results"/f"battery_bfull_{CITY}.csv")
kap=float(csv[(csv.seed==42)&(csv.backbone=="B_blind")&(csv.method=="SIT")]["kstar"].iloc[0])
dft=ds["df_test"];ute=dft["u_idx"].values.astype(np.int64);ite=dft["i_idx"].values.astype(np.int64);tcat=icm[ite]
rg=np.random.default_rng(11); SAMPLE=rg.choice(len(dft),2000,replace=False)
top3=np.argsort(-b_z,axis=1)[:,:3]; bot3=np.argsort(b_z,axis=1)[:,:3]
def ranks(r,nv):
    u=int(ute[r]); cc=excl.indices[excl.indptr[u]:excl.indptr[u+1]]
    s=np.asarray(sb[u],dtype=np.float64)+kap*float(gam[r])*nv[icm]; s[cc]=-np.inf
    t=int(ite[r]); ri=int((s>s[t]).sum())+1
    mt=icm==tcat[r]; best=s[mt].max(); rc=int((s>best).sum())+1
    return ri,rc
res={}
t0=time.perf_counter()
for tag,gets in (("Ctrue",lambda r:set(top3[int(kte[r])].tolist())),
                 ("Cfalse",lambda r:set(bot3[int(kte[r])].tolist())),
                 ("Crandom",lambda r:set(rg.choice(nmac,3,replace=False).tolist()))):
    acc={f"PN_item_top{t}":0 for t in (5,10,20)}; acc.update({f"PN_cat_top{t}":0 for t in (5,10,20)})
    acc.update({f"PS_cat_top{t}":0 for t in (5,10,20)}); acc.update({f"PS_item_top{t}":0 for t in (5,10,20)})
    n_i={t:0 for t in (5,10,20)}; n_c={t:0 for t in (5,10,20)}
    for r in SAMPLE:
        nv=nudge[r].astype(np.float64); C=gets(r)
        ri0,rc0=ranks(r,nv)
        nabl=nv.copy(); nabl[list(C)]=0.0
        nkeep=np.zeros_like(nv)
        for c in C: nkeep[c]=nv[c]
        ri1,rc1=ranks(r,nabl); ri2,rc2=ranks(r,nkeep)
        for t in (5,10,20):
            if ri0<=t: n_i[t]+=1; acc[f"PN_item_top{t}"]+=(ri1>t); acc[f"PS_item_top{t}"]+=(ri2<=t)
            if rc0<=t: n_c[t]+=1; acc[f"PN_cat_top{t}"]+=(rc1>t); acc[f"PS_cat_top{t}"]+=(rc2<=t)
    out={"support_item":n_i,"support_cat":n_c}
    for k,v in acc.items():
        t=int(k[-2:].replace("p","")); base=n_i[t] if "item" in k else n_c[t]
        out[k]=round(v/base,4) if base else None
    res[tag]=out
    print(tag, json.dumps(out), flush=True)
res["time_s"]=round(time.perf_counter()-t0,1); res["n_sample"]=len(SAMPLE); res["kappa"]=kap
json.dump(res,open(SCRATCH/"t0_measure3.json","w"),indent=2)
print(json.dumps(res,indent=2))
