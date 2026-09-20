"""TURNO 0 - round 2: supporto di PN/PS, strati su TUTTO il test, costo. SOLA LETTURA."""
import sys, json, time, os
from pathlib import Path
import numpy as np, pandas as pd
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
SCRATCH = Path("/private/tmp/claude-501/-Users-lucaaliberti-Downloads-xsage-clean/5a2b49a6-047b-489b-ac20-87da8428826a/scratchpad")
sys.path.insert(0, str(CLEAN/"scripts"/"mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z
CITY = sys.argv[1] if len(sys.argv)>1 else "ml1m"
t00=time.perf_counter()
D0 = build_descriptor(CITY, splits=("train","val","test"))
ds=D0["ds"]; nmac=D0["n_macros"]; icm=D0["icm"]; excl=D0["excl"]; sb=D0["sb"]; cmt=D0["cmt"]
vtr,vva,vte=D0["vs"]["train"],D0["vs"]["val"],D0["vs"]["test"]
rng=np.random.default_rng(SEED)
K=select_K(vtr,int(D0["attractors"].sum())+2,rng); eps=select_eps(vtr,vva,K)
fit=fit_rough_kmeans(vtr,K=K,eps=eps,seed=SEED,max_iter=80)
b_z=fit_situation_biases_z(fit.core_label.astype(np.int64),cmt,K,nmac,alpha=ALPHA)
_,kte,compte,isbte=_assign(vte,fit.prototypes,eps)
gam=1.0/np.maximum(compte.sum(1),1).astype(np.float32)
mem=membership_from_assign(kte.astype(np.int64),compte,isbte,K)
nudge=(mem.astype(np.float32)@b_z)
setup=time.perf_counter()-t00
print(f"[setup] {setup:.1f}s", flush=True)
csv=pd.read_csv(CLEAN/"outputs_results"/f"battery_bfull_{CITY}.csv")
kap=float(csv[(csv.seed==42)&(csv.backbone=="B_blind")&(csv.method=="SIT")]["kstar"].iloc[0])
dft=ds["df_test"]; ute=dft["u_idx"].values.astype(np.int64); ite=dft["i_idx"].values.astype(np.int64)
n=len(dft); tcat=icm[ite]
BATCH=1024
rank_item_b=np.zeros(n,np.int32); rank_item_s=np.zeros(n,np.int32)
rank_cat_b=np.zeros(n,np.int32); rank_cat_s=np.zeros(n,np.int32)
t0=time.perf_counter()
for bs in range(0,n,BATCH):
    be=min(n,bs+BATCH); ub=ute[bs:be]
    B=np.asarray(sb[ub],dtype=np.float32).copy()
    S=B+ (kap*gam[bs:be][:,None]*nudge[bs:be][:,icm]).astype(np.float32)
    for j in range(be-bs):
        u=int(ub[j]); cc=excl.indices[excl.indptr[u]:excl.indptr[u+1]]
        if len(cc): B[j,cc]=-np.inf; S[j,cc]=-np.inf
    it=ite[bs:be]; ar=np.arange(be-bs)
    rank_item_b[bs:be]=(B>B[ar,it][:,None]).sum(1)+1
    rank_item_s[bs:be]=(S>S[ar,it][:,None]).sum(1)+1
    mt=(icm[None,:]==tcat[bs:be,None])
    bb=np.where(mt,B,-np.inf).max(1); ss=np.where(mt,S,-np.inf).max(1)
    rank_cat_b[bs:be]=(B>bb[:,None]).sum(1)+1
    rank_cat_s[bs:be]=(S>ss[:,None]).sum(1)+1
full=time.perf_counter()-t0
print(f"[ranghi su TUTTO il test n={n}] {full:.1f}s", flush=True)
R={"city":CITY,"n_test":int(n),"setup_s":round(setup,1),"rank_alltest_s":round(full,1),
   "kappa":kap,"K":int(K),"eps":float(eps),"bfrac":float(isbte.mean())}
def q(a): return [int(np.percentile(a,p)) for p in (25,50,75,90,99)]
R["rank_item_base_pct"]=q(rank_item_b); R["rank_item_sit_pct"]=q(rank_item_s)
R["rank_cat_base_pct"]=q(rank_cat_b); R["rank_cat_sit_pct"]=q(rank_cat_s)
for TK in (5,10,20):
    R[f"item_in_top{TK}_BASE"]=int((rank_item_b<=TK).sum()); R[f"item_in_top{TK}_SIT"]=int((rank_item_s<=TK).sum())
    R[f"cat_in_top{TK}_SIT"]=int((rank_cat_s<=TK).sum())
d=rank_item_s.astype(int)-rank_item_b.astype(int)
dc=rank_cat_s.astype(int)-rank_cat_b.astype(int)
core=~isbte.astype(bool); bnd=isbte.astype(bool)
R["strati_item"]={"vittorie":int((d<0).sum()),"neutri":int((d==0).sum()),"danni":int((d>0).sum()),
                  "core":int(core.sum()),"boundary":int(bnd.sum())}
R["strati_cat"]={"vittorie":int((dc<0).sum()),"neutri":int((dc==0).sum()),"danni":int((dc>0).sum())}
R["strati_incrocio_item"]={f"{a}_{b}":int(((d<0 if a=='vitt' else (d==0 if a=='neut' else d>0)) & (core if b=='core' else bnd)).sum())
    for a in ("vitt","neut","dann") for b in ("core","boundary")}
R["strati_incrocio_cat"]={f"{a}_{b}":int(((dc<0 if a=='vitt' else (dc==0 if a=='neut' else dc>0)) & (core if b=='core' else bnd)).sum())
    for a in ("vitt","neut","dann") for b in ("core","boundary")}
# --- supporto PN: su quante richieste l'ablazione delle top-3 categorie della situazione MUOVE qualcosa
top3=np.argsort(-b_z,axis=1)[:,:3]
sel=np.where(rank_item_s<=20)[0]
R["n_item_top20_SIT"]=int(len(sel))
sub=sel if len(sel)<=3000 else np.random.default_rng(7).choice(sel,3000,replace=False)
t0=time.perf_counter(); out_pn=0; out_ps=0; tot=0
for r in sub:
    u=int(ute[r]); cc=excl.indices[excl.indptr[u]:excl.indptr[u+1]]
    base=np.asarray(sb[u],dtype=np.float64).copy()
    nv=nudge[r].astype(np.float64)
    C=set(top3[int(kte[r])].tolist())
    nabl=nv.copy(); nabl[list(C)]=0.0
    nkeep=np.zeros_like(nv); 
    for c in C: nkeep[c]=nv[c]
    s_full=base+kap*float(gam[r])*nv[icm]; s_abl=base+kap*float(gam[r])*nabl[icm]; s_keep=base+kap*float(gam[r])*nkeep[icm]
    s_full[cc]=-np.inf; s_abl[cc]=-np.inf; s_keep[cc]=-np.inf
    t=int(ite[r])
    r_abl=int((s_abl>s_abl[t]).sum())+1; r_keep=int((s_keep>s_keep[t]).sum())+1
    tot+=1; out_pn+= (r_abl>20); out_ps+= (r_keep<=20)
pn_t=time.perf_counter()-t0
R["PN_support_n"]=tot; R["PN_rate_top3sit"]=round(out_pn/max(tot,1),4); R["PS_rate_top3sit"]=round(out_ps/max(tot,1),4)
R["PN_PS_time_s"]=round(pn_t,2); R["PN_PS_ms_per_req"]=round(1000*pn_t/max(tot,1),2)
print(json.dumps(R,ensure_ascii=False,indent=2))
json.dump(R,open(SCRATCH/f"t0_measure2_{CITY}.json","w"),ensure_ascii=False,indent=2)
