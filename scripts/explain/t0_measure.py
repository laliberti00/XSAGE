"""TURNO 0 - misure di fattibilita'. SOLA LETTURA. Non scrive nulla nel repo.
Output: stdout + /scratchpad/t0_measure.json
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd

CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
SCRATCH = Path("/private/tmp/claude-501/-Users-lucaaliberti-Downloads-xsage-clean/5a2b49a6-047b-489b-ac20-87da8428826a/scratchpad")
sys.path.insert(0, str(CLEAN / "scripts" / "mind")); sys.path.insert(0, str(CLEAN))
sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from mind_prep import build_descriptor, membership_from_assign, ALPHA
from eval_kappa import select_K, select_eps, SEED
from pipeline.step02_models.xsage.l2_comprehension import _assign, fit_rough_kmeans
from xsage.recommendation import fit_situation_biases_z

CITY = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
T = {}
def tic(): return time.perf_counter()
def toc(t0, k):
    T[k] = round(time.perf_counter() - t0, 2); print(f"  [t] {k} = {T[k]}s", flush=True); return T[k]

R = {"city": CITY}

# ---------- FASE 1: descrittore ----------
print(f"[{CITY}] FASE 1 build_descriptor ...", flush=True)
t0 = tic(); D0 = build_descriptor(CITY, splits=("train","val","test")); toc(t0, "build_descriptor")
ds=D0["ds"]; nmac=D0["n_macros"]; icm=D0["icm"]; excl=D0["excl"]; sb=D0["sb"]; cmt=D0["cmt"]
vtr,vva,vte = D0["vs"]["train"], D0["vs"]["val"], D0["vs"]["test"]
R.update(n_users=int(ds["n_users"]), n_items=int(ds["n_items"]), n_macros=int(nmac),
         n_train=int(len(ds["df_train"])), n_val=int(len(ds["df_val"])), n_test=int(len(ds["df_test"])),
         dim_v=int(vtr.shape[1]))
print(f"  n_test={R['n_test']} n_items={R['n_items']} n_macros={nmac} dim(v)={R['dim_v']}", flush=True)

# ---------- FASE 1b: matrice punteggi ----------
R["sb_shape"]=list(map(int, sb.shape)); R["sb_dtype"]=str(sb.dtype)
R["sb_is_memmap"]=isinstance(sb, np.memmap)
R["sb_nbytes_MB"]=round(sb.size*sb.dtype.itemsize/1e6,1)
R["excl_shape"]=list(map(int, excl.shape)); R["excl_nnz"]=int(excl.nnz)
print(f"  sb {R['sb_shape']} {R['sb_dtype']} memmap={R['sb_is_memmap']} {R['sb_nbytes_MB']}MB", flush=True)

# ---------- FASE 2: selezione K/eps ----------
print(f"[{CITY}] FASE 2 select_K + select_eps ...", flush=True)
rng = np.random.default_rng(SEED)
t0=tic(); K = select_K(vtr, int(D0["attractors"].sum())+2, rng); toc(t0,"select_K")
t0=tic(); eps = select_eps(vtr, vva, K); toc(t0,"select_eps")
R["K"]=int(K); R["eps"]=float(eps)
print(f"  K={K} eps={eps}", flush=True)

# ---------- FASE 3: fit + assign + b_z ----------
t0=tic(); fit = fit_rough_kmeans(vtr, K=K, eps=eps, seed=SEED, max_iter=80); toc(t0,"fit_rough_kmeans")
z_tr = fit.core_label.astype(np.int64)
t0=tic(); b_z = fit_situation_biases_z(z_tr, cmt, K, nmac, alpha=ALPHA); toc(t0,"fit_situation_biases_z")
t0=tic(); _, kte, compte, isbte = _assign(vte, fit.prototypes, eps); toc(t0,"assign_test")
gam_te = 1.0/np.maximum(compte.sum(1),1).astype(np.float32)
mem_te = membership_from_assign(kte.astype(np.int64), compte, isbte, K)
t0=tic(); nudge_boundary_aware = (mem_te.astype(np.float32) @ b_z); toc(t0,"nudge_mem_at_bz")
nudge_core_only = b_z[kte]

R["bfrac"]=float(isbte.mean())
R["b_z_shape"]=list(map(int,b_z.shape))
R["nudge_shape"]=list(map(int,nudge_boundary_aware.shape))
R["nudge_MB_float32"]=round(nudge_boundary_aware.size*4/1e6,2)
# |T| distribution
Tsize = np.asarray(compte.sum(1)).ravel().astype(int)
R["T_dist"]={str(int(v)):int(c) for v,c in zip(*np.unique(Tsize, return_counts=True))}
# boundary: il nudge boundary-aware differisce davvero da b_z[k]?
bmask = isbte.astype(bool)
if bmask.any():
    d = np.abs(nudge_boundary_aware[bmask]-nudge_core_only[bmask]).max(1)
    R["boundary_maxabs_diff_median"]=float(np.median(d))
    R["boundary_rows_where_identical"]=int((d<1e-6).sum())
    R["boundary_rows"]=int(bmask.sum())
core = ~bmask
R["core_identical_to_bz"]=bool(np.allclose(nudge_boundary_aware[core], nudge_core_only[core], atol=1e-5))
print(f"  bfrac={R['bfrac']:.4f}  |T| dist={R['T_dist']}", flush=True)
print(f"  core nudge == b_z[k]: {R['core_identical_to_bz']}", flush=True)

# ---------- FASE 4: kappa congelato ----------
csv = pd.read_csv(CLEAN/"outputs_results"/f"battery_bfull_{CITY}.csv")
kap = float(csv[(csv.seed==42)&(csv.backbone=="B_blind")&(csv.method=="SIT")]["kstar"].iloc[0])
R["kappa_star"]=kap
print(f"  kappa*={kap}", flush=True)

# ---------- FASE 5: costo del RECORD PER RICHIESTA ----------
dft = ds["df_test"]
ute = dft["u_idx"].values.astype(np.int64); ite = dft["i_idx"].values.astype(np.int64)
rec = dict(row=np.arange(len(dft),dtype=np.int32), u=ute.astype(np.int32), i=ite.astype(np.int32),
           k=kte.astype(np.int8), isb=isbte.astype(np.bool_), gam=gam_te.astype(np.float32),
           Tsize=Tsize.astype(np.int8), true_cat=icm[ite].astype(np.int8),
           mem=mem_te.astype(np.float32), nudge=nudge_boundary_aware.astype(np.float32))
t0=tic()
np.savez_compressed(SCRATCH/f"_rec_{CITY}.npz", **rec)
toc(t0,"save_record_npz")
R["record_npz_MB"]=round(os.path.getsize(SCRATCH/f"_rec_{CITY}.npz")/1e6,2)
t0=tic()
np.savez(SCRATCH/f"_rec_{CITY}_raw.npz", **rec)
toc(t0,"save_record_npz_uncompressed")
R["record_npz_raw_MB"]=round(os.path.getsize(SCRATCH/f"_rec_{CITY}_raw.npz")/1e6,2)
print(f"  record: {R['record_npz_MB']}MB compresso / {R['record_npz_raw_MB']}MB grezzo", flush=True)

# ---------- FASE 6: RANGHI BASE/SIT + ABLAZIONE A COLONNE AZZERATE ----------
def ranks_for(rows, zero_cats=None, keep_only=None):
    """rank (1-based) dell'item held-out in BASE e SIT, con nudge eventualmente ablato."""
    out_b=np.zeros(len(rows),np.int32); out_s=np.zeros(len(rows),np.int32)
    for j,r in enumerate(rows):
        u=int(ute[r]); cc=excl.indices[excl.indptr[u]:excl.indptr[u+1]]
        base=sb[u].astype(np.float64,copy=True)
        nv=nudge_boundary_aware[r].astype(np.float64,copy=True)
        if zero_cats is not None: nv[list(zero_cats)]=0.0
        if keep_only is not None:
            m=np.zeros_like(nv); m[list(keep_only)]=nv[list(keep_only)]; nv=m
        sit=base + kap*float(gam_te[r])*nv[icm]
        base[cc]=-np.inf; sit[cc]=-np.inf
        t=int(ite[r])
        out_b[j]=int((base>base[t]).sum())+1; out_s[j]=int((sit>sit[t]).sum())+1
    return out_b,out_s

rng2=np.random.default_rng(7)
sample=rng2.choice(len(dft),500,replace=False)
t0=tic(); rb,rs = ranks_for(sample); toc(t0,"rank_500_no_ablation")
top3=np.argsort(-b_z, axis=1)[:,:3]
t0=tic()
rb2,rs2 = ranks_for(sample, zero_cats=set(top3[0].tolist()))
toc(t0,"rank_500_with_ablation")
R["rank_500_s"]=T["rank_500_with_ablation"]
R["rank_per_request_ms"]=round(1000*T["rank_500_with_ablation"]/500,2)
R["sample_rank_base_median"]=int(np.median(rb)); R["sample_rank_sit_median"]=int(np.median(rs))
print(f"  ablazione+riordino: {R['rank_per_request_ms']} ms/richiesta", flush=True)

# ---------- FASE 7: sanity sul caso di riferimento k=4 ----------
R["b_z_round2"]=np.round(b_z,2).tolist()
i2m={v:k for k,v in ds["macro_to_idx"].items()}
R["macro_names"]=[i2m[i] for i in range(nmac)]
fav={}
for k in range(K):
    o=np.argsort(-b_z[k])[:3]; fav[str(k)]=[[i2m[int(c)], round(float(b_z[k][c]),3)] for c in o]
R["fav_genres"]=fav
print(json.dumps(fav,ensure_ascii=False), flush=True)

R["timings"]=T
json.dump(R, open(SCRATCH/f"t0_measure_{CITY}.json","w"), ensure_ascii=False, indent=2)
print("\n=== OK ===", flush=True)
print(json.dumps({k:v for k,v in R.items() if k not in ("b_z_round2","macro_names","fav_genres")}, ensure_ascii=False, indent=2))
