"""[venv-cornac] B_blind per MIND via BPR (backbone context-blind, sostituisce l'FM importato).
Allena BPR sulle coppie train MIND, costruisce la matrice [n_users x n_items] negli INDICI NOSTRI,
salva FM.scores.npy (B_blind) + Bfull.scores.npy (stub = B_blind[test_users], non context-aware).
"""
import sys
import numpy as np

import sys
CITY = sys.argv[1] if len(sys.argv)>1 else "mind"
IN = f"/Users/lucaaliberti/Downloads/xsage-clean/data/processed/{CITY}/_cornac_in.npz"
BB = f"/Users/lucaaliberti/Downloads/xsage-clean/data/{CITY}/backbone"

d = np.load(IN)
tu, ti = d["train_u"].astype(np.int64), d["train_i"].astype(np.int64)
nU, nI = int(d["n_users"]), int(d["n_items"])
test_users = d["test_users"].astype(np.int64)
print(f"train pairs={len(tu)} n_users={nU} n_items={nI}", flush=True)

import cornac
from cornac.data import Dataset
from cornac.models import BPR
triples = list(zip(tu.astype(str).tolist(), ti.astype(str).tolist(),
                   np.ones(len(tu), dtype=np.float32).tolist()))
ds = Dataset.from_uir(triples)
print(f"cornac Dataset {ds.num_users}x{ds.num_items}", flush=True)
m = BPR(k=64, max_iter=150, learning_rate=0.01, lambda_reg=0.001, seed=42, verbose=False)
m.fit(ds)
print("BPR fit done", flush=True)

# mapping indici: cornac internal -> nostro i_idx
citem_to_our = np.full(ds.num_items, -1, np.int64)
for s, c in ds.iid_map.items():
    citem_to_our[int(c)] = int(s)
cidx_known = np.where(citem_to_our >= 0)[0]
our_known = citem_to_our[cidx_known]
uid_map = ds.uid_map

scores = np.zeros((nU, nI), dtype=np.float32)  # ~2.1GB
miss = 0
for u in range(nU):
    us = str(u)
    if us not in uid_map:
        miss += 1; continue
    sc = np.asarray(m.score(int(uid_map[us])), dtype=np.float32)
    scores[u, our_known] = sc[cidx_known]
    if (u + 1) % 20000 == 0:
        print(f"  scored {u+1}/{nU}", flush=True)
print(f"utenti senza score (cold)={miss}", flush=True)

np.save(f"{BB}/FM.scores.npy", scores)
np.save(f"{BB}/Bfull.scores.npy", scores[test_users])  # STUB (non context-aware)
print(f"[OK] FM.scores.npy {scores.shape} + Bfull.scores.npy (stub) {scores[test_users].shape} -> {BB}")
