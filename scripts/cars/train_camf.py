"""[venv-xsage] Baseline CARS — CAMF (Context-Aware Matrix Factorization, Baltrunas et al. 2011),
nel NOSTRO harness/task top-N (Cat-MRR/R@20/LT/Gini), confronto apples-to-apples con SIT/B_full/Steck-b.
Due varianti:
  - CAMF_C : s(u,i,c)=μ+b_u+b_i+p_u·q_i+Σ_d bias_d[c_d]            (contesto = bias additivi)
  - CAMF_CI: s(u,i,c)=...+Σ_d B_d[c_d, macro(i)]                   (contesto × CATEGORIA — rivale di SIT)
Contesto = (hour, dow, isweekend, month). Loss BPR (ranking). Anti-circolare: train per il modello,
val per early-stop/selezione, test misurato una volta. Salva outputs_results/cars_<city>.csv.

Uso:  python scripts/cars/train_camf.py <city> [variant=both]
"""
import sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
CLEAN = Path("/Users/lucaaliberti/Downloads/xsage-clean")
sys.path.insert(0, str(CLEAN)); sys.path.insert(0, "/Users/lucaaliberti/Downloads/IntentAwareRS_thesis")
from xsage import data as D
from xsage.metrics import long_tail_groups

EMB_D, EPOCHS, LR, L2, NEG, BATCH, SEED = 32, 12, 0.01, 1e-5, 1, 4096, 42
K_TOP = 20
CTX = [("c_hour", 24), ("c_dow", 7), ("c_isweekend", 2), ("c_month", 13)]


def gini(x):
    x = np.sort(np.asarray(x, np.float64)); n = len(x); s = x.sum()
    return 0.0 if s <= 0 else float((2*np.sum(np.arange(1, n+1)*x)/(n*s)) - (n+1)/n)


class CAMF(nn.Module):
    def __init__(self, nU, nI, nM, d, variant):
        super().__init__(); self.variant = variant; self.nM = nM
        self.P = nn.Embedding(nU, d); self.Q = nn.Embedding(nI, d)
        self.bu = nn.Embedding(nU, 1); self.bi = nn.Embedding(nI, 1)
        self.mu = nn.Parameter(torch.zeros(1))
        for e in (self.P, self.Q): nn.init.normal_(e.weight, std=0.01)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)
        sz = 1 if variant == "C" else nM
        self.ctx = nn.ModuleDict({c: nn.Embedding(n, sz) for c, n in CTX})
        for e in self.ctx.values(): nn.init.zeros_(e.weight)

    def _ctx_pos(self, ctxd, mi):  # bias contesto per (richiesta, item osservato) → (B,)
        s = 0.0
        for c, _ in CTX:
            b = self.ctx[c](ctxd[c])                       # (B,1) o (B,nM)
            s = s + (b.squeeze(-1) if self.variant == "C" else b.gather(1, mi[:, None]).squeeze(-1))
        return s

    def score_pos(self, u, i, ctxd, mi):
        return (self.mu + self.bu(u).squeeze(-1) + self.bi(i).squeeze(-1)
                + (self.P(u) * self.Q(i)).sum(-1) + self._ctx_pos(ctxd, mi))

    def score_all(self, u, ctxd, icm_t):  # (B, nI) per il ranking
        s = (self.mu + self.bu(u) + self.bi.weight.squeeze(-1)[None, :]
             + self.P(u) @ self.Q.weight.t())               # (B,nI)
        for c, _ in CTX:
            b = self.ctx[c](ctxd[c])
            s = s + (b if self.variant == "C" else b[:, icm_t])  # (B,1) broadcast | (B,nI) gather macro
        return s


def feats(df, dev):
    return {c: torch.from_numpy(df[c].values.astype(np.int64)).to(dev) for c, _ in CTX}


def eval_split(model, df, icm, icm_t, excl, G1, dev, nI):
    u = df["u_idx"].values.astype(np.int64); it = df["i_idx"].values.astype(np.int64); tm = icm[it]
    ctxd = feats(df, dev); n = len(df)
    cm = np.zeros(n); hit = np.zeros(n); lt = np.zeros(n); expo = np.zeros(nI)
    model.eval()
    with torch.no_grad():
        for bs in range(0, n, 512):
            be = min(n, bs+512); ub = torch.from_numpy(u[bs:be]).to(dev)
            cb = {c: ctxd[c][bs:be] for c, _ in CTX}
            S = model.score_all(ub, cb, icm_t).cpu().numpy()
            for j in range(be-bs):
                uu = int(u[bs+j]); cc = excl.indices[excl.indptr[uu]:excl.indptr[uu+1]]
                if len(cc): S[j, cc] = -np.inf
            part = np.argpartition(-S, K_TOP-1, axis=1)[:, :K_TOP]
            order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
            topk = np.take_along_axis(part, order, 1); macros = icm[topk]
            match = macros == tm[bs:be, None]; has = match.any(1); first = np.where(has, match.argmax(1)+1, 0)
            cm[bs:be] = np.where(first > 0, 1.0/np.maximum(first, 1), 0.0)
            s_t = S[np.arange(be-bs), it[bs:be]]; hit[bs:be] = ((S > s_t[:, None]).sum(1)+1 <= K_TOP)
            lt[bs:be] = G1[topk].mean(1); np.add.at(expo, topk.ravel(), 1.0)
    uq, inv = np.unique(u, return_inverse=True); ss = np.zeros(len(uq)); cc = np.zeros(len(uq))
    np.add.at(ss, inv, hit); np.add.at(cc, inv, 1)
    return dict(CatMRR=float(cm.mean()), R20=float((ss/cc).mean()), LT20=float(lt.mean()),
                Coverage=float((expo > 0).mean()), Gini=gini(expo))


def train_eval(city, variant, ds, icm, icm_t, excl, G1, dev, nU, nI, nM):
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = CAMF(nU, nI, nM, EMB_D, variant).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=L2)
    tr = ds["df_train"]; u = tr["u_idx"].values.astype(np.int64); it = tr["i_idx"].values.astype(np.int64)
    mi = icm[it]; ctxd = feats(tr, dev); ntr = len(tr); best = (-1, None)
    for ep in range(EPOCHS):
        m.train(); perm = np.random.permutation(ntr)
        for bs in range(0, ntr, BATCH):
            idx = perm[bs:bs+BATCH]
            ub = torch.from_numpy(u[idx]).to(dev); ib = torch.from_numpy(it[idx]).to(dev)
            mib = torch.from_numpy(mi[idx]).to(dev)
            jb = torch.from_numpy(np.random.randint(0, nI, len(idx))).to(dev)
            mjb = icm_t[jb]
            cb = {c: ctxd[c][idx] for c, _ in CTX}
            sp = m.score_pos(ub, ib, cb, mib); sn = m.score_pos(ub, jb, cb, mjb)
            loss = -torch.nn.functional.logsigmoid(sp - sn).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep >= 3 and ep % 2 == 1:
            v = eval_split(m, ds["df_val"], icm, icm_t, excl, G1, dev, nI)["CatMRR"]
            if v > best[0]: best = (v, {k: x.detach().clone() for k, x in m.state_dict().items()})
    if best[1]: m.load_state_dict(best[1])
    return eval_split(m, ds["df_test"], icm, icm_t, excl, G1, dev, nI)


def main():
    city = sys.argv[1] if len(sys.argv) > 1 else "ml1m"
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    import os
    dev = torch.device("cpu" if os.environ.get("CAMF_CPU") == "1"
                       else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ds = D.load_city(city, data_root=str(CLEAN)); m2i = ds["macro_to_idx"]
    nU, nI, nM = ds["n_users"], ds["n_items"], ds["n_macros"]
    dfa = __import__("pandas").concat([ds["df_train"], ds["df_val"], ds["df_test"]], ignore_index=True)
    icm = (dfa.groupby("i_idx")["cat_macro"].first().map(m2i).reindex(np.arange(nI), fill_value=0).values.astype(np.int64))
    icm_t = torch.from_numpy(icm).to(dev)
    pop = np.asarray((ds["urm_train"] + ds["urm_val"]).sum(0)).ravel(); _, G1 = long_tail_groups(pop, short_head_share=0.20)
    excl = D.load_excluded_mask(city, nI, data_root=str(CLEAN))
    variants = ["C", "CI"] if which == "both" else [which]
    import pandas as pd
    rows = []
    print(f"=== CAMF su {city} (device={dev}) nU={nU} nI={nI} nM={nM} ===", flush=True)
    for v in variants:
        r = train_eval(city, v, ds, icm, icm_t, excl, G1, dev, nU, nI, nM)
        r = {"city": city, "method": f"CAMF_{v}", **r}; rows.append(r)
        print(f"  CAMF_{v}: CatMRR={r['CatMRR']:.4f} R@20={r['R20']:.4f} LT={r['LT20']:.4f} Cov={r['Coverage']:.4f} Gini={r['Gini']:.4f}", flush=True)
    OUT = CLEAN / "outputs_results"; pd.DataFrame(rows).to_csv(OUT / f"cars_{city}.csv", index=False)
    print(f"→ outputs_results/cars_{city}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
