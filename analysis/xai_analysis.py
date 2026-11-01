#!/usr/bin/env python3
"""XAI step for the data-mining workflow (Lecture Note 03): what drives an agent's / person's next order?
 - Interpretable model: OLS coefficients (order_t on condition, day, previous order, [previous demand]).
 - Black-box model: gradient boosting, explained with EXACT Shapley values (average marginal contribution over all
   feature subsets, interventional expectation over a background sample), exactly the procedure on lecture slides 17-22.
Human files contain orders only (no dice), so the like-for-like comparison uses features available for both groups:
 cond_O, day, prev_order.  AI data also has dice, so an extended AI model adds prev_demand and prev_change.
Run: python3 analysis/xai_analysis.py   -> analysis/xai_results.json, analysis/fig_xai.png"""
import json, itertools, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
ROOT = Path(__file__).resolve().parent.parent
rng = np.random.default_rng(0)

def human_long():
    fr = []
    for fn in ["ChenEtAl_OnlineReplication_Mich.csv","ChenEtAl_OnlineReplication_UTD.csv","ChenEtAl_InPersonReplication_Mich.csv"]:
        d = pd.read_csv(ROOT/"data/human"/fn, encoding="utf-8-sig"); d = d[d.ComprehensionFail==0]
        d["pid"] = fn[:12]+"_"+d.Participant.astype(str); fr.append(d)
    d = pd.concat(fr); oc = [f"order{i}" for i in range(1,26)]
    l = d.melt(id_vars=["pid","Condition"], value_vars=oc, var_name="day", value_name="order")
    l["day"] = l.day.str[5:].astype(int); l = l.sort_values(["pid","day"])
    l["prev_order"] = l.groupby("pid").order.shift(1); l["cond_O"] = (l.Condition=="O").astype(int)
    return l.dropna(subset=["prev_order"]).rename(columns={"pid":"id"})

def ai_long():
    l = pd.read_csv(ROOT/"data/ai_agent_long.csv").sort_values(["agent_id","day"])
    g = l.groupby("agent_id")
    l["prev_order"] = g.order.shift(1); l["prev_demand"] = g.demand.shift(1); l["prev_change"] = g.change.shift(1)
    l["cond_O"] = (l.condition=="O").astype(int)
    return l.dropna(subset=["prev_order"]).rename(columns={"agent_id":"id"})

def ols(df, feats):
    X = np.c_[np.ones(len(df)), df[feats].values.astype(float)]; y = df["order"].values.astype(float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X@b
    ids = df["id"].values; G = len(np.unique(ids)); XtXi = np.linalg.inv(X.T@X); meat = np.zeros((X.shape[1],)*2)
    for gi in np.unique(ids):
        m = ids == gi; s = X[m].T@e[m]; meat += np.outer(s, s)
    V = XtXi@meat@XtXi*(G/(G-1)); se = np.sqrt(np.diag(V))
    return {n: dict(coef=float(c), se_cluster=float(s)) for n, c, s in zip(["const"]+feats, b, se)} | {"R2": float(1-e.var()/y.var())}

def exact_shapley(model, X, background, nmax=400):
    """Exact Shapley values: phi_j = sum_S |S|!(k-|S|-1)!/k! * [v(S+j) - v(S)], v(S)=E_bg[f(x_S, bg_rest)]."""
    k = X.shape[1]; idx = rng.choice(len(X), min(nmax, len(X)), replace=False); Xs = X[idx]
    bg = background[rng.choice(len(background), min(60, len(background)), replace=False)]
    subsets = [frozenset(c) for r in range(k+1) for c in itertools.combinations(range(k), r)]
    phi = np.zeros((len(Xs), k))
    def v(x, S):
        Z = bg.copy()
        for j in S: Z[:, j] = x[j]
        return model.predict(Z).mean()
    for i, x in enumerate(Xs):
        val = {S: v(x, S) for S in subsets}
        for j in range(k):
            for S in subsets:
                if j in S: continue
                w = math.factorial(len(S))*math.factorial(k-len(S)-1)/math.factorial(k)
                phi[i, j] += w*(val[S|{j}] - val[S])
    return phi, Xs

def run(df, feats, label):
    X, y, g = df[feats].values.astype(float), df["order"].values.astype(float), df["id"].values
    model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=.05, subsample=.8, random_state=0)
    pred = cross_val_predict(model, X, y, groups=g, cv=GroupKFold(5))
    r2 = float(1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum())
    model.fit(X, y); phi, Xs = exact_shapley(model, X, X)
    imp = dict(zip(feats, np.abs(phi).mean(axis=0).round(4)))
    cond_shap = {"O": float(phi[Xs[:, feats.index("cond_O")]==1, feats.index("cond_O")].mean()),
                 "C": float(phi[Xs[:, feats.index("cond_O")]==0, feats.index("cond_O")].mean())}
    return dict(label=label, cv_R2_by_agent_group=r2, mean_abs_shap={k: float(v) for k, v in imp.items()}, base_value=float(model.predict(X).mean()), cond_O_shap=cond_shap, n=int(len(df)))

H, A = human_long(), ai_long()
common = ["cond_O", "day", "prev_order"]; ext = common + ["prev_demand", "prev_change"]
out = dict(
  ols_human=ols(H, common), ols_ai_common=ols(A, common), ols_ai_extended=ols(A, ext),
  shap_human=run(H, common, "Humans (3 replications)"), shap_ai_common=run(A, common, "AI agents, same features"), shap_ai_extended=run(A, ext, "AI agents + dice features"))
json.dump(out, open(ROOT/"analysis/xai_results.json","w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for a, keys, ttl in [(ax[0], ["shap_human","shap_ai_common"], "Same features, humans vs AI"), (ax[1], ["shap_ai_extended"], "AI agents with dice features")]:
    feats = list(out[keys[0]]["mean_abs_shap"]); x = np.arange(len(feats)); w = .8/len(keys)
    for i, k in enumerate(keys): a.bar(x+i*w-.4+w/2, [out[k]["mean_abs_shap"][f] for f in feats], w, label=out[k]["label"]+f" (CV R²={out[k]['cv_R2_by_agent_group']:.2f})")
    a.set_xticks(x); a.set_xticklabels(feats, fontsize=8); a.set_ylabel("mean |Shapley value| (order units)"); a.set_title(ttl); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(ROOT/"analysis/fig_xai.png", dpi=160)
print(json.dumps(out, indent=1))
