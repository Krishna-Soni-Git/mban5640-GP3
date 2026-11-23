#!/usr/bin/env python3
"""Compile AI-agent logs -> CSV, validate, and compare with the human data (original + 3 replications).
Run:  python3 analysis/analyze.py            (from the project root)
Outputs: data/ai_agent_long.csv, data/ai_agent_wide.csv, data/ai_agent_participants.csv,
         analysis/results.json, analysis/fig_*.png, analysis/summary_table.csv"""
import json, glob, math, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parent.parent
Q_STAR = 10.5  # expected-profit-maximising order for 3d6 with p=2,c=1: F(10)=0.5 -> any q in [10,11]

# ---------- 1. compile ----------
STATE = Path(os.environ.get("AI_STATE_DIR", ROOT/"data/state")); STATE = STATE if STATE.is_absolute() else ROOT/STATE
TAG = os.environ.get("AI_TAG", "")
rows, parts = [], []
for f in sorted(glob.glob(str(STATE/"A*.json"))):
    s = json.loads(Path(f).read_text())
    if len(s["rounds"]) < 25 or not s["debrief"]:   # incomplete runs are reported, not analysed
        print("INCOMPLETE:", s["id"], len(s["rounds"])); continue
    for r in s["rounds"]:
        rows.append(dict(agent_id=s["id"], condition=s["cond"], persona=s["persona"], **r))
    def num(x):
        try: return int(float(str(x).replace("+", "").strip()))
        except Exception: return None
    d = s["debrief"]
    ok_profit = num(d["answer"]) == 4                                  # check 1: order 10, demand 7 -> +4
    has_range = "range_min" in d
    ok_range = (num(d.get("range_min")) == 3 and num(d.get("range_max")) == 18) if has_range else True   # check 2: dice range 3..18
    parts.append(dict(agent_id=s["id"], condition=s["cond"], persona=s["persona"], comprehension_answer=d["answer"],
                      range_min_answer=d.get("range_min"), range_max_answer=d.get("range_max"),
                      fail_profit_check=int(not ok_profit), fail_range_check=int(not ok_range),
                      comprehension_fail=int(not (ok_profit and ok_range)), final_cash=s["cash"], strategy=d["strategy"]))
long = pd.DataFrame(rows); P = pd.DataFrame(parts)
long.to_csv(ROOT/f"data/ai_agent_long{TAG}.csv", index=False)
wide = long.pivot(index="agent_id", columns="day", values="order"); wide.columns = [f"order{c}" for c in wide.columns]
wide = P[["agent_id","comprehension_fail","condition"]].merge(wide.reset_index(), on="agent_id")
wide.rename(columns={"agent_id":"Participant","comprehension_fail":"ComprehensionFail","condition":"Condition"}, inplace=True)
wide.to_csv(ROOT/f"data/ai_agent_wide{TAG}.csv", index=False)

# ---------- 2. validate ----------
chk = dict(n_agents=len(P), n_rows=len(long), orders_out_of_range=int(((long.order<3)|(long.order>18)).sum()),
           missing=int(long.isna().sum().sum()), demand_mismatch=int((long.demand != long.d1+long.d2+long.d3).sum()),
           comprehension_fail=int(P.comprehension_fail.sum()),
           fail_profit_check=int(P.fail_profit_check.sum()), fail_range_check=int(P.fail_range_check.sum()))
# payoff rule check (O: 2*sold-q ; C: q-2*left)
exp = np.where(long.condition=="O", 2*long.sold - long.order, long.order - 2*long.leftover)
chk["payoff_rule_violations"] = int((exp != long.change).sum())
chk["dice_mean_demand"] = round(float(long.demand.mean()), 3)  # should be ~10.5
P["avg_order"] = long.groupby("agent_id").order.mean().reindex(P.agent_id).values
P.to_csv(ROOT/f"data/ai_agent_participants{TAG}.csv", index=False)

# ---------- 3. human data ----------
def load_h(fn):
    d = pd.read_csv(ROOT/"data/human"/fn, encoding="utf-8-sig"); oc = [c for c in d if c.startswith("order")]
    d["avg_order"] = d[oc].mean(axis=1); return d, oc
H = {"Original (Duke) *": None, "UM online": "ChenEtAl_OnlineReplication_Mich.csv",
     "UTD online": "ChenEtAl_OnlineReplication_UTD.csv", "UM in-person": "ChenEtAl_InPersonReplication_Mich.csv"}

def cluster_test(df_long):
    """OLS order ~ O dummy, SEs clustered by participant, HC1 small-sample (matches R vcovCL type='HC1')."""
    y = df_long.order.values.astype(float); x = (df_long.condition=="O").astype(float).values
    X = np.c_[np.ones_like(x), x]; XtXi = np.linalg.inv(X.T@X); b = XtXi@X.T@y; e = y - X@b
    g = df_long.agent.values; G = len(np.unique(g)); n, k = X.shape; meat = np.zeros((k,k))
    for gi in np.unique(g):
        m = g == gi; sg = X[m].T@e[m]; meat += np.outer(sg, sg)
    V = XtXi@meat@XtXi * (G/(G-1)) * ((n-1)/(n-k)); se = math.sqrt(V[1,1])
    t = b[1]/se; p = 2*(1-stats.t.cdf(abs(t), G-1)); return b[1], se, p

def summarise(label, wide_df, oc, agents_long=None):
    d = wide_df[wide_df.ComprehensionFail == 0]
    o, c = d[d.Condition=="O"].avg_order, d[d.Condition=="C"].avg_order
    t = stats.ttest_ind(o, c, equal_var=True); lw = d.melt(id_vars=["Participant","Condition"], value_vars=oc, value_name="order").rename(columns={"Participant":"agent","Condition":"condition"})
    b, se, pc = cluster_test(lw)
    pooled = math.sqrt(((len(o)-1)*o.var()+(len(c)-1)*c.var())/(len(o)+len(c)-2))
    mw = stats.mannwhitneyu(o, c, alternative="two-sided")
    return dict(sample=label, n_O=len(o), n_C=len(c), mean_O=o.mean(), mean_C=c.mean(), sd_between_O=o.std(), sd_between_C=c.std(),
                diff=o.mean()-c.mean(), t=t.statistic, p_ols=t.pvalue, p_cluster=pc, p_mannwhitney=mw.pvalue, cohen_d=(o.mean()-c.mean())/pooled,
                ci_lo=(o.mean()-c.mean())-1.96*math.sqrt(o.var()/len(o)+c.var()/len(c)), ci_hi=(o.mean()-c.mean())+1.96*math.sqrt(o.var()/len(o)+c.var()/len(c)),
                dev_O=o.mean()-Q_STAR, dev_C=c.mean()-Q_STAR)

res = []
for lab, fn in H.items():
    if fn is None:  # original study numbers come from the replication report (Table 1A); raw original data not provided
        res.append(dict(sample=lab, n_O=29, n_C=21, mean_O=11.728, mean_C=9.749, diff=1.979, p_ols=1.612e-6, dev_O=11.728-Q_STAR, dev_C=9.749-Q_STAR))
        continue
    d, oc = load_h(fn); res.append(summarise(lab, d.rename(columns={}), oc))
ai_wide = wide.copy(); ai_wide["avg_order"] = ai_wide[[f"order{i}" for i in range(1,26)]].mean(axis=1)
res.append(summarise("AI agents (all 50)", ai_wide.assign(ComprehensionFail=0), [f"order{i}" for i in range(1,26)]))
if wide.ComprehensionFail.sum() and (wide.ComprehensionFail==0).groupby(wide.Condition).sum().min()>=3:
    res.append(summarise("AI agents (passed check)", ai_wide, [f"order{i}" for i in range(1,26)]))
R = pd.DataFrame(res); R.to_csv(ROOT/f"analysis/summary_table{TAG}.csv", index=False)

# ---------- 4. extra AI diagnostics ----------
extra = {}
ai = long.copy()
extra["share_orders_equal_10_or_11"] = ai.groupby("condition").order.apply(lambda s: float(s.isin([10,11]).mean())).to_dict()
extra["within_agent_sd"] = ai.groupby(["condition","agent_id"]).order.std().groupby("condition").mean().to_dict()
extra["mean_final_cash"] = P.groupby("condition").final_cash.mean().to_dict()
extra["order_by_half"] = {f"{c}_{h}": float(ai[(ai.condition==c)&(ai.day.between(*r))].order.mean()) for c in "OC" for h, r in {"days1_5":(1,5),"days6_25":(6,25)}.items()}
# human within-participant SD for comparison
for lab, fn in H.items():
    if fn:
        d, oc = load_h(fn); d = d[d.ComprehensionFail==0]; extra.setdefault("human_within_sd", {})[lab] = d.groupby("Condition").apply(lambda x: float(x[oc].std(axis=1).mean())).to_dict()
# equivalence of payoffs: realised profit per day by condition
extra["mean_daily_profit"] = ai.groupby("condition").change.mean().to_dict()
# pooled human replications
frames = []
for lab, fn in H.items():
    if fn: d, oc = load_h(fn); frames.append(d[d.ComprehensionFail==0][["Condition","avg_order"]])
hp = pd.concat(frames); extra["human_pooled_diff"] = float(hp[hp.Condition=="O"].avg_order.mean() - hp[hp.Condition=="C"].avg_order.mean())
json.dump(dict(checks=chk, extra=extra, table=json.loads(R.to_json(orient="records"))), open(ROOT/f"analysis/results{TAG}.json","w"), indent=1, default=float)

# ---------- 5. figures ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
col = {"O":"#C0504D", "C":"#2E75B6"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
labs = [r["sample"] for r in res if r["sample"] != "AI agents (passed check)"]
x = np.arange(len(labs)); w = .36
ax[0].bar(x-w/2, [r["mean_O"] for r in res if r["sample"] in labs], w, color=col["O"], label="O (own financing)")
ax[0].bar(x+w/2, [r["mean_C"] for r in res if r["sample"] in labs], w, color=col["C"], label="C (customer financing)")
ax[0].axhline(Q_STAR, ls="--", c="k", lw=1, label="profit-maximizing (10–11)")
ax[0].set_xticks(x); ax[0].set_xticklabels([l.replace(" (","\n(") for l in labs], fontsize=8); ax[0].set_ylim(8, 12.5); ax[0].set_ylabel("Mean order quantity"); ax[0].legend(fontsize=8); ax[0].set_title("Mean order by payment scheme")
for cnd in "OC":
    m = ai[ai.condition==cnd].groupby("day").order.mean(); ax[1].plot(m.index, m.values, c=col[cnd], marker="o", ms=3, label=f"AI – {cnd}")
ax[1].axhline(Q_STAR, ls="--", c="k", lw=1); ax[1].set_xlabel("Day"); ax[1].set_ylabel("Mean order"); ax[1].set_title("AI-agent orders by day"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(ROOT/f"analysis/fig_comparison{TAG}.png", dpi=160); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
bins = np.arange(2.5, 19.5, 1)
hum = pd.concat([load_h(fn)[0].assign(src=lab) for lab, fn in H.items() if fn]); hum = hum[hum.ComprehensionFail==0]
hoc = [c for c in hum if c.startswith("order")]
for a, (ttl, data) in zip(ax, [("Human (3 replications pooled)", hum.melt(id_vars=["Condition"], value_vars=hoc, value_name="order").rename(columns={"Condition":"condition"})), ("AI agents", ai)]):
    for cnd in "OC": a.hist(data[data.condition==cnd].order, bins=bins, alpha=.55, color=col[cnd], density=True, label=cnd)
    a.axvline(Q_STAR, ls="--", c="k", lw=1); a.set_title(ttl); a.set_xlabel("Order quantity"); a.legend()
ax[0].set_ylabel("Share of decisions"); plt.tight_layout(); plt.savefig(ROOT/f"analysis/fig_distributions{TAG}.png", dpi=160); plt.close()

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("CHECKS", json.dumps(chk)); print(R.round(4).to_string()); print(json.dumps(extra, indent=1, default=float))
