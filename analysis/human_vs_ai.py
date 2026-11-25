#!/usr/bin/env python3
"""Human vs AI comparison (Task 5): the formal tests and the two key charts.

Run (after build_master_dataset.py):   python analysis/human_vs_ai.py

Samples
  Humans: the three replication datasets (UM online, UTD online, UM in-person), comprehension failures removed,
          plus the original Duke study from its published summary numbers (raw data not available).
  AI:     every model x wording in data/ai_dataset_all_long.csv. PRIMARY rule = the original study's rule:
          drop agents that got the dice range (3 and 18) wrong. The profit check is our own addition, so it is
          not used to exclude agents here (its failure rate differs by scheme, which would bias O vs C).

For each sample
  gap O-C of participants' 25-round averages, 95% CI (Welch) and p (pooled-variance t-test = the paper's
  OLS/contrast test); round-1 gap (the only decision not shaped by feedback); follow-up question gap (if asked);
  within-participant SD; share of orders at 10-11.
For each AI run, a pooled regression with all human replications:
  order ~ O + AI + O x AI, standard errors clustered by participant (CR1, as R vcovCL type="HC1").
  O x AI = how much smaller (negative) or larger the AI's O-C gap is than the human gap.

Outputs
  analysis/human_vs_ai_table.csv     one row per sample
  analysis/fig_gap_ci.png            O-C gap with 95% CI for every sample (the headline chart)
  analysis/fig_human_vs_ai_days.png  average order by day, humans vs the AI base runs (small multiples)
"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore", message="Precision loss")   # round-1 orders can be identical across agents
ORD = [f"order{i}" for i in range(1, 26)]
HUMAN = {"UM online": "ChenEtAl_OnlineReplication_Mich.csv", "UTD online": "ChenEtAl_OnlineReplication_UTD.csv",
         "UM in-person": "ChenEtAl_InPersonReplication_Mich.csv"}
# Original Duke study, Table 3 of Chen et al. (2013): means (SD) of the 25-round average, round-1 and follow-up gaps
ORIG = dict(mO=11.728, sO=1.392, nO=29, mC=9.749, sC=1.058, nC=21, p=1.612e-6, r1=1.831, fu=2.188)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"
SRC_COL = {"Human": "#52514e", "Claude Haiku 5.5": "#2a78d6", "Llama 3.1": "#1baf7a"}


# ------------------------------------------------------------------ helpers
def ols_cluster(y, X, groups):
    """OLS with cluster-robust CR1 standard errors (same as R lm + vcovCL(type='HC1')); df = N-K like R."""
    y, X, g = np.asarray(y, float), np.asarray(X, float), np.asarray(groups)
    n, k = X.shape; XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y; u = y - X @ b
    G = len(np.unique(g)); meat = np.zeros((k, k))
    for gid in np.unique(g):
        m = g == gid; s = X[m].T @ u[m]; meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V)); p = 2 * stats.t.sf(np.abs(b / se), n - k)
    return b, se, p


def welch_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean(); h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h


def load_humans():
    frames = []
    for site, f in HUMAN.items():
        d = pd.read_csv(ROOT / "data" / "human" / f, encoding="utf-8-sig")
        d = d.dropna(subset=["Participant"])               # the UTD file has summary rows with no participant
        d = d[d["ComprehensionFail"] == 0].copy()
        d["pid"] = site + "_" + d["Participant"].astype(str); d["site"] = site
        frames.append(d)
    w = pd.concat(frames, ignore_index=True)
    l = w.melt(id_vars=["pid", "Condition", "site"], value_vars=ORD, var_name="r", value_name="order").dropna(subset=["order"])
    l["day"] = l["r"].str[5:].astype(int)
    return l.rename(columns={"Condition": "condition"})[["pid", "condition", "day", "order", "site"]]


def load_ai():
    L = pd.read_csv(ROOT / "data" / "ai_dataset_all_long.csv")
    P = pd.read_csv(ROOT / "data" / "ai_dataset_all_participants.csv")
    keep = P[P["fail_range_check"] == 0][["model", "wording", "agent_id"]]          # original exclusion rule
    excluded = P.groupby(["model", "wording"])["fail_range_check"].sum()
    L = L.merge(keep, on=["model", "wording", "agent_id"])
    L["pid"] = L["model"] + "|" + L["wording"] + "|" + L["agent_id"]
    return L, P, excluded


def describe(l, label, source, followup=None):
    avg = l.groupby(["pid", "condition"])["order"].mean().reset_index()
    o, c = avg[avg.condition == "O"]["order"], avg[avg.condition == "C"]["order"]
    gap, lo, hi = welch_ci(o, c)
    r1 = l[l.day == 1]; r1o, r1c = r1[r1.condition == "O"]["order"], r1[r1.condition == "C"]["order"]
    row = dict(sample=label, source=source, n_O=len(o), n_C=len(c), mean_O=o.mean(), mean_C=c.mean(),
               gap=gap, ci_low=lo, ci_high=hi, p_gap=stats.ttest_ind(o, c).pvalue,
               round1_gap=r1o.mean() - r1c.mean(), p_round1=stats.ttest_ind(r1o, r1c).pvalue,
               within_sd=l.groupby("pid")["order"].std().mean(), between_sd=avg["order"].std(),
               pct_10_11=100 * l["order"].isin([10, 11]).mean())
    if followup is not None and followup.notna().any():
        fo, fc = followup[followup.index.get_level_values("condition") == "O"], followup[followup.index.get_level_values("condition") == "C"]
        row.update(followup_gap=fo.mean() - fc.mean(), p_followup=stats.ttest_ind(fo.dropna(), fc.dropna()).pvalue)
    return row


def interaction(hl, al):
    P = pd.concat([hl.assign(ai=0.0), al.assign(ai=1.0)])
    o = (P.condition == "O").astype(float)
    b, se, p = ols_cluster(P["order"], np.c_[np.ones(len(P)), o, P["ai"], o * P["ai"]], P["pid"])
    return dict(OxAI=b[3], OxAI_se=se[3], OxAI_p=p[3], AI_shift_in_C=b[2], AI_shift_p=p[2])


# ------------------------------------------------------------------ table
def main():
    hl = load_humans()
    L, P, excluded = load_ai()
    rows = []
    se =np.sqrt(ORIG["sO"] ** 2 / ORIG["nO"] + ORIG["sC"] ** 2 / ORIG["nC"])
    rows.append(dict(sample="Original (Duke)", source="Human", n_O=ORIG["nO"], n_C=ORIG["nC"], mean_O=ORIG["mO"], mean_C=ORIG["mC"],
                     gap=ORIG["mO"] - ORIG["mC"], ci_low=ORIG["mO"] - ORIG["mC"] - 1.96 * se, ci_high=ORIG["mO"] - ORIG["mC"] + 1.96 * se,
                     p_gap=ORIG["p"], round1_gap=ORIG["r1"], followup_gap=ORIG["fu"]))
    for site in HUMAN:
        rows.append(describe(hl[hl.site == site], f"Human – {site}", "Human"))
    rows.append(describe(hl, "Human – 3 replications pooled", "Human"))

    order = ["base", "faithful", "v1", "v2", "v3"]
    for (m, w), al in sorted(L.groupby(["model", "wording"]), key=lambda x: (x[0][0], order.index(x[0][1]) if x[0][1] in order else 9)):
        pm = P[(P.model == m) & (P.wording == w) & (P.fail_range_check == 0)]
        fu = pm.set_index(["agent_id", "condition"])["followup_order"] if "followup_order" in pm else None
        r = describe(al, f"{m} – {w}", m, fu)
        r.update(interaction(hl, al), excluded_range_fail=int(excluded.get((m, w), 0)))
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(ROOT / "analysis" / "human_vs_ai_table.csv", index=False)

    show = T[["sample", "n_O", "n_C", "mean_O", "mean_C", "gap", "ci_low", "ci_high", "p_gap", "round1_gap", "OxAI", "OxAI_p"]]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3g}".format):
        print(show.to_string(index=False))

    # ------------------------------------------------------------------ figure 1: gap with 95% CI
    T2 = T.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(T2) + 1.2), dpi=160)
    for i, r in T2.iterrows():
        col = SRC_COL.get(r["source"], INK)
        ax.plot([r.ci_low, r.ci_high], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.gap, i, "o", color=col, ms=7, mec="white", mew=1.5)
        ax.text(max(r.ci_high, r.gap) + 0.08, i, f"{r.gap:+.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    hp = T[T["sample"] == "Human – 3 replications pooled"].iloc[0]
    ax.axvspan(hp.ci_low, hp.ci_high, color="#d9d8d3", alpha=0.45, lw=0)
    ax.set_yticks(range(len(T2))); ax.set_yticklabels(T2["sample"].str.replace("– faithful", "– human-like feedback", regex=False), fontsize=8)
    ax.set_xlabel("O − C gap in average order (units), with 95% confidence interval")
    fig.suptitle("Humans order more under own financing; the AI agents barely do", x=0.01, ha="left", fontsize=11, color=INK)
    ax.set_title("Grey band = pooled human replications. Gap = 0 means no framing effect.", loc="left", fontsize=8, color=MUTED)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(ROOT / "analysis" / "fig_gap_ci.png", bbox_inches="tight"); plt.close(fig)

    # ------------------------------------------------------------------ figure 2: by day, small multiples
    panels = [("Humans (3 replications)", hl)]
    for m in ["Claude Haiku 5.5", "Llama 3.1"]:
        for w, name in [("base", "study wording"), ("faithful", "study wording, human-like feedback")]:
            d = L[(L.model == m) & (L.wording == w)]
            if len(d): panels.append((f"{m}\n{name}", d))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.6), dpi=160, sharey=True)
    for ax, (title, d) in zip(np.atleast_1d(axes), panels):
        ends = {}
        for cond, col in [("O", BLUE), ("C", ORANGE)]:
            m = d[d.condition == cond].groupby("day")["order"].mean()
            ax.plot(m.index, m.values, color=col, lw=2); ends[cond] = m.values[-1]
        nudge = max(0, 0.7 - abs(ends["O"] - ends["C"])) / 2          # keep end labels apart when lines meet
        up = 1 if ends["O"] >= ends["C"] else -1
        ax.text(25.6, ends["O"] + up * nudge, "O (own)", color=INK, fontsize=7, va="center")
        ax.text(25.6, ends["C"] - up * nudge, "C (customer)", color=INK, fontsize=7, va="center")
        ax.axhline(10.5, color=MUTED, lw=1, ls=":")
        ax.set_title(title, fontsize=9, loc="left", color=INK); ax.set_xlabel("Day", fontsize=8)
        ax.set_xlim(1, 25); ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Average order (dotted = mean demand 10.5)", fontsize=8)
    fig.suptitle("Average order by day: only the humans keep a steady O–C gap",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); fig.savefig(ROOT / "analysis" / "fig_human_vs_ai_days.png", bbox_inches="tight"); plt.close(fig)
    print("wrote analysis/human_vs_ai_table.csv, fig_gap_ci.png, fig_human_vs_ai_days.png")


if __name__ == "__main__":
    main()
