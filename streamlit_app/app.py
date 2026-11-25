"""Payment Framing Lab (Streamlit): the presentation for MBAN 5640 Group Project 3.

Replicates Chen, Kok & Tong (2013) with AI agents. The app is the 10-minute talk itself: nine steps,
moved with Back / Next or the step numbers, reading the project's result files (data/, analysis/).

Run:  python -m streamlit run streamlit_app/app.py
Re-running the analysis updates the numbers; if the page shows old values, press C and clear the cache.
"""
import json, sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OC = [f"order{i}" for i in range(1, 26)]
O_COL, C_COL, OPT = "#C8681A", "#17808A", "#6B5BB5"

st.set_page_config(page_title="Payment Framing Lab", page_icon="🎲", layout="wide")

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stButton button {font-family:'IBM Plex Sans','Segoe UI',sans-serif;}
h1,h2,h3,.hero h1 {font-family:'Bricolage Grotesque','Segoe UI',sans-serif !important; letter-spacing:-.02em;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden; height:0;}
.block-container {padding-top:1.2rem; max-width:1240px;}
.hero {background:linear-gradient(135deg,#14213D 0%,#1F3A66 60%,#17808A 130%); color:#F6F4EE; border-radius:18px; padding:26px 32px; margin-bottom:14px;}
.hero h1 {color:#fff; font-size:2.3rem; margin:0 0 4px 0; line-height:1.1;}
.hero p {color:#C9D3E3; margin:0; font-size:1.02rem;}
.chip {display:inline-block; font:600 .78rem 'IBM Plex Mono',monospace; letter-spacing:.05em; padding:3px 10px; border-radius:99px; margin-right:6px; margin-top:12px;}
.chip.o {background:#F6E3D2; color:#9A4D0C;} .chip.c {background:#D5ECEE; color:#0E5A62;} .chip.n {background:rgba(255,255,255,.16); color:#fff;}
[data-testid="stMetric"] {background:#fff; border:1px solid #D5DCE3; border-radius:14px; padding:14px 18px; box-shadow:0 1px 2px rgba(20,33,61,.05);}
[data-testid="stMetricValue"] {font-family:'Bricolage Grotesque',sans-serif; font-size:2.1rem; color:#14213D;}
[data-testid="stMetricLabel"] p {color:#5A6B7A; text-transform:uppercase; letter-spacing:.07em; font-size:.74rem;}
.stTabs [data-baseweb="tab-list"] {gap:4px; border-bottom:1px solid #D5DCE3; flex-wrap:wrap;}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:.98rem; padding:10px 15px; border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"] {background:#14213D; color:#fff !important;}
.stTabs [aria-selected="true"] p {color:#fff !important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px; background:#fff; box-shadow:0 1px 3px rgba(20,33,61,.06);}
.stButton button {border-radius:10px; font-weight:600;}
.stButton button[kind="primary"] {background:#14213D; border-color:#14213D;}
.card-o {border-left:6px solid #C8681A; background:#fff; border-radius:12px; padding:16px 20px; font-size:1.1rem; box-shadow:0 1px 2px rgba(20,33,61,.06);}
.card-c {border-left:6px solid #17808A; background:#fff; border-radius:12px; padding:16px 20px; font-size:1.1rem; box-shadow:0 1px 2px rgba(20,33,61,.06);}
.tile {background:#F6F4EE; border:1px solid #E1DDD0; border-radius:12px; padding:14px 16px; height:100%; font-size:1.02rem; line-height:1.5;}
.tile b {color:#14213D;}
.diceline {font-size:64px; line-height:1.1; letter-spacing:10px; margin:4px 0 8px 0;}
.kick {font:600 .75rem 'IBM Plex Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:#5A6B7A; margin-bottom:6px;}
.rub {display:inline-block; font:600 .72rem 'IBM Plex Mono',monospace; letter-spacing:.04em; padding:2px 9px; border-radius:99px; background:#E9E5F6; color:#463A8C; margin-left:8px; text-transform:none;}
.big {font-family:'Bricolage Grotesque',sans-serif; font-size:2.4rem; font-weight:700; letter-spacing:-.02em; line-height:1.1; margin:0 0 .5rem 0; color:#14213D;}
.take {font-size:1.25rem; font-weight:600; color:#0E5A62; margin:0 0 .9rem 0;}
.sec {font:700 1.15rem 'Bricolage Grotesque',sans-serif; color:#14213D; margin:0 0 2px 0;}
.cap {font-size:.88rem; color:#5A6B7A; margin-bottom:8px;}
.verdict {background:#fff; border:1px solid #D5DCE3; border-left:6px solid #6B5BB5; border-radius:12px; padding:16px 20px; margin:14px 0; font-size:1.02rem; line-height:1.55;}
.pace-ok {color:#0E7A4F; font-weight:600;} .pace-bad {color:#B3261E; font-weight:600;}
.footer {text-align:center; color:#7A8794; font-size:.8rem; margin:26px 0 6px 0;}
.lead {font-size:1.25rem; line-height:1.5; color:#14202B;}
.fhead {font:700 1.02rem 'Bricolage Grotesque',sans-serif; padding:4px 10px; border-radius:8px; display:inline-block; margin-bottom:2px;}
.fhead.o {background:#F6E3D2; color:#9A4D0C;} .fhead.c {background:#D5ECEE; color:#0E5A62;} .fhead.e {background:#E9E5F6; color:#463A8C;}
.fnote {font-size:.95rem; line-height:1.45; color:#3A4558;}
.fgrid {display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; margin:6px 0 16px 0;}
.fbox {background:#fff; border:1px solid #D5DCE3; border-radius:14px; padding:14px 16px; display:flex; flex-direction:column; gap:10px;}
.fmath {font:italic clamp(.95rem, 1.6vw, 1.25rem) 'Cambria Math','Cambria','Times New Roman',serif; text-align:center; color:#14213D; padding:6px 0; white-space:nowrap;}
.fmath sub {font-size:.7em;}
@media (max-width: 760px) {.fgrid {grid-template-columns:1fr;}}
.vega-bind {font:600 .95rem 'IBM Plex Sans',sans-serif; color:#14213D; margin-top:8px; display:flex; align-items:center; gap:8px;}
.vega-bind input[type=range] {width:320px; accent-color:#C8681A; cursor:pointer;}
</style>
"""


DARK_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {background:#0E1522 !important; color:#E3E8EF;}
.stApp .stMarkdown, .stApp .stMarkdown p, .stApp li, .stApp .katex, [data-testid="stWidgetLabel"] p, .stApp label p {color:#E3E8EF !important;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .cap, .kick, .footer {color:#9AA8B5 !important;}
.big, .sec, .tile b, .stApp h1, .stApp h2, .stApp h3, .stMarkdown b, .stMarkdown strong {color:#F4F6FA !important;}
.take {color:#5CC6CF !important;} .lead {color:#E3E8EF !important;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#151E2E !important; border-color:#2A3850 !important; box-shadow:none !important;}
.tile {background:#1B2638 !important; border-color:#2C3B54 !important; color:#E3E8EF !important;}
.card-o, .card-c, .verdict {background:#1B2638 !important; color:#E3E8EF !important; box-shadow:none !important;}
[data-testid="stMetric"] {background:#1B2638 !important; border-color:#2C3B54 !important; box-shadow:none !important;}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {color:#F4F6FA !important;}
[data-testid="stMetricLabel"] p {color:#9AA8B5 !important;}
.fnote {color:#C9D3E3 !important;} .vega-bind, .vega-bind label, .vega-bind span {color:#E3E8EF !important;} .fbox {background:#1B2638 !important; border-color:#2C3B54 !important;} .fmath {color:#F4F6FA !important;}
.fhead.o {background:#4A2B12 !important; color:#F5B47A !important;} .fhead.c {background:#103E44 !important; color:#7FD3DA !important;} .fhead.e {background:#2E2754 !important; color:#C3B8F5 !important;}
.stApp button {background:#1B2638; color:#E3E8EF; border-color:#34445F;}
.stApp button p {color:#E3E8EF !important;}
.stApp button[kind="primary"], .stApp button[kind="primary"]:hover {background:#17808A !important; border-color:#17808A !important;}
.stApp button[kind="primary"] p {color:#fff !important;}
.stApp button[kind="pillsActive"], .stApp button[kind="segmented_controlActive"] {background:#3A2A1A !important; border-color:#C8681A !important;}
.stApp button[kind="pillsActive"] p, .stApp button[kind="segmented_controlActive"] p {color:#F5B47A !important;}
[data-testid="stExpander"] details, [data-testid="stExpander"] summary {background:#151E2E !important; border-color:#2A3850 !important; color:#E3E8EF !important;}
[data-testid="stCode"] pre, .stCode pre {background:#1B2638 !important;} [data-testid="stCode"] code {color:#E3E8EF !important;}
[data-testid="stAlert"] {background:#1B2638 !important; color:#E3E8EF !important;}
[data-testid="stProgress"] [role="progressbar"] > div {background:#2A3850 !important;} [data-testid="stProgress"] [role="progressbar"] > div > div {background:#C8681A !important;}
blockquote {border-left-color:#5CC6CF !important;} blockquote p {color:#C9D3E3 !important;}
</style>
"""


def is_dark():
    return bool(st.session_state.get("dark", False))


def ink():
    """Colour for 'human' marks: navy on light, near-white on dark."""
    return "#E3E8EF" if is_dark() else "#14213D"


def polish(chart):
    """Consistent chart styling."""
    if is_dark():
        return (chart.configure(background="transparent")
                     .configure_axis(gridColor="#26334A", domainColor="#5A6B7A", tickColor="#5A6B7A", labelFontSize=13, titleFontSize=13, labelColor="#C9D3E3", titleColor="#C9D3E3")
                     .configure_legend(labelFontSize=13, titleFontSize=13, labelColor="#C9D3E3", titleColor="#9AA8B5")
                     .configure_header(labelColor="#E3E8EF", titleColor="#E3E8EF")
                     .configure_view(stroke=None))
    return (chart.configure_axis(gridColor="#E3E7EB", domainColor="#9AA8B5", tickColor="#9AA8B5", labelFontSize=13, titleFontSize=13, labelColor="#3A4558", titleColor="#3A4558")
                 .configure_legend(labelFontSize=13, titleFontSize=13)
                 .configure_view(stroke=None))


def btn(container, label, **k):
    """Full-width button on old and new Streamlit versions."""
    try:
        return container.button(label, width="stretch", **k)
    except TypeError:
        return container.button(label, use_container_width=True, **k)


def _w(fn, *a, **k):
    """Call a Streamlit display function with full width on both old and new Streamlit versions."""
    try:
        return fn(*a, width="stretch", **k)
    except TypeError:
        return fn(*a, use_container_width=True, **k)


def tile(container, html):
    container.markdown(f"<div class='tile'>{html}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data
def load_all():
    R = json.loads((ROOT / "analysis/results.json").read_text(encoding="utf-8"))
    samples = []
    for r in R["table"]:
        samples.append(dict(sample=r["sample"].replace(" *", ""), n_O=int(r["n_O"]), n_C=int(r["n_C"]), mean_O=r["mean_O"], mean_C=r["mean_C"],
                            diff=r["diff"], d=r.get("cohen_d"), p=r.get("p_ols")))
    frames = []
    for f in ["ChenEtAl_OnlineReplication_Mich.csv", "ChenEtAl_OnlineReplication_UTD.csv", "ChenEtAl_InPersonReplication_Mich.csv"]:
        d = pd.read_csv(DATA / "human" / f, encoding="utf-8-sig")
        frames.append(d[d.ComprehensionFail == 0])
    H = pd.concat(frames)
    H = H.assign(avg=H[OC].mean(axis=1))
    out = dict(samples=samples, human=H, checks=R["checks"], extra=R["extra"], results=R)
    for tag, key in [("", "claude"), ("_langchain", "llama"), ("_faithful", "claude_f")]:
        wf = DATA / f"ai_agent_wide{tag}.csv"
        if wf.exists():
            W = pd.read_csv(wf); W = W.assign(avg=W[OC].mean(axis=1))
            P_ = pd.read_csv(DATA / f"ai_agent_participants{tag}.csv")
            # primary analysis = the original study's rule: drop agents that got the dice range (3 and 18) wrong.
            # "wide_all" keeps every agent (used by the replay tab and the failure counts).
            keep = set(P_[P_.get("fail_range_check", 0) == 0].agent_id) if "fail_range_check" in P_ else set(P_.agent_id)
            out[key] = dict(wide=W[W.Participant.isin(keep)].copy(), wide_all=W, long=pd.read_csv(DATA / f"ai_agent_long{tag}.csv"), part=P_)
    cf = ROOT / "analysis/human_vs_ai_table.csv"          # written by analysis/human_vs_ai.py
    out["cmp"] = pd.read_csv(cf) if cf.exists() else pd.DataFrame()
    pf = DATA / "ai_dataset_all_participants.csv"
    out["strat"] = pd.read_csv(pf) if pf.exists() else pd.DataFrame()
    xf = ROOT / "analysis/xai_results.json"
    out["xai"] = json.loads(xf.read_text(encoding="utf-8")) if xf.exists() else {}
    xl = ROOT / "analysis/xai_results_langchain.json"
    out["xai_llama"] = json.loads(xl.read_text(encoding="utf-8")) if xl.exists() else {}
    rl = ROOT / "analysis/results_langchain.json"
    if rl.exists():
        RL = json.loads(rl.read_text(encoding="utf-8")); t = RL["table"][-1]
        out["llama_row"] = dict(sample="Llama agents", n_O=int(t["n_O"]), n_C=int(t["n_C"]), mean_O=t["mean_O"], mean_C=t["mean_C"], diff=t["diff"], d=t.get("cohen_d"), p=t.get("p_ols"))
        out["llama_extra"] = RL["extra"]; out["llama_checks"] = RL["checks"]; out["llama_results"] = RL
    vf = ROOT / "analysis/verdict.json"
    out["verdict"] = json.loads(vf.read_text(encoding="utf-8")) if vf.exists() else {}
    return out


D = load_all()


def grp(df, cond):
    return df[df.Condition == cond]


def kpis():
    H = D["human"]; cw = D["claude"]["wide"]
    k = dict(hg=grp(H, "O").avg.mean() - grp(H, "C").avg.mean(), orig=D["samples"][0]["diff"],
             cg=grp(cw, "O").avg.mean() - grp(cw, "C").avg.mean(), c_share=float(D["claude"]["long"].order.isin([10, 11]).mean()),
             h_sd=float(np.mean([grp(H, c).avg.std() for c in "OC"])), c_sd=float(np.mean([grp(cw, c).avg.std() for c in "OC"])),
             lg=None, l_share=None, fg=None)
    if "claude_f" in D:
        fw = D["claude_f"]["wide"]; k["fg"] = grp(fw, "O").avg.mean() - grp(fw, "C").avg.mean()
    if "llama" in D:
        lw = D["llama"]["wide"]
        k["lg"] = grp(lw, "O").avg.mean() - grp(lw, "C").avg.mean(); k["l_share"] = float(D["llama"]["long"].order.isin([10, 11]).mean())
    return k


K = kpis()


def cmp_row(sample):
    """One row of analysis/human_vs_ai_table.csv (None if the table or the row is missing)."""
    t = D.get("cmp", pd.DataFrame())
    if not len(t) or sample not in set(t["sample"]):
        return None
    return t[t["sample"] == sample].iloc[0]


def pfmt(p):
    return "–" if p is None or pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


KEYNAME = {"faithful": "human-like feedback"}     # display name of the faithful run


def n_ai_agents():
    t = D.get("cmp", pd.DataFrame())
    return int(((t.source != "Human").sum()) * 50) if len(t) else 400


@st.cache_data
def reason_shares():
    """Codes the agents' end-of-game strategy sentences with the keyword rules in analysis/reason_coding.py."""
    S = D.get("strat", pd.DataFrame())
    if not len(S) or "strategy" not in S:
        return None
    try:
        sys.path.insert(0, str(ROOT / "analysis"))
        import reason_coding as rc
    except Exception:
        return None
    C = pd.DataFrame([rc.code(t) for t in S["strategy"]])
    X = pd.concat([S[["model", "wording", "condition"]].reset_index(drop=True), C], axis=1)
    cl = X[(X.model.str.startswith("Claude")) & (X.wording.isin(["base", "faithful"]))]
    return dict(n_all=len(X), ot_all=int(X["order-time payment"].sum()),
                cl_n=len(cl), cl_avg=cl["average demand (10.5 / expected)"].mean(), cl_left=cl["leftover / penalty risk"].mean(),
                lost_O=cl[cl.condition == "O"]["lost sales / running short"].mean(), lost_C=cl[cl.condition == "C"]["lost sales / running short"].mean(),
                n_O=int((cl.condition == "O").sum()), n_C=int((cl.condition == "C").sum()))


# ---- prompt-wording robustness data (both models, four wordings, 50 agents each)
WORDINGS = [("base", "Base (study wording)", "The study's instructions, word for word (our main run)"),
            ("faithful", "Base + human-like feedback", "Study wording; no daily net-change line and the study's C wording ('a profit'); Claude only"),
            ("v1", "Wording v1", "Customer story: customers pay you, unserved customers leave (C text supplied by the team, O text drafted by us)"),
            ("v2", "Wording v2", "Adds bankruptcy and FAILURE warnings and 'you cannot over order' (C text supplied by the team, O text drafted by us)"),
            ("v3", "Wording v3", "Base wording plus a token-spend incentive instead of the bonus paragraph")]
MODELS = [("claude", "Claude", "Claude Haiku 5.5", ""), ("llama", "Llama", "Llama 3.1", "_langchain")]   # key, chart group, name, file tag


def _row_from_results(path):
    R = json.loads(path.read_text(encoding="utf-8"))
    t = [r for r in R["table"] if str(r["sample"]).startswith("AI agents (all")]
    t = t[-1] if t else R["table"][-1]
    sh = R["extra"].get("share_orders_equal_10_or_11", {})
    share = float(np.mean([sh["O"], sh["C"]])) if "O" in sh and "C" in sh else None
    c = R["checks"]
    return dict(mean_O=t["mean_O"], mean_C=t["mean_C"], diff=t["diff"], p=t.get("p_cluster", t.get("p_ols")), d=t.get("cohen_d"), lo=t.get("ci_lo"), hi=t.get("ci_hi"),
                n=int(t["n_O"] + t["n_C"]), fail=int(c["comprehension_fail"]), fail_profit=int(c.get("fail_profit_check", c["comprehension_fail"])),
                fail_range=int(c.get("fail_range_check", 0)), share=share)


@st.cache_data
def load_wording():
    """{model key: [row per wording]} from analysis/human_vs_ai_table.csv (original exclusion rule: dice-range failures
    dropped) plus failure counts from analysis/results_summary.csv. Falls back to results{tag}.json if the table is missing."""
    out = {}
    t = D.get("cmp", pd.DataFrame())
    sf = ROOT / "analysis/results_summary.csv"
    summ = pd.read_csv(sf) if sf.exists() else pd.DataFrame()
    for mkey, group, name, mtag in MODELS:
        rows = []
        for key, label, desc in WORDINGS:
            r = cmp_row(f"{name} – {key}") if len(t) else None
            if r is not None:
                fails = summ[(summ.model == name) & (summ.wording == key) & (summ.subset == "all agents")] if len(summ) else pd.DataFrame()
                f = fails.iloc[0] if len(fails) else None
                rows.append(dict(key=key, label=label, desc=desc, model=name, group=group, mean_O=r.mean_O, mean_C=r.mean_C, diff=r.gap,
                                 p=r.p_gap, lo=r.ci_low, hi=r.ci_high, n=int(r.n_O + r.n_C), share=r.pct_10_11 / 100,
                                 fail_profit=int(f.profit_check_fails) if f is not None else 0, fail_range=int(f.range_check_fails) if f is not None else 0,
                                 total=int(f.agents_total) if f is not None else int(r.n_O + r.n_C), oxai=r.OxAI, oxai_p=r.OxAI_p))
            elif not len(t):
                path = ROOT / f"analysis/results{mtag}{'' if key == 'base' else '_' + key}.json"
                if path.exists():
                    row = _row_from_results(path); row.update(key=key, label=label, desc=desc, model=name, group=group, total=row["n"])
                    rows.append(row)
        out[mkey] = rows
    return out


# ----------------------------------------------------------------- charts
def dumbbell(rows):
    """Mean order under O and C per group. Click O or C in the legend to highlight one scheme."""
    df = pd.DataFrame([r for r in rows if r["O"] is not None and r["C"] is not None])
    df["gap"] = (df.O - df.C).round(2)
    order = list(df.label)
    long = pd.concat([df[["label", "C", "gap"]].rename(columns={"C": "mean"}).assign(scheme="C"),
                      df[["label", "O", "gap"]].rename(columns={"O": "mean"}).assign(scheme="O")])
    XS = alt.Scale(domain=[8.5, 14])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False))
    pick = alt.selection_point(fields=["scheme"], bind="legend")
    band = alt.Chart(pd.DataFrame({"a": [10], "b": [11]})).mark_rect(opacity=0.13, color=OPT).encode(x=alt.X("a:Q", scale=XS), x2="b:Q", tooltip=alt.value(None))
    rule = alt.Chart(df).mark_rule(color="#5A6B7A", strokeWidth=2).encode(y=y, x=alt.X("C:Q", scale=XS, title="Mean order (units)", axis=alt.Axis(values=[9, 10, 11, 12, 13, 14], format="d")), x2="O:Q",
                                                                           tooltip=["label", alt.Tooltip("gap:Q", title="O − C", format="+.2f")])
    pts = alt.Chart(long).mark_circle(size=190).encode(
        y=y, x=alt.X("mean:Q", scale=XS),
        color=alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL]), legend=alt.Legend(title="Scheme (click to highlight)", orient="top")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.15)),
        tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("scheme:N", title="Scheme"), alt.Tooltip("mean:Q", title="Mean order", format=".2f"), alt.Tooltip("gap:Q", title="O − C", format="+.2f")]
    ).add_params(pick)
    return (band + rule + pts).properties(height=alt.Step(44))   # fixed row height, so every row keeps its label


def rows_for_dumbbell():
    S = D["samples"]
    rows = [dict(label="Original (Duke)", O=S[0]["mean_O"], C=S[0]["mean_C"])]
    for s in S[1:4]:
        rows.append(dict(label=s["sample"], O=s["mean_O"], C=s["mean_C"]))
    if "claude" in D:
        w = D["claude"]["wide"]; rows.append(dict(label="Claude agents", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    if "claude_f" in D:
        w = D["claude_f"]["wide"]; rows.append(dict(label="Claude, human-like feedback", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    if "llama" in D:
        w = D["llama"]["wide"]; rows.append(dict(label="Llama agents", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    return rows


def wording_chart(rows):
    """Dot = O minus C gap; line = 95% confidence interval. Click a group in the legend to highlight it."""
    d = pd.DataFrame(rows)
    order = list(d.label)
    pal = alt.Scale(domain=["Humans", "Claude", "Llama"], range=[ink(), OPT, "#8A97A6"])
    pick = alt.selection_point(fields=["kind"], bind="legend")
    col = alt.Color("kind:N", scale=pal, legend=alt.Legend(title="Group (click to highlight)", orient="top"))
    op = alt.condition(pick, alt.value(1), alt.value(0.12))
    XS = alt.Scale(domain=[-1.2, 2.8])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13, labelOverlap=False))
    tip = [alt.Tooltip("label:N", title="Run"), alt.Tooltip("diff:Q", title="O − C", format="+.2f"),
           alt.Tooltip("lo:Q", title="95% CI low", format="+.2f"), alt.Tooltip("hi:Q", title="95% CI high", format="+.2f")]
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#9AA8B5", strokeWidth=2).encode(x=alt.X("x:Q", scale=XS))
    pts = alt.Chart(d).mark_circle(size=210).encode(y=y, x=alt.X("diff:Q", scale=XS, title="Mean order, O minus C (units). Right of 0 = people-like direction"),
                                                   color=col, opacity=op, tooltip=tip).add_params(pick)
    layers = [zero]
    ci = d.dropna(subset=["lo", "hi"])
    if len(ci):
        layers.append(alt.Chart(ci).mark_rule(strokeWidth=3).encode(y=y, x=alt.X("lo:Q", scale=XS), x2="hi:Q", color=col, opacity=op, tooltip=tip))
    layers.append(pts)
    return alt.layer(*layers).properties(height=alt.Step(40))


# ----------------------------------------------------------------- tables


def behaviour_table():
    rows = []

    def add(name, W, fail):
        sh = float(W[OC].isin([10, 11]).to_numpy().mean())
        between = float(np.mean([grp(W, c).avg.std() for c in "OC"]))
        within = float(W[OC].std(axis=1).mean())
        rows.append({"Group": name, "Orders at 10–11": f"{sh:.0%}", "Spread between participants": f"{between:.2f}",
                     "Spread within a participant": f"{within:.2f}", "Failed comprehension check": fail})
    add("Humans (3 replications)", D["human"], "excluded from the data")
    if "claude" in D:
        add("Claude agents", D["claude"]["wide"], f"{D['checks']['comprehension_fail']} of 50")
    if "claude_f" in D:
        pf_ = D["claude_f"]["part"]; add("Claude, human-like feedback", D["claude_f"]["wide"], f"{int(pf_.comprehension_fail.sum())} of {len(pf_)}")
    if "llama" in D:
        add("Llama agents", D["llama"]["wide"], f"{D['llama_checks']['comprehension_fail']} of 50")
    return pd.DataFrame(rows)



GROUPS = [("human", "Humans (3 replications)"), ("claude", "Claude"), ("claude_f", "Claude, human-like feedback"), ("llama", "Llama")]


@st.cache_data
def by_day():
    """Mean order per day and scheme for humans and each AI run (AI: original exclusion rule)."""
    rows = []
    for key, name in GROUPS:
        W = D["human"] if key == "human" else (D[key]["wide"] if key in D else None)
        if W is None:
            continue
        for cond in "OC":
            m = grp(W, cond)[OC].mean()
            rows += [dict(group=name, scheme=cond, day=i + 1, mean=float(v)) for i, v in enumerate(m.values)]
    return pd.DataFrame(rows)


def day_chart(df):
    """Small multiples: average order by day, O vs C. Hover a day to read the values."""
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["day"], empty=False, clear="pointerout")
    # slider drawn by the chart itself, so the lines update live while dragging (no wait for release)
    days = alt.param(name="days", value=25, bind=alt.binding_range(min=1, max=25, step=1, name="Days shown: "))
    base = alt.Chart().encode(x=alt.X("day:Q", title="Day", scale=alt.Scale(domain=[1, 25]), axis=alt.Axis(values=[1, 5, 10, 15, 20, 25])))
    color = alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL]), legend=alt.Legend(title="Scheme", orient="top"))
    YS = alt.Scale(domain=[6, 17], zero=False, clamp=True)
    band = base.transform_calculate(lo="10", hi="11").mark_area(opacity=0.14, color=OPT).encode(y=alt.Y("lo:Q", scale=YS), y2="hi:Q")
    shown = base.transform_filter(alt.datum.day <= days)
    lines = shown.mark_line(strokeWidth=2.5).encode(y=alt.Y("mean:Q", title="Average order", scale=YS), color=color)
    pts = shown.mark_circle(size=70).encode(y="mean:Q", color=color, opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                                           tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("day:Q", title="Day"), alt.Tooltip("scheme:N", title="Scheme"),
                                                    alt.Tooltip("mean:Q", title="Average order", format=".2f")]).add_params(hover)
    rule = shown.mark_rule(color="#9AA8B5").encode(opacity=alt.condition(hover, alt.value(0.8), alt.value(0)))
    return alt.layer(band, lines, rule, pts, data=df).properties(width=330, height=210).facet(
        facet=alt.Facet("group:N", title=None, sort=[n for _, n in GROUPS], header=alt.Header(labelFontSize=14, labelFontWeight="bold")), columns=2).add_params(days)


@st.cache_data
def reason_table():
    """Every agent's end-of-game strategy sentence, coded with the keyword rules in analysis/reason_coding.py."""
    S = D.get("strat", pd.DataFrame())
    if not len(S) or "strategy" not in S:
        return None, []
    try:
        sys.path.insert(0, str(ROOT / "analysis"))
        import reason_coding as rc
    except Exception:
        return None, []
    C = pd.DataFrame([rc.code(t) for t in S["strategy"]])
    X = pd.concat([S[["model", "wording", "condition", "strategy"]].reset_index(drop=True), C], axis=1)
    return X, list(rc.CATS)


REASON_NAMES = {"average demand (10.5 / expected)": "Average demand (10.5)", "order-time payment": "Money at order time",
                "leftover / penalty risk": "Leftover / penalty risk", "lost sales / running short": "Lost sales",
                "balance / trade-off": "Trade-off", "recent demand / adjusting": "Recent demand", "explicit profit calculation": "Profit calculation"}


def reasons_chart(t):
    pick = alt.selection_point(fields=["reason", "scheme"], name="pick")
    order = list(t.groupby("reason")["share"].mean().sort_values(ascending=False).index)
    return alt.Chart(t).mark_bar(cornerRadiusEnd=4, height=11).encode(
        y=alt.Y("reason:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        yOffset=alt.YOffset("scheme:N", sort=["O", "C"]),
        x=alt.X("share:Q", title="Share of agents mentioning it", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL]), legend=alt.Legend(title="Scheme", orient="top")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("reason:N", title="Reason"), alt.Tooltip("scheme:N", title="Scheme"), alt.Tooltip("share:Q", title="Share", format=".0%"),
                 alt.Tooltip("n:Q", title="Agents")]
    ).add_params(pick).properties(height=alt.Step(26))


def shap_chart(X, XL):
    rows = []
    for who, src, key in [("Humans", X, "shap_human"), ("Claude", X, "shap_ai_common"), ("Llama", XL, "shap_ai_common")]:
        if src and key in src:
            v = src[key]["mean_abs_shap"]; tot = sum(v.values())
            for f, name in [("cond_O", "Payment scheme"), ("prev_order", "Own last order"), ("day", "Day of the game")]:
                rows.append(dict(who=who, feature=name, share=v.get(f, 0) / tot))
    d = pd.DataFrame(rows)
    pick = alt.selection_point(fields=["feature"], bind="legend")
    return alt.Chart(d).mark_bar(height=26).encode(
        y=alt.Y("who:N", sort=["Humans", "Claude", "Llama"], title=None),
        x=alt.X("share:Q", stack="normalize", title="Share of what explains the next order (Shapley values)", axis=alt.Axis(format="%")),
        color=alt.Color("feature:N", sort=["Payment scheme", "Own last order", "Day of the game"],
                        scale=alt.Scale(domain=["Payment scheme", "Own last order", "Day of the game"], range=[OPT, "#8A97A6", "#C9D1DA"]),
                        legend=alt.Legend(title="Driver (click to highlight)", orient="top")),
        order=alt.Order("feature:N", sort="descending"),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.2)),
        tooltip=[alt.Tooltip("who:N", title="Who"), alt.Tooltip("feature:N", title="Driver"), alt.Tooltip("share:Q", title="Share", format=".0%")]
    ).add_params(pick).properties(height=alt.Step(40))


def pick_one(label, options, default, key):
    """Single choice as buttons; falls back to the default when nothing is selected."""
    v = st.segmented_control(label, options, default=default, key=key)
    return v if v else default


def pick_many(label, options, default, key):
    v = st.pills(label, options, selection_mode="multi", default=default, key=key)
    return v or []



# ----------------------------------------------------------------- presenter steps
def step_business():
    st.markdown("<div class='lead'><b>Research question:</b> to what extent can AI agents reproduce human decision making and the behavioural bias found in the original experiment?</div>", unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    tile(a, "<b>The business problem</b><br>A shop must order stock <i>before</i> it knows demand. Order too little and it loses sales. Order too much and stock is wasted. This is the classic <b>newsvendor</b> inventory problem.")
    tile(b, "<b>Why it matters</b><br>Supply chains pay for stock in different ways: the buyer pays up front, or a customer or supplier advances the money. If the profit is the same, decisions should be the same. If they are not, wording is costing money.")
    tile(c, "<b>The bias studied</b><br><b>Mental accounting.</b> People treat money paid <i>before</i> demand is known differently from money paid <i>after</i>, even when the totals are identical.")
    st.write("")
    m = st.columns(4)
    m[0].metric("AI models", 2 if "llama" in D else 1); m[1].metric("AI agents", f"{n_ai_agents()}", help="2 models × 4 wordings × 50 agents, plus 50 Claude agents with human-like feedback; the main comparison uses the base wording")
    m[2].metric("Rounds each", 25); m[3].metric("Human samples", 4)
    st.caption("Replicating Chen, Kök & Tong (2013), Management Science 59(2), 436–451.")


def step_original():
    a, b = st.columns(2)
    a.markdown("<div class='card-o'><b>O · Own financing</b><br>You pay 1 play dollar per unit when you order. You receive 2 play dollars per unit sold.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card-c'><b>C · Customer financing</b><br>You receive 1 play dollar per unit when you order. You pay 2 play dollars per unit left unsold.</div>", unsafe_allow_html=True)
    st.markdown("<div class='cap' style='margin-top:12px'><b>q</b> = units ordered &nbsp;·&nbsp; <b>D</b> = demand (the dice total) &nbsp;·&nbsp; "
                "<b>Δ</b> = the day's profit in play dollars</div>", unsafe_allow_html=True)
    st.markdown("""<div class='fgrid'>
  <div class='fbox'><div class='fhead o'>① Profit under O</div>
    <div class='fmath'>Δ<sub>O</sub> = 2 · min(q, D) − q</div>
    <div class='fnote'><b>$2 × units sold</b>, minus <b>$1 × units ordered</b> (paid up front).</div></div>
  <div class='fbox'><div class='fhead c'>② Profit under C</div>
    <div class='fmath'>Δ<sub>C</sub> = q − 2 · max(q − D, 0)</div>
    <div class='fnote'><b>$1 × units ordered</b> (received up front), minus <b>$2 × units left over</b>.</div></div>
  <div class='fbox'><div class='fhead e'>③ Why they are always equal</div>
    <div class='fmath'>Δ<sub>O</sub> = Δ<sub>C</sub></div>
    <div class='fnote'><b>Units sold = ordered − leftover.</b> Put that into ① and you get ②, for every order and every dice roll.</div></div>
</div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    tile(c1, "<b>How the experiment worked</b><br>99 Duke students played a widget shop for <b>25 days</b>, starting with <b>100 play dollars</b>, using poker chips and real dice. Each day: choose an order, then <b>three dice</b> are rolled and their sum is demand (3 to 18, average 10.5). Leftovers are discarded. A real bonus depended on the final play dollars.")
    tile(c2, "<b>Same profit, different timing</b><br>For every order and every dice roll the schemes pay exactly the same, and the best order is <b>10 or 11</b>. Each person played one scheme. The original also had a neutral scheme <b>S</b> (all money moves after demand); the replications, and we, compare O with C.")


def step_human():
    S = D["samples"]
    _w(st.altair_chart, polish(dumbbell(rows_for_dumbbell()[:4])))
    m = st.columns(5)
    m[0].metric("Original O − C", f"{S[0]['diff']:+.2f}")
    for i in range(3):
        m[i + 1].metric(S[i + 1]["sample"], f"{S[i + 1]['diff']:+.2f}")
    m[4].metric("Spread", f"{K['h_sd']:.1f}", help="Standard deviation of participants' average orders")
    st.caption(f"Shaded band: the best order, 10 or 11. O − C in units of order quantity; the three replications pool to {K['hg']:+.2f}. Spread = standard deviation of participants' average orders.")
    st.markdown("**Why? Prospective accounting**, a form of mental accounting: people give less weight to money that changes hands *at order time*, before demand is known. "
                "Under **O** that money is a cost, so ordering feels cheaper and people **order more**. Under **C** it is income, so ordering feels less rewarding and people **order less**. "
                "In the original, the neutral scheme S (no money at order time) sat at the optimum (10.57), and the loss-aversion and time-discounting explanations predicted patterns the data rejected. "
                "A fully rational person would ignore the timing.")


def step_demo():
    st.markdown("<div class='lead'>One example day: order <b>11</b> units, the dice show 3, 2 and 4, so demand is <b>9</b>.</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    tile(a, "<b>O · Own financing</b><br>At order time: you <b>pay</b> 11 × 1 = <b>−11</b><br>After demand: 9 sold × 2 = <b>+18</b><br>2 left over, discarded<br><b>Day's result: +7</b>")
    tile(b, "<b>C · Customer financing</b><br>At order time: you <b>receive</b> 11 × 1 = <b>+11</b><br>After demand: 9 sold, nothing received<br>2 left over × 2 penalty = <b>−4</b><br><b>Day's result: +7</b>")
    st.write("")
    st.markdown("<div class='lead'>Same order, same dice, same <b>+7</b>. Only the timing and the description of the money differ. "
                "This is exactly the task the human participants had, and the task we gave the AI agents.</div>", unsafe_allow_html=True)


def step_design():
    a, b, c, d = st.columns(4)
    tile(a, "<b>1 Instructions only</b><br>The study's O or C text, word for word, plus the bonus rules. No age, gender or other background.")
    tile(b, "<b>2 Game engine</b><br>Checks every order (whole number 3–18), rolls real dice <i>after</i> the order, applies the payoff rule.")
    tile(c, "<b>3 Memory</b><br>Each agent sees all of its earlier days (orders, dice, results).")
    tile(d, "<b>4 Two comprehension checks</b><br>Profit for order 10 and demand 7 (answer +4) and the lowest and highest possible demand (3 and 18).")
    st.write("")
    a, b = st.columns(2)
    tile(a, "<b>Two models, four wordings, 50 agents each run</b><br>25 per scheme, the same size as the original sample.<br>• <b>Claude Haiku 5.5</b> through the Claude API and LangGraph<br>• <b>Llama 3.1 8B</b> through LangGraph on a free local Ollama<br>• Study wording plus three alternative wordings<br>• Plus a Claude run with <b>human-like feedback</b> (see right)")
    tile(b, "<b>Kept the same:</b> payoffs, 25 rounds, dice, instructions word for word.<br><b>Changed, and why:</b> a structured answer instead of a screen (agents cannot click); orders limited to 3–18, as in the online replication; no real money.<br><b>Checked:</b> our main runs showed agents a daily net 'change in play dollars' and called C's money 'an advance'. Humans saw the two payments separately, so we re-ran Claude without both (the <i>human-like feedback</i> run).")
    st.code("decide (order 3–18)  →  engine rolls dice and applies payoff  →  repeat for 25 days  →  debrief: follow-up question, two comprehension checks, strategy sentence", language="text")


def step_compare():
    view = pick_one("View", ["Averages", "Day by day"], "Averages", key="cmp_view")
    if view == "Averages":
        rows = rows_for_dumbbell()
        labels = [r["label"] for r in rows]
        show = pick_many("Show", labels, labels, key="cmp_show")
        sel = [r for r in rows if r["label"] in show]
        if sel:
            _w(st.altair_chart, polish(dumbbell(sel)))
        else:
            st.info("Pick at least one group above.")
        st.caption("Each row: average order under O and C, the line between them is the gap. Shaded band: the best order, 10 or 11. "
                   "Hover a dot for its values; click O or C in the legend to highlight one scheme.")
    else:
        df = by_day()
        names = [n for _, n in GROUPS if n in set(df.group)]
        show = pick_many("Show", names, names, key="day_show")
        d = df[df.group.isin(show)]
        if len(d):
            st.altair_chart(polish(day_chart(d)))
        else:
            st.info("Pick at least one group above.")
        st.caption("Average order each day under O and C. People keep a gap all game; Claude settles on 10–11 within a few days; Llama drifts upward. "
                   "Drag the **Days shown** slider under the chart to replay the game; hover a day to read the values. Shaded band: the best order.")
    m = st.columns(5)
    m[0].metric("Humans", f"{K['hg']:+.2f}", help="Pooled across the three replications")
    m[1].metric("Claude", f"{K['cg']:+.2f}")
    m[2].metric("Feedback run", "–" if K["fg"] is None else f"{K['fg']:+.2f}")
    m[3].metric("Llama", "–" if K["lg"] is None else f"{K['lg']:+.2f}", help="Agents that got the dice range wrong are excluded, as in the original study")
    m[4].metric("At 10–11", f"{K['c_share']:.0%}")
    st.caption("Tiles: the O − C gap in average order (units); feedback run = Claude with human-like feedback; the last tile is the share of Claude's orders at 10 or 11.")
    rc_, rl_, rf_ = cmp_row("Claude Haiku 5.5 – base"), cmp_row("Llama 3.1 – base"), cmp_row("Claude Haiku 5.5 – faithful")
    test = ""
    if rc_ is not None and rl_ is not None:
        test = (f" A pooled regression with the human data confirms the agents' gap is significantly <b>smaller</b> than the human gap: "
                f"Claude by {abs(rc_.OxAI):.2f} units (p {pfmt(rc_.OxAI_p)}), Llama by {abs(rl_.OxAI):.2f} (p {pfmt(rl_.OxAI_p)})"
                + (f", and Claude with human-like feedback by {abs(rf_.OxAI):.2f} (p {pfmt(rf_.OxAI_p)})." if rf_ is not None else "."))
    st.markdown(f"<div class='lead' style='font-size:1.05rem'><b>With the study's wording, neither model shows the human gap.</b> People order about {K['hg']:.2f} units more under O; Claude's gap is {K['cg']:+.2f} and Llama's {'–' if K['lg'] is None else format(K['lg'], '+.2f')}.{test} Claude orders almost only 10 or 11; Llama over-orders and drifts upward.</div>", unsafe_allow_html=True)
    with st.expander("Behaviour table: orders at 10–11, spread, comprehension checks"):
        st.dataframe(behaviour_table(), hide_index=True)


def step_wording():
    rows = wording_rows_for_chart()
    wl = [label for _, label, _ in WORDINGS if any(r["wording"] == label for r in rows)]
    c1, c2 = st.columns([2, 1])
    with c1:
        models = pick_many("Models", ["Claude", "Llama"], ["Claude", "Llama"], key="w_models")
    with c2:
        st.write("")
        humans = st.toggle("Show humans", value=True, key="w_humans")
    words = pick_many("Wordings", wl, wl, key="w_words")
    sel = [r for r in rows if (r["kind"] == "Humans" and humans) or (r["kind"] in models and r["wording"] in words)]
    if sel:
        _w(st.altair_chart, polish(wording_chart(sel)))
    else:
        st.info("Pick at least one model and wording above.")
    st.caption("Dot = O − C gap, line = 95% confidence interval; 0 = no framing effect. Hover for exact values; click a group in the legend to highlight it.")
    W = load_wording()
    cg = ", ".join(f"{KEYNAME.get(r['key'], r['key'])} {r['diff']:+.2f}" for r in W.get("claude", [])); lg = ", ".join(f"{KEYNAME.get(r['key'], r['key'])} {r['diff']:+.2f}" for r in W.get("llama", []))
    st.markdown(f"**The gap moves with the wording, but never reaches the human gap.** O − C for Claude: {cg}. Llama: {lg}. We kept the study's wording for the main results and report the others openly.")


def step_why():
    X, XL = D.get("xai", {}), D.get("xai_llama", {})
    if X:
        st.markdown("**What drives each next order?** A machine-learning model of every decision, explained with Shapley values.")
        _w(st.altair_chart, polish(shap_chart(X, XL)))
        st.caption("Share of the explanation for each driver. People and Claude both mostly follow their own last order; the payment scheme matters for people, "
                   "hardly at all for the agents. Hover for values; click a driver in the legend to highlight it.")
    T, cats = reason_table()
    if T is not None:
        st.markdown("**In their own words:** each agent's end-of-game sentence on how it decided, sorted into reasons.")
        c1, c2 = st.columns([1, 2])
        with c1:
            model = pick_one("Model", ["Claude", "Llama"], "Claude", key="r_model")
        present = [w for w in ["base", "faithful", "v1", "v2", "v3"] if ((T.model.str.startswith(model)) & (T.wording == w)).any()]
        names = [KEYNAME.get(w, w) for w in present]
        default = [n for n in names if n in ("base", "human-like feedback")]
        with c2:
            words = pick_many("Wordings", names, default, key=f"r_words_{model}")
        inv = {KEYNAME.get(w, w): w for w in present}
        sub = T[T.model.str.startswith(model) & T.wording.isin([inv[w] for w in words])]
        if len(sub):
            rows = []
            for cat in cats:
                for sch in "OC":
                    g = sub[sub.condition == sch]
                    if len(g):
                        rows.append(dict(reason=REASON_NAMES.get(cat, cat), scheme=sch, share=float(g[cat].mean()), n=int(g[cat].sum()), cat=cat))
            tbl = pd.DataFrame(rows)
            try:
                ev = st.altair_chart(polish(reasons_chart(tbl)), on_select="rerun", key=f"r_chart_{model}", width="stretch")
            except TypeError:
                ev = st.altair_chart(polish(reasons_chart(tbl)), on_select="rerun", key=f"r_chart_{model}", use_container_width=True)
            picked = []
            try:
                picked = ev["selection"]["pick"] or []
            except Exception:
                picked = []
            if picked and "reason" in picked[0] and "scheme" in picked[0]:
                reason, sch = picked[0]["reason"], picked[0]["scheme"]
                cat = next(c for c in cats if REASON_NAMES.get(c, c) == reason)
                ex = sub[(sub.condition == sch) & (sub[cat] == 1)]["strategy"].head(3).tolist()
                st.markdown(f"**{reason}**, scheme {sch}: what the agents wrote" if ex else f"No {sch} agent mentioned {reason.lower()}.")
                for e in ex:
                    st.markdown(f"> {e}")
            else:
                st.caption("Click a bar to read what those agents actually wrote.")
        rs = reason_shares()
        if rs:
            st.markdown(f"Only **{rs['ot_all']} of {rs['n_all']}** agents ever mention the money that moves at order time, the thing the theory is about. "
                        f"The scheme shapes what Claude says (lost sales: {rs['lost_O']:.0%} of O agents vs {rs['lost_C']:.0%} of C agents) but not what it orders.")


def step_conclusion():
    a, b, c = st.columns(3)
    tile(a, "<b>What we learned</b><br>Neither model reproduced the human framing gap; the agents' gap is significantly smaller than people's, also with human-like feedback. Some alternative wordings gave Claude a small gap in the human direction. The agents order around average demand by rule of thumb and almost never think about the money that moves at order time; Llama over-orders and drifts.")
    tile(b, "<b>Limits of our simulation</b><br>• One model per run, so agents are not independent<br>• No real stakes<br>• Results depend on the model and the wording<br>• The same prompt gave different day-1 orders on different run dates (model updates or sampling)<br>• The profit check is our addition; few agents pass it")
    tile(c, "<b>Caution for researchers</b><br>Use AI to screen ideas quickly, then confirm with people. Always report the model, prompt and settings, and test whether the result survives a change in wording.")
    st.write("")
    st.subheader("Questions?")


def wording_rows_for_chart():
    ro, rp = cmp_row("Original (Duke)"), cmp_row("Human – 3 replications pooled")
    rows = [dict(label="Humans, original study", kind="Humans", wording=None, diff=float(K["orig"]), lo=None if ro is None else ro.ci_low, hi=None if ro is None else ro.ci_high),
            dict(label="Humans, replications (pooled)", kind="Humans", wording=None, diff=float(K["hg"]), lo=None if rp is None else rp.ci_low, hi=None if rp is None else rp.ci_high)]
    for mkey, rs in load_wording().items():
        for r in rs:
            rows.append(dict(label=f"{r['label']} · {r['group']}", kind=r["group"], wording=r["label"], diff=float(r["diff"]), lo=r.get("lo"), hi=r.get("hi")))
    return rows


# key, title, seconds at 10 minutes, rubric key, takeaway, function
STEPS = [
    ("business", "The business problem", 45, "bg", "Same money, different words. Do AI agents react like people?", step_business),
    ("original", "The original experiment", 75, "orig", "Two payment wordings, identical profit.", step_original),
    ("human", "What people did, and why", 60, "orig", "People order more when they pay up front.", step_human),
    ("demo", "Same task, same profit", 30, "qual", "Same order, same dice, same profit; only the timing differs.", step_demo),
    ("design", "How we built the AI participants", 90, "ai", "Same instructions and engine, no personas, 50 agents per run.", step_design),
    ("compare", "Results: AI versus people", 160, "cmp", "In the study wording, neither model shows the human gap.", step_compare),
    ("wording", "Does the prompt wording matter?", 40, "cmp", "The gap changes with the wording for both models.", step_wording),
    ("why", "What explains the difference?", 40, "cmp", "Agents order around average demand and ignore the payment timing.", step_why),
    ("conclude", "Conclusions and limits", 60, "conc", "Weak or no framing effect in AI agents.", step_conclusion),
]


def present_tab():
    ss = st.session_state
    ss.setdefault("step", 0)
    cur = ss.step
    cols = st.columns(len(STEPS))
    for i, s in enumerate(STEPS):
        if btn(cols[i], str(i + 1), key=f"jump{i}", help=s[1], type="primary" if i == cur else "secondary"):
            ss.step = i; st.rerun()
    st.progress((cur + 1) / len(STEPS))
    key, title, _, rub, take, fn = STEPS[cur]
    with st.container(border=True):
        st.markdown(f"<div class='kick'>Step {cur + 1} of {len(STEPS)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big'>{title}</div><div class='take'>{take}</div>", unsafe_allow_html=True)
        fn()
    a, b, _ = st.columns([1.5, 1.5, 7])
    if btn(a, "← Back", disabled=cur == 0):
        ss.step -= 1; st.rerun()
    if btn(b, "Next →", type="primary", disabled=cur == len(STEPS) - 1):
        ss.step += 1; st.rerun()


# ----------------------------------------------------------------- game


# ----------------------------------------------------------------- AI system


# ----------------------------------------------------------------- AI replay


# ----------------------------------------------------------------- results


# ----------------------------------------------------------------- prompt wording


# ----------------------------------------------------------------- Q&A


# ----------------------------------------------------------------- checklist


# ----------------------------------------------------------------- layout
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    _, tcol = st.columns([4, 1])
    tcol.toggle("Dark mode", key="dark")
    if is_dark():
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.markdown("""<div class='hero'><h1>Payment Framing Lab</h1>
      <p>Same profit, different wording. Can AI agents reproduce the bias people show?</p>
      <span class='chip o'>O · OWN FINANCING</span><span class='chip c'>C · CUSTOMER FINANCING</span><span class='chip n'>Chen, Kök &amp; Tong (2013) replication with AI agents</span></div>""", unsafe_allow_html=True)
    present_tab()
    st.markdown("<div class='footer'>MBAN 5640 · Group Project 3 · Replication of Chen, Kök &amp; Tong (2013) with AI agents</div>", unsafe_allow_html=True)


main()
