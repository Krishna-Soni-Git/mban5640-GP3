"""Payment Framing Lab (Streamlit): a presentation dashboard for MBAN 5640 Group Project 3.

Replicates Chen, Kok & Tong (2013) with AI agents. Tabs:
  Present         guided 10 or 12 minute talk, one step per grading criterion, timer, optional speaker notes
  Game            the newsvendor game (presenter plays a few days to show what participants saw)
  AI system       how the agents were made, original vs replication settings, the prompt, validation
  AI replay       watch any saved agent play its 25 days
  Results         human vs AI comparison with statistics
  Prompt wording  robustness check: does the result survive different instruction wording?
  Q&A, Checklist  presenter-only tabs (add ?presenter=1 to the address, which also shows the timer, speaker notes and grading badges)

Run:  python -m streamlit run streamlit_app/app.py
Reads the project's own files (data/, analysis/, prompts/), so re-running the analysis updates the app.
"""
import json, re, secrets, sys, time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OC = [f"order{i}" for i in range(1, 26)]
FACES = ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"]
O_COL, C_COL, OPT = "#C8681A", "#17808A", "#6B5BB5"
RNG = secrets.SystemRandom()
# Class view by default. Add ?presenter=1 to the address to show the timer, speaker notes, grading badges, Q&A and Checklist tabs.
PRESENTER = str(st.query_params.get("presenter", "")).lower() in ("1", "true", "yes")

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
</style>
"""


def polish(chart):
    """Consistent chart styling."""
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


def card(title, caption=None):
    """Bordered card with a title; returns the container."""
    box = st.container(border=True)
    box.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)
    if caption:
        box.markdown(f"<div class='cap'>{caption}</div>", unsafe_allow_html=True)
    return box


def tile(container, html):
    container.markdown(f"<div class='tile'>{html}</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data
def load_all():
    R = json.loads((ROOT / "analysis/results.json").read_text())
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
    for tag, key in [("", "claude"), ("_langchain", "llama")]:
        wf = DATA / f"ai_agent_wide{tag}.csv"
        if wf.exists():
            W = pd.read_csv(wf); W = W.assign(avg=W[OC].mean(axis=1))
            out[key] = dict(wide=W, long=pd.read_csv(DATA / f"ai_agent_long{tag}.csv"), part=pd.read_csv(DATA / f"ai_agent_participants{tag}.csv"))
    xf = ROOT / "analysis/xai_results.json"
    out["xai"] = json.loads(xf.read_text()) if xf.exists() else {}
    xl = ROOT / "analysis/xai_results_langchain.json"
    out["xai_llama"] = json.loads(xl.read_text()) if xl.exists() else {}
    rl = ROOT / "analysis/results_langchain.json"
    if rl.exists():
        RL = json.loads(rl.read_text()); t = RL["table"][-1]
        out["llama_row"] = dict(sample="Llama agents", n_O=int(t["n_O"]), n_C=int(t["n_C"]), mean_O=t["mean_O"], mean_C=t["mean_C"], diff=t["diff"], d=t.get("cohen_d"), p=t.get("p_ols"))
        out["llama_extra"] = RL["extra"]; out["llama_checks"] = RL["checks"]; out["llama_results"] = RL
    vf = ROOT / "analysis/verdict.json"
    out["verdict"] = json.loads(vf.read_text()) if vf.exists() else {}
    return out


D = load_all()


def grp(df, cond):
    return df[df.Condition == cond]


def kpis():
    H = D["human"]; cw = D["claude"]["wide"]
    k = dict(hg=grp(H, "O").avg.mean() - grp(H, "C").avg.mean(), orig=D["samples"][0]["diff"],
             cg=grp(cw, "O").avg.mean() - grp(cw, "C").avg.mean(), c_share=float(D["claude"]["long"].order.isin([10, 11]).mean()),
             h_sd=float(np.mean([grp(H, c).avg.std() for c in "OC"])), c_sd=float(np.mean([grp(cw, c).avg.std() for c in "OC"])),
             lg=None, l_share=None)
    if "llama" in D:
        lw = D["llama"]["wide"]
        k["lg"] = grp(lw, "O").avg.mean() - grp(lw, "C").avg.mean(); k["l_share"] = float(D["llama"]["long"].order.isin([10, 11]).mean())
    return k


K = kpis()


def pct(x):
    return "–" if x is None else f"{x:.0%}"


# ---- prompt-wording robustness data (both models, four wordings, 50 agents each)
WORDINGS = [("base", "Base (study wording)", "The study's instructions, word for word (our main run)"),
            ("v1", "Wording v1", "Customer story: customers pay you, unserved customers leave (C text supplied by the team, O text drafted by us)"),
            ("v2", "Wording v2", "Adds bankruptcy and FAILURE warnings and 'you cannot over order' (C text supplied by the team, O text drafted by us)"),
            ("v3", "Wording v3", "Base wording plus a token-spend incentive instead of the bonus paragraph")]
MODELS = [("claude", "Claude", "Claude Haiku 5.5", ""), ("llama", "Llama", "Llama 3.1", "_langchain")]   # key, chart group, name, file tag


def _row_from_results(path):
    R = json.loads(path.read_text())
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
    """{model key: [row per wording]} from analysis/results{tag}.json written by analyze.py."""
    out = {}
    for mkey, group, name, mtag in MODELS:
        rows = []
        for key, label, desc in WORDINGS:
            path = ROOT / f"analysis/results{mtag}{'' if key == 'base' else '_' + key}.json"
            if path.exists():
                row = _row_from_results(path)
                row.update(key=key, label=label, desc=desc, model=name, group=group)
                rows.append(row)
        out[mkey] = rows
    return out


@st.cache_data
def load_vs_base():
    f = ROOT / "analysis/results_vs_base.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def claude_gap_row():
    t = [r for r in D["results"]["table"] if str(r["sample"]).startswith("AI agents (all")][-1]
    return dict(diff=t["diff"], lo=t.get("ci_lo"), hi=t.get("ci_hi"))


# ----------------------------------------------------------------- charts
def dumbbell(rows):
    df = pd.DataFrame([r for r in rows if r["O"] is not None and r["C"] is not None])
    df["gap"] = (df.O - df.C).round(2)
    order = list(df.label)
    long = pd.concat([df[["label", "C", "gap"]].rename(columns={"C": "mean"}).assign(scheme="C"),
                      df[["label", "O", "gap"]].rename(columns={"O": "mean"}).assign(scheme="O")])
    XS = alt.Scale(domain=[8.5, 14])
    band = alt.Chart(pd.DataFrame({"a": [10], "b": [11]})).mark_rect(opacity=0.13, color=OPT).encode(x=alt.X("a:Q", scale=XS), x2="b:Q")
    rule = alt.Chart(df).mark_rule(color="#5A6B7A", strokeWidth=2).encode(y=alt.Y("label:N", sort=order, title=None), x=alt.X("C:Q", scale=XS, title="Mean order (units)", axis=alt.Axis(values=[9, 10, 11, 12, 13, 14], format="d")), x2="O:Q")
    pts = alt.Chart(long).mark_circle(size=170, opacity=1).encode(
        y=alt.Y("label:N", sort=order, title=None), x=alt.X("mean:Q", scale=XS),
        color=alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL]), legend=alt.Legend(title="Scheme", orient="top")),
        tooltip=["label", "scheme", alt.Tooltip("mean:Q", format=".2f"), "gap"])
    return (band + rule + pts).properties(height=44 * len(df) + 40)


def by_day_chart(series):
    rows = []
    for name, df, cond in series:
        g = grp(df, cond)[OC].mean()
        for i, v in enumerate(g.values, 1):
            rows.append(dict(series=name, scheme=cond, day=i, mean=float(v)))
    d = pd.DataFrame(rows)
    return alt.Chart(d).mark_line(strokeWidth=2.2, clip=True).encode(
        x=alt.X("day:Q", title="Day", scale=alt.Scale(domain=[1, 25], nice=False), axis=alt.Axis(values=[1, 5, 10, 15, 20, 25])), y=alt.Y("mean:Q", scale=alt.Scale(domain=[8, 15], nice=False), title="Mean order"),
        color=alt.Color("scheme:N", title="Scheme", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL])),
        strokeDash=alt.StrokeDash("series:N", title="Group"), tooltip=["series", "scheme", "day", alt.Tooltip("mean:Q", format=".2f")]).properties(height=280)


def strips(groups):
    rows = [dict(group=g, scheme=c, avg=float(v)) for g, c, vals in groups for v in vals]
    d = pd.DataFrame(rows)
    order = [g for g, _, _ in groups]
    base = alt.Chart(d).encode(x=alt.X("group:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=90, labelFontSize=12)))
    pts = base.mark_circle(size=40, opacity=0.55).encode(y=alt.Y("avg:Q", scale=alt.Scale(domain=[5, 17]), title="Participant average order"),
        color=alt.Color("scheme:N", title="Scheme", legend=None, scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL])), xOffset="jitter:Q") \
        .transform_calculate(jitter="random()")
    means = base.mark_tick(color="black", thickness=3, size=60).encode(y="mean(avg):Q")
    rule = alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y=alt.Y("y:Q", title="Order"))
    return (pts + means + rule).properties(height=300)


def profit_curve():
    pm = np.zeros(19)
    for a in range(1, 7):
        for b in range(1, 7):
            for c in range(1, 7):
                pm[a + b + c] += 1 / 216
    ep = lambda q: sum(pm[d] * (2 * min(q, d) - q) for d in range(3, 19))
    d = pd.DataFrame(dict(q=range(3, 19), profit=[ep(q) for q in range(3, 19)]))
    return alt.Chart(d).mark_line(point=alt.OverlayMarkDef(color=OPT, size=55), color="#14213D", strokeWidth=2.5).encode(x=alt.X("q:Q", title="Order quantity", axis=alt.Axis(values=list(range(3, 19)), format="d")), y=alt.Y("profit:Q", title="Expected profit per day"), tooltip=["q", alt.Tooltip("profit:Q", format=".2f")]).properties(height=240)


def rows_for_dumbbell():
    S = D["samples"]
    rows = [dict(label="Original (Duke)", O=S[0]["mean_O"], C=S[0]["mean_C"])]
    for s in S[1:4]:
        rows.append(dict(label=s["sample"], O=s["mean_O"], C=s["mean_C"]))
    if "claude" in D:
        w = D["claude"]["wide"]; rows.append(dict(label="Claude agents", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    if "llama" in D:
        w = D["llama"]["wide"]; rows.append(dict(label="Llama agents", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    return rows


def wording_chart(rows):
    """Dot = O minus C gap; line = 95% confidence interval. Humans for reference."""
    d = pd.DataFrame(rows)
    order = list(d.label)
    pal = alt.Scale(domain=["Humans", "Claude", "Llama"], range=["#14213D", OPT, "#8A97A6"])
    col = alt.Color("kind:N", scale=pal, legend=alt.Legend(title="Group", orient="top"))
    XS = alt.Scale(domain=[-1.6, 2.4])
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, labelFontSize=13))
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color="#9AA8B5", strokeWidth=2).encode(x=alt.X("x:Q", scale=XS))
    pts = alt.Chart(d).mark_circle(size=200, opacity=1).encode(y=y, x=alt.X("diff:Q", scale=XS, title="Mean order, O minus C (units). Right of 0 = people-like direction"),
                                                              color=col, tooltip=["label", alt.Tooltip("diff:Q", format="+.2f")])
    layers = [zero]
    ci = d.dropna(subset=["lo", "hi"])
    if len(ci):
        layers.append(alt.Chart(ci).mark_rule(strokeWidth=3, opacity=0.65).encode(y=y, x=alt.X("lo:Q", scale=XS), x2="hi:Q", color=col))
    layers.append(pts)
    return alt.layer(*layers).properties(height=46 * len(d) + 50)


# ----------------------------------------------------------------- tables
def stats_table():
    t = pd.DataFrame(D["samples"] + ([D["llama_row"]] if "llama_row" in D else []))
    t = t.rename(columns={"sample": "Sample", "n_O": "n O", "n_C": "n C", "mean_O": "Mean O", "mean_C": "Mean C", "diff": "O − C", "d": "Cohen d", "p": "p"})
    t["Sample"] = t["Sample"].str.replace("AI agents", "Claude Haiku agents")
    for col in ("Mean O", "Mean C", "O − C"):
        t[col] = t[col].map(lambda v: f"{v:.2f}")
    t["Cohen d"] = t["Cohen d"].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f}")
    t["p"] = t["p"].map(lambda v: "–" if v is None or pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
    return t


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
    if "llama" in D:
        add("Llama agents", D["llama"]["wide"], f"{D['llama_checks']['comprehension_fail']} of 50")
    return pd.DataFrame(rows)


def wording_table(rows):
    out = []
    for r in rows:
        ci = "–" if r.get("lo") is None or pd.isna(r.get("lo")) else f"{r['lo']:+.2f} to {r['hi']:+.2f}"
        p = "–" if r.get("p") is None or pd.isna(r.get("p")) else ("<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}")
        out.append({"Model": r["model"], "Wording": r["label"], "Mean O": f"{r['mean_O']:.2f}", "Mean C": f"{r['mean_C']:.2f}", "O − C": f"{r['diff']:+.2f}",
                    "p": p, "95% CI": ci, "Orders at 10–11": pct(r.get("share")),
                    "Failed profit check": f"{r['fail_profit']} of {r['n']}", "Failed range check": f"{r['fail_range']} of {r['n']}"})
    return pd.DataFrame(out)


@st.cache_data
def prompt_text(cond):
    """Full prompt of the first agent in a condition, with the engine path shortened for display."""
    try:
        roster = json.loads((DATA / "state_api_haiku_5_5" / "roster.json").read_text())
        pid = next(a["id"] for a in roster if a["cond"] == cond)
        txt = (ROOT / "prompts" / f"{pid}.txt").read_text()
        txt = txt.split("HOW TO PLAY.")[0].rstrip() + "\n\n[Then each day: the day number, current play dollars and the question 'How many units would you like to order? (whole number from 3 to 18)'. After 25 days: two comprehension questions and a one-sentence strategy.]"
        return pid, txt
    except Exception:
        return None, None


@st.cache_data
def variant_texts():
    """Instruction text of every wording variant (read from sim/prompt_variants.py)."""
    try:
        sys.path.insert(0, str(ROOT / "sim"))
        import prompt_variants as pv, setup_agents as sa
        out = {"base": sa.INSTR}
        for v in ("v1", "v2", "v3"):
            out[v] = pv.get_instr(v, sa.INSTR)
        return out
    except Exception:
        return {}


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
    m[0].metric("AI models", 2 if "llama" in D else 1); m[1].metric("AI agents", "400", help="2 models × 4 wordings × 50 agents; the main comparison uses the 100 base-wording agents")
    m[2].metric("Rounds each", 25); m[3].metric("Human samples", 4)
    st.caption("Replicating Chen, Kök & Tong (2013), Management Science 59(2), 436–451.")


def step_original():
    a, b = st.columns(2)
    a.markdown("<div class='card-o'><b>O · Own financing</b><br>You pay 1 play dollar per unit when you order. You receive 2 play dollars per unit sold.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card-c'><b>C · Customer financing</b><br>You receive 1 play dollar per unit when you order. You pay 2 play dollars per unit left unsold.</div>", unsafe_allow_html=True)
    st.latex(r"\Delta_O = 2\min(q,D) - q \qquad \Delta_C = q - 2\max(q-D,0)\qquad\Rightarrow\ \Delta_O=\Delta_C")
    c1, c2 = st.columns(2)
    tile(c1, "<b>How the experiment worked</b><br>University students played a widget shop for <b>25 days</b>, starting with <b>100 play dollars</b>. Each day: order 3 to 18 widgets, then <b>three dice</b> are rolled and their sum is demand (average 10.5). Leftovers are discarded. A real bonus depended on the final play dollars.")
    tile(c2, "<b>Same profit, different wording</b><br>For every order and every dice roll the two schemes pay exactly the same, and the best order is <b>10 or 11</b>. Participants were randomly given O or C, so any difference in orders is caused by the wording.")


def step_human():
    S = D["samples"]
    _w(st.altair_chart, polish(dumbbell(rows_for_dumbbell()[:4])))
    m = st.columns(5)
    m[0].metric("Original O − C", f"{S[0]['diff']:+.2f}")
    for i in range(3):
        m[i + 1].metric(S[i + 1]["sample"], f"{S[i + 1]['diff']:+.2f}")
    m[4].metric("Spread between people", f"{K['h_sd']:.1f}")
    st.caption(f"O − C in units of order quantity. The three replications pool to {K['hg']:+.2f}.")
    st.markdown("**Why?** Under O the cost is paid at order time, before demand is known, so it feels like a sure loss and people order more to try to win it back. Under C the money arrives first and the risk feels like giving it back, so people order less. A fully rational person would ignore the wording.")


def step_demo():
    st.markdown("<div class='lead'>Open the <b>Game</b> tab, choose a scheme, and play two or three days. After each day, open <b>Same day in the other wording</b> to see that the profit is identical.</div>", unsafe_allow_html=True)
    st.write("")
    st.info("This is exactly what the human participants and the AI agents saw: the same instructions, the same dice, a different description of the money.")


def step_design():
    a, b, c, d = st.columns(4)
    tile(a, "<b>1 Instructions only</b><br>The study's O or C text, word for word, plus the bonus rules. No age, gender or other background.")
    tile(b, "<b>2 Game engine</b><br>Checks every order (whole number 3–18), rolls real dice <i>after</i> the order, applies the payoff rule.")
    tile(c, "<b>3 Memory</b><br>Each agent sees all of its earlier days (orders, dice, results).")
    tile(d, "<b>4 Two comprehension checks</b><br>Profit for order 10 and demand 7 (answer +4) and the lowest and highest possible demand (3 and 18).")
    st.write("")
    a, b = st.columns(2)
    tile(a, "<b>Two models, four wordings, 50 agents each run</b><br>25 per scheme, the same size as the original sample.<br>• <b>Claude Haiku 5.5</b> through the Claude API and LangGraph<br>• <b>Llama 3.1 8B</b> through LangGraph on a free local Ollama<br>• Study wording plus three alternative wordings")
    tile(b, "<b>Kept the same:</b> payoffs, 25 rounds, dice, order range, instructions.<br><b>Changed, and why:</b> the on-screen game became a structured answer inside a program (agents cannot click); no real money can be paid to an agent.")
    st.code("decide (order 3–18)  →  engine rolls dice and applies payoff  →  repeat for 25 days  →  two comprehension checks", language="text")
    st.caption("Full settings table, the exact prompt and the validation checks are in the AI system tab.")


def step_replay_pointer():
    st.markdown("<div class='lead'>Open the <b>AI replay</b> tab. Pick a Claude agent and press <b>Play</b>: it settles on 10 or 11 within a few days. Then pick a Llama agent: it often drifts upward to 13 or more.</div>", unsafe_allow_html=True)
    st.write("")
    st.info("At the end, read the agent's own sentence about how it decided.")


def step_compare():
    _w(st.altair_chart, polish(dumbbell(rows_for_dumbbell())))
    m = st.columns(5)
    m[0].metric("Humans O − C", f"{K['hg']:+.2f}", help="Pooled across the three replications")
    m[1].metric("Claude O − C", f"{K['cg']:+.2f}"); m[2].metric("Llama O − C", "–" if K["lg"] is None else f"{K['lg']:+.2f}")
    m[3].metric("Claude at 10–11", f"{K['c_share']:.0%}"); m[4].metric("Llama at 10–11", pct(K["l_share"]))
    st.dataframe(behaviour_table(), hide_index=True)
    st.markdown(f"**With the study's wording, neither model shows the human gap.** People order about {K['hg']:.1f} units more under O; Claude's gap is {K['cg']:+.2f} and Llama's {'–' if K['lg'] is None else format(K['lg'], '+.2f')}. Claude orders almost only 10 or 11 and barely varies; Llama over-orders. Both fail the profit check often, so the agents may not see that the two schemes pay the same.")


def step_wording():
    rows = wording_rows_for_chart()
    _w(st.altair_chart, polish(wording_chart(rows)))
    W = load_wording()
    cg = ", ".join(f"{r['diff']:+.2f}" for r in W.get("claude", [])); lg = ", ".join(f"{r['diff']:+.2f}" for r in W.get("llama", []))
    st.markdown(f"**The gap depends on the wording, for both models.** O − C for the study wording and v1, v2, v3: Claude {cg}; Llama {lg}. We kept the study's wording for the main results and report the others openly.")


def step_why():
    X, XL = D.get("xai", {}), D.get("xai_llama", {})

    def share(k, src):
        s_ = src[k]["mean_abs_shap"]; return s_["cond_O"] / sum(s_.values())
    a, b = st.columns(2)
    if X:
        tile(a, f"<b>Humans</b><br>Anchor on their own last order (mean |Shapley| {X['shap_human']['mean_abs_shap']['prev_order']:.2f}). The payment scheme is about {share('shap_human', X):.0%} of the explanation. Cross-validated R² {X['shap_human']['cv_R2_by_agent_group']:.2f}.")
        tile(b, f"<b>Claude agents</b><br>Also anchor on their own last order, but the payment scheme is only about {share('shap_ai_common', X):.0%} of the explanation, and with history they react to last demand and profit. R² {X['shap_ai_common']['cv_R2_by_agent_group']:.2f}, up to {X['shap_ai_extended']['cv_R2_by_agent_group']:.2f} with history.")
    xf = ROOT / "analysis/fig_xai.png"
    if xf.exists():
        _w(st.image, str(xf))
    if "llama_checks" in D:
        extra = f" The payment scheme is about {share('shap_ai_common', XL):.0%} of its explanation." if XL else ""
        st.markdown(f"**Llama:** {D['llama_checks']['comprehension_fail']} of 50 failed a comprehension check and its orders drift upward through the game, so its results reflect weak task understanding more than human-like bias.{extra}")


def step_conclusion():
    a, b, c = st.columns(3)
    tile(a, "<b>What we learned</b><br>In the study's wording neither model reproduced the human framing gap; some alternative wordings gave Claude a small gap in the human direction. Claude acts almost like an optimiser; Llama does not understand the game. Most agents fail the profit check, which may explain why the framing does not bite.")
    tile(b, "<b>Limits of our simulation</b><br>• One model family per run, so agents are not independent<br>• No real stakes; Haiku ran at default sampling<br>• Agents may reason silently about the optimum<br>• Results depend on the model and the wording<br>• The agents that passed both checks are few and unevenly split between O and C")
    tile(c, "<b>Caution for researchers</b><br>Use AI to screen ideas quickly, then confirm with people. Always report the model, prompt and settings, and test whether the result survives a change in wording.")
    st.write("")
    st.subheader("Questions?")


def notes(i):
    K_ = K
    L = {
        0: ["Start with the question, not the method.", "A shop orders before it knows demand: too little loses sales, too much wastes stock.", "Same money, different wording: do people, and AI agents, decide differently?"],
        1: ["Walk the two cards slowly: O pays up front, C is paid up front.", "Point to the formula: both pay 2·min(q,D) − q. Identical for every order and every dice roll.", "Best order is 10 or 11. 25 days, 100 play dollars, orders 3 to 18."],
        2: [f"Original study gap {K_['orig']:+.2f}; replications found the same direction, pooled {K_['hg']:+.2f}.", "Mechanism: mental accounting. Paying before demand is known feels like a sure loss.", f"People vary a lot (spread about {K_['h_sd']:.1f})."],
        3: ["Play one or two days, then open 'Same day in the other wording'.", "Say: this is what every participant, human or AI, saw."],
        4: ["Four building blocks: instructions only (no personas), engine, memory, two comprehension checks.", "Same instructions word for word; real dice rolled by code after the order.", "Two models and four wordings, 50 agents per run, 25 per scheme.", "Say what changed and why: GUI to a program, no real money."],
        5: ["AI replay tab: a Claude agent first, then a Llama agent.", "Read the agent's own strategy sentence."],
        6: [f"Claude: gap {K_['cg']:+.2f} versus {K_['orig']:+.2f} for people; {K_['c_share']:.0%} of orders at 10–11.", f"Llama: gap {'–' if K_['lg'] is None else format(K_['lg'], '+.2f')} and it orders too much in both schemes.", f"Spread between participants: people about {K_['h_sd']:.1f}, Claude about {K_['c_sd']:.1f}.", "Answer the question: in the study's wording the human gap did not appear in either model."],
        7: ["Same game, different wording: both models' gaps move.", "We did not tune prompts to get a result; the main run uses the study's wording.", "Comparisons are within the same scheme (variant C against base C, variant O against base O).", "Caveat: the O text for v1 and v2 was drafted by us."],
        8: ["People anchor on their last order; for the agents the payment scheme explains very little.", "Most agents fail the profit check, so they may not see that O and C are identical."],
        9: ["AI did not reproduce the human gap in the study's wording; a few wordings gave Claude a small one.", "Limits: one model family, not independent, silent reasoning, no real stakes, wording sensitivity, few agents pass both checks.", "Validate with people before relying on AI participants.", "Invite questions."],
    }
    return L[i]


def wording_rows_for_chart():
    rows = [dict(label="Humans, original study", kind="Humans", diff=float(K["orig"]), lo=None, hi=None),
            dict(label="Humans, replications (pooled)", kind="Humans", diff=float(K["hg"]), lo=None, hi=None)]
    for mkey, rs in load_wording().items():
        for r in rs:
            rows.append(dict(label=f"{r['label']} · {r['group']}", kind=r["group"], diff=float(r["diff"]), lo=r.get("lo"), hi=r.get("hi")))
    return rows


# key, title, seconds at 10 minutes, rubric key, takeaway, function
STEPS = [
    ("business", "The business problem", 45, "bg", "Same money, different words. Do AI agents react like people?", step_business),
    ("original", "The original experiment", 75, "orig", "Two payment wordings, identical profit.", step_original),
    ("human", "What people did, and why", 60, "orig", "People order more when they pay up front.", step_human),
    ("demo", "Try the game", 30, "qual", "You see exactly what participants saw.", step_demo),
    ("design", "How we built the AI participants", 90, "ai", "Same instructions and engine, no personas, 50 agents per run.", step_design),
    ("replay", "Watch an agent play", 30, "ai", "Claude settles quickly; Llama often drifts upward.", step_replay_pointer),
    ("compare", "Results: AI versus people", 130, "cmp", "In the study wording, neither model shows the human gap.", step_compare),
    ("wording", "Does the prompt wording matter?", 40, "cmp", "The gap changes with the wording for both models.", step_wording),
    ("why", "What explains the difference?", 40, "cmp", "Agents barely react to the payment scheme.", step_why),
    ("conclude", "Conclusions and limits", 60, "conc", "Weak or no framing effect in AI agents.", step_conclusion),
]
RUBRIC = {"bg": ("Business background and research question", 15), "orig": ("Original findings and theory", 15), "ai": ("AI-agent system and experimental design", 25),
          "cmp": ("Comparison between AI agents and humans", 25), "qual": ("Presentation quality and organisation", 10), "conc": ("Conclusions and limitations (part of quality and the brief)", 0)}


def fmt_t(sec):
    sec = int(round(sec)); return f"{sec // 60}:{sec % 60:02d}"


def present_tab():
    ss = st.session_state
    ss.setdefault("step", 0); ss.setdefault("t0", None)
    cur = ss.step
    talk, show_notes = 10, False
    secs = [s[2] for s in STEPS]
    if PRESENTER:
        top = st.columns([2, 2, 3, 2])
        talk = top[0].radio("Talk length", [10, 12], horizontal=True, format_func=lambda x: f"{x} min", key="talk")
        secs = [s[2] * talk / 10 for s in STEPS]
        total = sum(secs)
        if btn(top[1], "⏱ Start timer" if ss.t0 is None else "⏱ Reset timer", key="timer"):
            ss.t0 = time.time() if ss.t0 is None else None
            st.rerun()
        show_notes = top[3].checkbox("Speaker notes", key="notes")
        plan_end = sum(secs[:cur + 1])
        if ss.t0:
            e = time.time() - ss.t0
            cls, msg = ("pace-ok", "on pace") if e <= plan_end else ("pace-bad", f"over by {fmt_t(e - plan_end)}")
            top[2].markdown(f"**{fmt_t(e)}** of {fmt_t(total)} · <span class='{cls}'>{msg}</span> (this step should end by {fmt_t(plan_end)}; updates when you click)", unsafe_allow_html=True)
        else:
            top[2].caption(f"Planned {fmt_t(total)} plus Q&A. This step should end by {fmt_t(plan_end)}.")
    cols = st.columns(len(STEPS))
    for i, s in enumerate(STEPS):
        if btn(cols[i], str(i + 1), key=f"jump{i}", help=s[1], type="primary" if i == cur else "secondary"):
            ss.step = i; st.rerun()
    st.progress((cur + 1) / len(STEPS))
    key, title, _, rub, take, fn = STEPS[cur]
    rname, rpts = RUBRIC[rub]
    with st.container(border=True):
        if PRESENTER:
            badge = f"<span class='rub'>{rname}{f' · {rpts} pts' if rpts else ''}</span>"
            st.markdown(f"<div class='kick'>Step {cur + 1} of {len(STEPS)} · about {fmt_t(secs[cur])}{badge}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='kick'>Step {cur + 1} of {len(STEPS)}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big'>{title}</div><div class='take'>{take}</div>", unsafe_allow_html=True)
        fn()
        if show_notes:
            with st.expander("Speaker notes", expanded=True):
                for line in notes(cur):
                    st.markdown(f"- {line}")
    a, b, _ = st.columns([1, 1, 8])
    if btn(a, "← Back", disabled=cur == 0):
        ss.step -= 1; st.rerun()
    if btn(b, "Next →", type="primary", disabled=cur == len(STEPS) - 1):
        ss.step += 1; st.rerun()


# ----------------------------------------------------------------- game
INSTR = {"O": ["Decide order quantity: each day, choose how many widgets to stock.", "−1 per unit ordered: you pay 1 play dollar for every widget you order.",
               "Random demand: three dice are rolled. Their sum is the day’s demand.", "+2 per unit sold: you sell the smaller of demand and your stock and receive 2 play dollars for each unit sold.", "Leftovers are discarded."],
         "C": ["Decide order quantity: each day, choose how many widgets to stock.", "+1 per unit ordered: you receive 1 play dollar for every widget you order.",
               "Random demand: three dice are rolled. Their sum is the day’s demand.", "−2 per unit unsold: you sell the smaller of demand and your stock and pay 2 play dollars for every leftover unit.", "Leftovers are discarded."]}


def outcome(cond, q, demand):
    sold, left = min(q, demand), max(q - demand, 0)
    if cond == "O":
        return f"You ordered {q} and paid {q}. Demand {demand}: you sold {sold} and received {2 * sold}. {left} left over."
    return f"You ordered {q} and received {q} in advance. Demand {demand}: you sold {sold}. {left} unsold, so you pay {2 * left}."


def game_tab():
    ss = st.session_state
    ss.setdefault("sid", secrets.token_hex(4)); ss.setdefault("g", None)
    g = ss.g
    if g is None:
        forced = st.query_params.get("cond")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='sec' style='font-size:1.5rem'>Run a widget shop</div>", unsafe_allow_html=True)
            st.markdown("Each day you choose how many widgets to order, three dice decide demand, and you earn or lose play dollars. Both pay schemes give the same profit for any order and any demand, but they are worded differently.")
            choice = st.radio("Pay scheme", ["Random", "O · own financing", "C · customer financing"], horizontal=True)
            n = st.selectbox("Length", [25, 10, 5], format_func=lambda x: {25: "25 days (original)", 10: "10 days (quick)", 5: "5 days (demo)"}[x])
            if st.button("Start game", type="primary"):
                cond = forced if forced in ("O", "C") else (choice[0] if choice[0] in ("O", "C") else RNG.choice(["O", "C"]))
                ss.g = dict(cond=cond, N=n, day=1, cash=100, hist=[], last=None); st.rerun()
        with c2:
            st.markdown("<div class='sec' style='font-size:1.5rem'>How a day works</div>", unsafe_allow_html=True)
            st.markdown("1. Choose how many widgets to order (3 to 18).\n2. Three dice are rolled. Their sum is the day's demand.\n3. See your profit, then open **Same day in the other wording** to check that the money works out the same under both schemes.")
            if PRESENTER:
                st.caption("Presenter: choose 5 days (demo) to reach the end-of-game comparison quickly. Add ?cond=O or ?cond=C to force a scheme.")
        return
    c = g["cond"]
    left, right = st.columns([3, 2])
    with left:
        finished = len(g["hist"]) >= g["N"]
        h = st.columns(3); h[0].metric("Day", f"{min(g['day'], g['N'])} / {g['N']}"); h[1].metric("Play dollars", g["cash"]); h[2].markdown(f"**{'OWN FINANCING' if c == 'O' else 'CUSTOMER FINANCING'}**")
        other = "C" if c == "O" else "O"
        if g["last"]:
            L = g["last"]; st.markdown(f"<div class='diceline'>{' '.join(FACES[x - 1] for x in L['d'])}</div>", unsafe_allow_html=True)
            st.info(outcome(c, L["q"], L["demand"]) + f"  \n**{L['change']:+d} play dollars**")
            with st.expander("Same day in the other wording"):
                st.write(outcome(other, L["q"], L["demand"])); st.write(f"Change: {L['change']:+d} play dollars, identical to what you got.")
        if finished:
            qs = [r["q"] for r in g["hist"]]
            a, b, d = st.columns(3)
            a.metric("Your average order", f"{np.mean(qs):.1f}")
            hum = grp(D["human"], c).avg.mean(); a2 = grp(D["claude"]["wide"], c).avg.mean() if "claude" in D else None
            b.metric(f"Humans in {c}", f"{hum:.1f}"); d.metric(f"Claude agents in {c}", f"{a2:.1f}" if a2 else "–")
            st.markdown("Under both schemes, expected profit is highest at 10 or 11 units. " + ("You ordered above the optimum." if np.mean(qs) > 11.2 else "You ordered below the optimum." if np.mean(qs) < 9.8 else "You stayed close to the optimum."))
            if st.button("Play again"):
                ss.g = None; st.rerun()
        else:
            st.markdown(f"**How many widgets for Day {g['day']}?** " + ("At this moment you pay 1 play dollar per unit." if c == "O" else "At this moment you receive 1 play dollar per unit as an advance."))
            q = st.radio("Order quantity", list(range(3, 19)), horizontal=True, index=None, key=f"q{g['day']}", label_visibility="collapsed")
            if st.button(f"Order {q} and roll" if q else "Choose a quantity", type="primary", disabled=q is None):
                dice = [RNG.randint(1, 6) for _ in range(3)]; dem = sum(dice); sold, lft = min(q, dem), max(q - dem, 0)
                ch = 2 * sold - q if c == "O" else q - 2 * lft
                g["cash"] += ch; g["last"] = dict(q=q, d=dice, demand=dem, change=ch)
                g["hist"].append(dict(day=g["day"], q=q, demand=dem, change=ch)); g["day"] += 1
                st.rerun()
    with right:
        st.markdown("**Instructions you were given**")
        for r in INSTR[c]:
            st.markdown(f"- {r}")
        st.caption("Start with 100 play dollars. Goal: finish with as many as possible.")
        if g["hist"]:
            d = pd.DataFrame(dict(day=[r["day"] for r in g["hist"]], order=[r["q"] for r in g["hist"]]))
            ch = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=45, color=O_COL if c == "O" else C_COL), color=O_COL if c == "O" else C_COL).encode(x=alt.X("day:Q", title="Day", scale=alt.Scale(domain=[1, g["N"]])), y=alt.Y("order:Q", title="Order", scale=alt.Scale(domain=[3, 18])))
            _w(st.altair_chart, polish(ch + alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y=alt.Y("y:Q", title="Order"))))


# ----------------------------------------------------------------- AI system
def system_tab():
    V = D["verdict"]; k = D["checks"]
    st.markdown("<div class='lead'>How the AI participants were generated, what they were told, what they could see, and how every decision was recorded and checked.</div>", unsafe_allow_html=True)
    st.write("")
    box = card("1 · How the agents are generated and initialised")
    a, b, c, d = box.columns(4)
    tile(a, "<b>Assign</b><br>50 agents per model, 25 per scheme, random order (fixed seed 2026).")
    tile(b, "<b>No background</b><br>No age, gender, major or degree: the agent gets only the study's instructions.")
    tile(c, "<b>Brief</b><br>Paid lab participant, not an AI assistant, with the study's instructions and bonus rules.")
    tile(d, "<b>Play</b><br>25 days, one order per day, then two comprehension checks (profit +4; dice range 3 and 18) and a one-sentence strategy.")

    box = card("2 · Original study versus our replication", "Nothing was changed to make the AI look more human. Every difference is listed with its reason.")
    t = pd.DataFrame([
        ("Participants", "University students: 50 in the original (29 O, 21 C), plus three later replications", "50 agents per model and wording (25 O, 25 C), random assignment", "Same size. Agents of one model are not independent people."),
        ("Instructions", "Written game instructions for scheme O or C", "The same text, word for word", "None"),
        ("Payoffs", "O: −1 per unit ordered, +2 per unit sold. C: +1 per unit ordered, −2 per unit unsold", "Identical, applied by the engine", "None"),
        ("Rounds and start", "25 rounds, 100 play dollars", "25 rounds, 100 play dollars", "None"),
        ("Demand and orders", "Sum of three dice, order 3 to 18", "Same; the engine rolls real dice after the order", "None"),
        ("Interface", "On-screen game", "Text commands (or structured output) to a game engine", "Agents cannot click. The engine enforces the rules."),
        ("Incentive", "Real bonus: 5 dollars plus 1 dollar per 50 play dollars", "Same text shown; no money can be paid", "Agents cannot be paid. Kept as described."),
        ("Participant background", "Students from many majors", "None given", "The prompt does not specify a background, so we did not invent one."),
        ("Memory", "A repeated 25-day task", "Agent sees all earlier days (orders, dice, results)", "Needed for a repeated task; people remember too."),
        ("Interaction", "Each person decides alone", "Each agent decides alone", "None"),
        ("Comprehension check", "Used to screen human participants", "Two checks after day 25: profit for order 10 and demand 7 (+4), and the lowest and highest possible demand (3 and 18). Results shown for all agents and for those that passed both", "Same exclusion idea; most agents fail the profit check, so we show both views."),
    ], columns=["Element", "Original human study", "Our AI replication", "Change and why"])
    box.dataframe(t, hide_index=True)

    box = card("3 · What the agent was told", "This is the real prompt file of one agent (base wording). Agents differ only in their scheme (O or C) and, in the wording tests, in the instruction text.")
    pick = box.radio("Scheme", ["O", "C"], horizontal=True, key="prompt_cond")
    pid, txt = prompt_text(pick)
    if txt:
        with box.expander(f"Full prompt of agent {pid} ({pick})"):
            st.text(txt)
    else:
        box.info("Prompt files were not found in the prompts folder.")
    box.markdown("**Each day the agent sees:** the day number, its current play dollars, a reminder of when the 1 play dollar moves (at order time), and the question *How many units would you like to order? (whole number from 3 to 18)*. After the order it sees the dice, demand, units sold and its change in play dollars.")

    box = card("4 · How decisions are generated, recorded and validated")
    a, b, c = box.columns(3)
    tile(a, "<b>Generated</b><br>Both models answer in a structured format (order 3–18) inside a LangGraph loop: decide, play, repeat for 25 days, debrief. Claude runs through the Claude API, Llama through a local Ollama.")
    tile(b, "<b>Recorded</b><br>The engine stores every day (order, three dice, demand, sold, leftover, change, cash) as JSON, then exports CSV.")
    tile(c, f"<b>Validated</b><br>{k['n_agents']} agents × 25 days = {k['n_rows']} orders. Out of range: {k['orders_out_of_range']}. Missing: {k['missing']}. Payoff-rule violations: {k['payoff_rule_violations']}. Demand mismatches: {k['demand_mismatch']}. Mean demand {k['dice_mean_demand']:.2f} (expected 10.5). All eight runs (2 models × 4 wordings) pass the same checks.")

    box = card("5 · Models, settings and limitations")
    ll = D["llama_checks"]["comprehension_fail"] if "llama_checks" in D else "–"
    models = pd.DataFrame([
        ("Claude Haiku 5.5 (Claude API + LangGraph)", "Structured output, message-history memory, one session per agent", "Default sampling (the model does not accept a temperature), at most 300 tokens per answer", f"50 per wording ({k['comprehension_fail']} failed a check in the base wording)"),
        ("Llama 3.1 8B (LangGraph + Ollama)", "Structured output, message-history memory, free and local", "Temperature 1.0, at most 300 tokens per answer", f"50 per wording ({ll} failed a check in the base wording)"),
        ("Wording variants v1, v2, v3 (both models)", "Same pipeline, different instruction wording", "Same settings", "50 per wording and model"),
    ], columns=["Model", "How it was run", "Settings", "Agents"])
    box.dataframe(models, hide_index=True)
    if V.get("m_limits"):
        box.markdown(f"**Limits.** {V['m_limits']}")

    box = card("6 · The dataset (Deliverable B)", "data/ai_dataset_all_long.csv (10,000 decisions) and ai_dataset_all_participants.csv (400 agents). Full description in DATA_DICTIONARY.md.")
    vars_ = pd.DataFrame([
        ("model, wording, agent_id, condition", "Which model and wording, the agent and its scheme (O or C). No personas"),
        ("day, order", "Round 1–25 and the units ordered (3–18)"),
        ("d1, d2, d3, demand", "The three dice and their sum, rolled by the engine after the order"),
        ("sold, leftover, change, cash_end", "Units sold, unsold units, play-dollar change that day, and balance"),
        ("comprehension_answer, range_min_answer, range_max_answer", "Answers to the two end-of-game checks (profit, correct +4; lowest and highest demand, correct 3 and 18)"),
        ("fail_profit_check, fail_range_check, comprehension_fail", "1 if the profit check, the range check, or either one failed"),
        ("strategy, avg_order, final_cash", "The agent's own strategy sentence, its mean order and final balance"),
    ], columns=["Variables", "Meaning"])
    box.dataframe(vars_, hide_index=True)


# ----------------------------------------------------------------- AI replay
def replay_tab():
    models = [m for m in ("claude", "llama") if m in D]
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", models, format_func=lambda m: "Claude Haiku 5.5 (Claude API + LangGraph)" if m == "claude" else "Llama 3.1 8B (LangGraph + Ollama)")
    scheme = c2.selectbox("Scheme", ["O", "C"])
    P = D[model]["part"]; P = P[P.condition == scheme]
    agent = c3.selectbox("Agent", list(P.agent_id), format_func=lambda a: f"{a} (passed check)" if int(P[P.agent_id == a].comprehension_fail.iloc[0]) == 0 else f"{a} (failed check)")
    a = P[P.agent_id == agent].iloc[0]
    L = D[model]["long"]; L = L[L.agent_id == agent].sort_values("day")
    st.markdown(f"**{'Own financing' if scheme == 'O' else 'Customer financing'}**  ·  Checks: profit answer {a.comprehension_answer} (correct +4), dice range {a.range_min_answer} to {a.range_max_answer} (correct 3 to 18)")
    st.session_state.setdefault("rday", 0)
    k1, k2, k3 = st.columns([1, 1, 3])
    play = btn(k1, "▶ Play replay", type="primary")
    speed = k2.select_slider("Speed", ["slow", "normal", "fast"], value="normal")
    delay = {"slow": 1.2, "normal": 0.7, "fast": 0.25}[speed]
    n = k3.slider("Day", 0, 25, key="rday")
    mph = st.empty()
    lc, rc = st.columns([2, 3])
    ph = lc.empty(); chart_ph = rc.empty()

    def draw(n):
        sub = L[L.day <= n]
        with mph.container():
            if n == 0:
                st.info("Press Play, or drag the Day slider, to step through the agent’s 25 days.")
            else:
                r = sub.iloc[-1]
                m = st.columns(4); m[0].metric("Day", f"{n} / 25"); m[1].metric("Play dollars", int(r.cash_end)); m[2].metric("Agent ordered", int(r.order)); m[3].metric("Change", f"{int(r.change):+d}")
        with ph.container():
            if n:
                r = sub.iloc[-1]
                st.markdown(f"<div class='diceline'>{' '.join(FACES[int(x) - 1] for x in (r.d1, r.d2, r.d3))}</div>", unsafe_allow_html=True)
                st.info(outcome(scheme, int(r.order), int(r.demand)))
        if n:
            d = sub[["day", "order"]]
            col = O_COL if scheme == "O" else C_COL
            ch = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=45, color=col), color=col).encode(x=alt.X("day:Q", title="Day", scale=alt.Scale(domain=[1, 25])), y=alt.Y("order:Q", title="Order", scale=alt.Scale(domain=[3, 18])))
            _w(chart_ph.altair_chart, polish(ch + alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y=alt.Y("y:Q", title="Order"))))
        else:
            chart_ph.empty()

    draw(n)
    if play:
        for i in range(1, 26):
            draw(i); time.sleep(delay)
    if n == 25 or play:
        box = card("In the agent’s own words")
        box.markdown(f"“{a.strategy}”")
        box.caption("This agent failed at least one of the two comprehension checks." if int(a.comprehension_fail) else "This agent passed both comprehension checks.")


# ----------------------------------------------------------------- results
def results_tab():
    V = D["verdict"]
    H = D["human"]; lw = D["llama"]["wide"] if "llama" in D else None
    k = st.columns(4)
    k[0].metric("Humans O − C (pooled)", f"{K['hg']:+.2f}")
    k[1].metric("Claude agents O − C", f"{K['cg']:+.2f}")
    k[2].metric("Llama agents O − C", "–" if K["lg"] is None else f"{K['lg']:+.2f}")
    k[3].metric("Original study O − C", f"{K['orig']:+.2f}")
    if V:
        st.markdown(f"<div class='verdict'><b>{V.get('headline', '')}</b><br>{V.get('verdict', '')}</div>", unsafe_allow_html=True)

    box = card("Do AI agents reproduce the main pattern?", "Orange = own financing (O), teal = customer financing (C). The shaded band is the profit-maximising 10 to 11.")
    _w(box.altair_chart, polish(dumbbell(rows_for_dumbbell())))

    c1, c2 = st.columns(2)
    groups = [("Human O", "O", grp(H, "O").avg), ("Human C", "C", grp(H, "C").avg)]
    for m, nm in (("claude", "Claude"), ("llama", "Llama")):
        if m in D:
            groups += [(f"{nm} O", "O", grp(D[m]["wide"], "O").avg), (f"{nm} C", "C", grp(D[m]["wide"], "C").avg)]
    with c1:
        box = card("Do they show the same variation?", "One dot per person or agent; the black bar is the group mean.")
        _w(box.altair_chart, polish(strips(groups)))
    with c2:
        box = card("Behaviour compared", "Orders at 10–11 are the profit-maximising choices.")
        box.dataframe(behaviour_table(), hide_index=True)

    ser = [("Humans", D["human"], "O"), ("Humans", D["human"], "C")]
    for m, nm in (("claude", "Claude"), ("llama", "Llama")):
        if m in D:
            ser += [(nm, D[m]["wide"], "O"), (nm, D[m]["wide"], "C")]
    c3, c4 = st.columns(2)
    with c3:
        box = card("Orders by day", "Different line styles are different groups; Llama drifts upward through the game.")
        _w(box.altair_chart, polish(by_day_chart(ser)))
    with c4:
        box = card("What a wrong order costs", "Expected profit per day is the same curve under both schemes.")
        _w(box.altair_chart, polish(profit_curve()))
    box = card("Statistics", "Unit of analysis is the participant’s 25-day average (two-sample t-test). The original-study row is taken from the paper.")
    _w(box.dataframe, stats_table(), hide_index=True)
    box = card("What might explain the differences?")
    box.markdown("- **Optimising versus anchoring.** Claude works out something close to the best order; people anchor on their last order and are pulled by the wording.\n- **Low variation.** Agents of one model behave alike, so their average gap is precise but tiny.\n- **Task understanding.** Most agents failed the profit check (Llama almost all), so they may not see that O and C pay the same. Few agents pass both checks, so we cannot test that cleanly.\n- **Mental accounting needs a feeling of loss.** There are no real stakes for an agent.")


# ----------------------------------------------------------------- prompt wording
def wording_tab():
    W = load_wording(); rows = [r for m in W.values() for r in m]
    st.markdown("<div class='lead'>A robustness check: the same game with the instruction text worded in different ways, for both models (50 agents per run). If the bias were real and stable, the gap should survive.</div>", unsafe_allow_html=True)
    st.write("")
    box = card("O − C gap under each wording", "Dot = gap. Line = 95% confidence interval. The vertical line at 0 means no framing effect. Humans shown for reference.")
    _w(box.altair_chart, polish(wording_chart(wording_rows_for_chart())))
    box = card("Numbers, Claude and Llama separately", "Unit of analysis: each agent's 25-day average order. All 50 agents per run.")
    box.dataframe(wording_table(rows), hide_index=True)
    vb = load_vs_base()
    if len(vb):
        box = card("Each wording compared with the base wording, within the same scheme", "Variant C against base C and variant O against base O (never variant C against base O). Welch t-test on agent averages.")
        t = vb.copy()
        t["p_value"] = t["p_value"].map(lambda v: "–" if pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
        t = t.rename(columns={"model": "Model", "comparison": "Comparison", "n_variant": "n variant", "n_base": "n base", "mean_variant": "Mean variant", "mean_base": "Mean base", "difference": "Difference", "p_value": "p", "cohens_d": "Cohen d"})
        box.dataframe(t, hide_index=True, height=460)
    c1, c2 = st.columns(2)
    cl, ll = W.get("claude", []), W.get("llama", [])
    with c1:
        box = card("What it shows")
        cg = ", ".join(f"{r['diff']:+.2f}" for r in cl); lg = ", ".join(f"{r['diff']:+.2f}" for r in ll)
        box.markdown(f"- O − C for the study wording and v1, v2, v3: **Claude {cg}**; **Llama {lg}**.\n- Claude keeps the human direction (O above C) in every wording, and the gap is significant in v1 and v2, but it is far smaller than the human gap of about 1.2 to 2.\n- Llama's gap changes sign and is never significant; it also over-orders in every wording.\n- The wording changes the orders themselves, for example Claude's O orders rise under v1 compared with base O.\n- Most agents fail the profit check in every wording (Claude 56–72%, Llama 92–98%).")
    with c2:
        box = card("Be transparent about")
        box.markdown("- Only the scheme-C text of v1 and v2 was supplied; we drafted the scheme-O text by flipping the payment direction.\n- v2 adds bankruptcy warnings that may push O orders down by themselves.\n- v3 replaces the real-money bonus paragraph with a token-spend incentive.\n- Agents that passed both checks are few and unevenly split between O and C (see results_summary.csv), so passed-only gaps are suggestive.\n- These runs are checks. The main results use the study's own wording.")
    texts = variant_texts()
    if texts:
        box = card("The exact wording of each variant")
        a, b = box.columns(2)
        v = a.selectbox("Wording", [w[0] for w in WORDINGS], format_func=lambda x: dict((w[0], w[1]) for w in WORDINGS)[x])
        s = b.radio("Scheme", ["O", "C"], horizontal=True, key="wording_scheme")
        with box.expander("Show instruction text", expanded=False):
            st.text(texts[v][s])


# ----------------------------------------------------------------- Q&A
QA = [
    ("Do the AI agents show the framing effect?", f"Not in the study's wording: Claude's O − C gap is {K['cg']:+.2f} and Llama's {'–' if K['lg'] is None else format(K['lg'], '+.2f')}, against about {K['orig']:.1f} for people in the original study. In some alternative wordings Claude shows a small gap in the human direction, but it is far smaller than people's."),
    ("Why are small gaps sometimes significant?", f"Claude's agents barely vary (spread about {K['c_sd']:.1f} units against about {K['h_sd']:.1f} for people), so even a small gap can be precisely measured. The p-value reflects low variance, not a large effect. Treat the 50 agents as one model, not 50 independent minds."),
    ("Why did you not give the agents ages, genders and degrees?", "The study's prompt does not specify them. Our first run used random student backgrounds, but they added a variable we could not justify (and some combinations made no sense), so the final run gives agents only the study's instructions."),
    ("What are the two comprehension checks?", "After day 25 each agent answers (1) by how many play dollars its balance changes when it orders 10 and demand is 7 (correct: +4 in both schemes) and (2) the lowest and highest possible demand (3 and 18). Claude gets the range right every time but often misses the profit; Llama misses both often."),
    ("Why does Claude run without a temperature setting?", "The Haiku 5.5 model does not accept a temperature, so it ran at its default sampling. Llama ran at temperature 1.0."),
    ("Did you change the prompts to get a human-like result?", "No. The main run uses the study's instructions word for word. We then tested three alternative wordings as a robustness check and report all of them, including the ones where the gap disappears."),
    ("Why did most Llama agents fail the comprehension check?", "A small 8-billion-parameter model often cannot work out the payoff arithmetic or the dice range. It plays the game, but without understanding it, so we do not treat its gap as evidence of a human-like bias."),
    ("Are the 50 agents independent?", "No. They share one model and differ only in the dice and in the pay scheme. That is why we describe results at the model level and avoid claims about 'populations' of AI."),
    ("How do you know the dice are fair?", f"The engine rolls three dice with the operating system's random source after each order. Mean demand across the agent days is {D['checks']['dice_mean_demand']:.2f} (Claude, base wording), against an expected 10.5."),
    ("Could the agents simply be calculating the best order?", "Yes, and that is a likely explanation for Claude: it often works out that 10 or 11 maximises expected profit. People rely on rules of thumb, such as repeating their last order, which leaves room for the framing effect."),
    ("Why 25 rounds and 50 agents?", "To match the original design: 25 rounds, and a sample the same size as the original study (50 participants)."),
    ("What does 'payoff-identical' mean?", "For any order q and demand D, scheme O pays 2·min(q,D) − q and scheme C pays q − 2·max(q−D, 0). These are equal, so any difference in orders is due to wording alone."),
    ("What is mental accounting?", "People keep separate mental accounts for money paid before and after an outcome is known. A cost paid up front feels like a sure loss and changes behaviour, even though the total is the same."),
    ("What should researchers be careful about when using AI participants?", "Results depend on the model, the prompt wording and the settings; agents are not independent; they may reason more rigorously than people; and they have no real stakes. Use them to screen ideas, then confirm with people."),
    ("What would you do next?", "Try more models, vary temperature, shorten the memory, hide the demand distribution, test a prompt that spells out that O and C pay the same, and compare with human data collected under exactly the same wording."),
]


def qa_tab():
    st.markdown("<div class='lead'>Likely questions from the audience, with short answers.</div>", unsafe_allow_html=True)
    st.write("")
    for q, a in QA:
        with st.expander(q):
            st.markdown(a)


# ----------------------------------------------------------------- checklist
def checklist_tab():
    st.markdown("<div class='lead'>Everything the project brief asks for, and where it is covered.</div>", unsafe_allow_html=True)
    st.write("")
    box = card("Deliverables")
    dl = pd.DataFrame([
        ("A  Source code", "sim/ (game engine, agent setup, prompt variants), langchain_sim/ (LangGraph agents: Claude API and Ollama), analysis/ (statistics), prompts/ (A01–A50 plus _v1 to _v3), streamlit_app/ (this dashboard), README.md, requirements.txt", "Done"),
        ("B  AI-generated dataset", "data/ai_dataset_all_long.csv (10,000 decisions) and ai_dataset_all_participants.csv (400 agents), the per-run ai_agent_*.csv files, analysis/results_summary.csv and results_vs_base.csv; DATA_DICTIONARY.md explains every variable and the collection procedure", "Done"),
        ("C  Short report (maximum 2 pages)", "Docs artifact: https://claude.ai/artifact/18vo23PDQQwnmjZBaW3GV6", "Export to PDF and confirm it is two pages"),
        ("D  Group presentation", "This dashboard (Present tab). Backup slides: https://claude.ai/artifact/SrQjhDbhZHyCnbpY4F2YAm", "Rehearse with the timer"),
    ], columns=["Deliverable", "Where it is", "Status"])
    box.dataframe(dl, hide_index=True)

    box = card("Required tasks")
    tk = pd.DataFrame([
        ("1  Understand the original study", "Present steps 1–3 (business problem, design, human results, theory)"),
        ("2  Design and generate AI agents", "AI system tab, sections 1, 3 and 5 (models, generation, prompts, information, memory, settings)"),
        ("3  Simulate the original experiment", "AI system tab, section 2 (sample size, instructions, rounds, rules, data collection, and every modification with its reason)"),
        ("4  Save and analyse the AI data", "AI system tab sections 4 and 6 (validation and dataset), Results tab (statistics)"),
        ("5  Compare AI with humans", "Results tab and Present steps 7–9 (similar patterns? same bias? small or large difference? why?)"),
    ], columns=["Task", "Where it is covered"])
    box.dataframe(tk, hide_index=True)

    box = card("Grading criteria (100 points) and the planned talk", "Minutes are for the 10-minute plan; use the Present tab switch for 12 minutes.")
    mins = {}
    for s in STEPS:
        mins[s[3]] = mins.get(s[3], 0) + s[2]
    where = {"bg": "Present step 1", "orig": "Present steps 2–3", "ai": "Present steps 5–6, AI system tab", "cmp": "Present steps 7–9, Results and Prompt wording tabs", "qual": "Whole talk: tables, figures and a live game; step 10 for conclusions"}
    desc = {"bg": "Business context, research question and why it matters", "orig": "Human results, behavioural biases and relevant theory", "ai": "Agent generation, prompts, settings, implementation and data collection",
            "cmp": "Meaningful quantitative comparison and whether AI reproduces the biases", "qual": "Clear and logical, with appropriate tables, figures and evidence"}
    rows = [(RUBRIC[k][0], RUBRIC[k][1], desc[k], where[k], fmt_t(mins.get(k, 0) + (mins.get("conc", 0) if k == "qual" else 0))) for k in ("bg", "orig", "ai", "cmp", "qual")]
    rows.append(("Time management", 10, "Finish within the allotted time", "Timer and pace indicator on the Present tab", "10:00 total"))
    box.dataframe(pd.DataFrame(rows, columns=["Criterion", "Points", "What graders look for", "Where it is covered", "Planned time"]), hide_index=True)
    box.caption("The brief gives two different time limits: 12 minutes plus 3 for Q&A in the presentation section, but 10 plus 2 in the grading table. Confirm with the instructor and set the talk length on the Present tab.")

    box = card("Before presenting")
    for i, item in enumerate(["Confirm the talk length with the instructor (10+2 or 12+3)", "Export the report to PDF and check it is two pages or fewer", "Run the dashboard on the presentation computer and click through all ten steps",
                              "Test the Game and AI replay tabs once", "Push the code to GitHub without the .env file", "Have the backup slides open in case the app does not start"]):
        box.checkbox(item, key=f"todo{i}")


# ----------------------------------------------------------------- layout
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class='hero'><h1>Payment Framing Lab</h1>
      <p>Same profit, different wording. Can AI agents reproduce the bias people show? Present the study, play the game, watch AI agents play it, and compare everyone.</p>
      <span class='chip o'>O · OWN FINANCING</span><span class='chip c'>C · CUSTOMER FINANCING</span><span class='chip n'>Chen, Kök &amp; Tong (2013) replication with AI agents</span></div>""", unsafe_allow_html=True)
    names = ["Present", "Game", "AI system", "AI replay", "Results", "Prompt wording"]
    funcs = [present_tab, game_tab, system_tab, replay_tab, results_tab, wording_tab]
    if PRESENTER:
        names += ["Q&A", "Checklist"]; funcs += [qa_tab, checklist_tab]
    for tab, fn in zip(st.tabs(names), funcs):
        with tab:
            fn()
    st.markdown("<div class='footer'>MBAN 5640 · Group Project 3 · Replication of Chen, Kök &amp; Tong (2013) with AI agents</div>", unsafe_allow_html=True)


main()
