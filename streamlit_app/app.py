"""Payment Framing Lab (Streamlit). Presenter mode, live class game, AI replay, Humans vs AI.

Run:  streamlit run streamlit_app/app.py
Reads the project's own files (data/, analysis/), so re-running the analysis updates this app.
Class plays are shared through data/class_plays.json (everyone connected to the same running app sees them).
"""
import json, os, secrets, tempfile, threading, time
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
LOCK = threading.Lock()
CLASS_FILE = DATA / "class_plays.json"

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
.stTabs [data-baseweb="tab-list"] {gap:6px; border-bottom:1px solid #D5DCE3;}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:1rem; padding:10px 18px; border-radius:10px 10px 0 0;}
.stTabs [aria-selected="true"] {background:#14213D; color:#fff !important;}
.stTabs [aria-selected="true"] p {color:#fff !important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px;}
.stButton button {border-radius:10px; font-weight:600;}
.stButton button[kind="primary"] {background:#14213D; border-color:#14213D;}
.card-o {border-left:6px solid #C8681A; background:#fff; border-radius:12px; padding:16px 20px; font-size:1.1rem; box-shadow:0 1px 2px rgba(20,33,61,.06);}
.card-c {border-left:6px solid #17808A; background:#fff; border-radius:12px; padding:16px 20px; font-size:1.1rem; box-shadow:0 1px 2px rgba(20,33,61,.06);}
.diceline {font-size:64px; line-height:1.1; letter-spacing:10px; margin:4px 0 8px 0;}
.kick {font:600 .75rem 'IBM Plex Mono',monospace; letter-spacing:.12em; text-transform:uppercase; color:#5A6B7A; margin-bottom:6px;}
.big {font-family:'Bricolage Grotesque',sans-serif; font-size:2.4rem; font-weight:700; letter-spacing:-.02em; line-height:1.1; margin:0 0 .6rem 0; color:#14213D;}
.sec {font:700 1.15rem 'Bricolage Grotesque',sans-serif; color:#14213D; margin:0 0 2px 0;}
.cap {font-size:.88rem; color:#5A6B7A; margin-bottom:8px;}
.verdict {background:#fff; border:1px solid #D5DCE3; border-left:6px solid #6B5BB5; border-radius:12px; padding:16px 20px; margin:14px 0; font-size:1.02rem; line-height:1.55;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; box-shadow:0 1px 3px rgba(20,33,61,.06);}
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
    out = dict(samples=samples, human=H, checks=R["checks"], extra=R["extra"])
    for tag, key in [("", "claude"), ("_langchain", "llama")]:
        wf = DATA / f"ai_agent_wide{tag}.csv"
        if wf.exists():
            W = pd.read_csv(wf); W = W.assign(avg=W[OC].mean(axis=1))
            out[key] = dict(wide=W, long=pd.read_csv(DATA / f"ai_agent_long{tag}.csv"), part=pd.read_csv(DATA / f"ai_agent_participants{tag}.csv"))
    rl = ROOT / "analysis/results_langchain.json"
    if rl.exists():
        RL = json.loads(rl.read_text()); t = RL["table"][-1]
        out["llama_row"] = dict(sample="Llama agents", n_O=int(t["n_O"]), n_C=int(t["n_C"]), mean_O=t["mean_O"], mean_C=t["mean_C"], diff=t["diff"], d=t.get("cohen_d"), p=t.get("p_ols"))
        out["llama_extra"] = RL["extra"]; out["llama_checks"] = RL["checks"]
    vf = ROOT / "analysis/verdict.json"
    out["verdict"] = json.loads(vf.read_text()) if vf.exists() else {}
    return out


D = load_all()


def grp(df, cond):
    return df[df.Condition == cond]


def load_plays():
    with LOCK:
        try:
            return json.loads(CLASS_FILE.read_text())
        except Exception:
            return {}


def save_play(sid, cond, orders, done):
    with LOCK:
        try:
            plays = json.loads(CLASS_FILE.read_text())
        except Exception:
            plays = {}
        plays[sid] = dict(cond=cond, orders=orders, done=done, ts=time.time())
        fd, tmp = tempfile.mkstemp(dir=str(DATA), suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(plays, fh)
        os.replace(tmp, CLASS_FILE)


def class_stats():
    plays = [p for p in load_plays().values() if p.get("orders")]
    out = {}
    for c in "OC":
        avgs = [float(np.mean(p["orders"])) for p in plays if p["cond"] == c]
        out[c] = dict(n=len(avgs), mean=float(np.mean(avgs)) if avgs else None, avgs=avgs)
    out["total"] = len(plays); out["done"] = sum(1 for p in plays if p.get("done"))
    return out


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


def rows_for(include_class=True):
    S = D["samples"]
    rows = [dict(label="Original (Duke)", O=S[0]["mean_O"], C=S[0]["mean_C"])]
    for s in S[1:4]:
        rows.append(dict(label=s["sample"], O=s["mean_O"], C=s["mean_C"]))
    if "claude" in D:
        w = D["claude"]["wide"]; rows.append(dict(label="Claude agents", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    if "llama" in D:
        w = D["llama"]["wide"]; rows.append(dict(label="Llama agents", O=grp(w, "O").avg.mean(), C=grp(w, "C").avg.mean()))
    if include_class:
        cs = class_stats(); rows.append(dict(label=f"This class (n={cs['O']['n'] + cs['C']['n']})", O=cs["O"]["mean"], C=cs["C"]["mean"]))
    return rows


def stats_table():
    t = pd.DataFrame(D["samples"] + ([D["llama_row"]] if "llama_row" in D else []))
    t = t.rename(columns={"sample": "Sample", "n_O": "n O", "n_C": "n C", "mean_O": "Mean O", "mean_C": "Mean C", "diff": "O − C", "d": "Cohen d", "p": "p"})
    t["Sample"] = t["Sample"].str.replace("AI agents", "Claude agents")
    for col in ("Mean O", "Mean C", "O − C"):
        t[col] = t[col].map(lambda v: f"{v:.2f}")
    t["Cohen d"] = t["Cohen d"].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f}")
    t["p"] = t["p"].map(lambda v: "–" if v is None or pd.isna(v) else ("<0.001" if v < 0.001 else f"{v:.3f}"))
    return t


# ----------------------------------------------------------------- live fragment
fragment = getattr(st, "fragment", None)


def live_box_body():
    cs = class_stats()
    a, b, c, d = st.columns(4)
    a.metric("Players", cs["total"]); b.metric("In O", cs["O"]["n"]); c.metric("In C", cs["C"]["n"])
    gap = (cs["O"]["mean"] - cs["C"]["mean"]) if cs["O"]["n"] and cs["C"]["n"] else None
    d.metric("Class gap O − C", "–" if gap is None else f"{gap:+.2f}")
    st.caption(f"{cs['done']} finished. O mean {cs['O']['mean']:.1f}" if cs["O"]["n"] else f"{cs['done']} finished.")


live_box = fragment(run_every=3)(live_box_body) if fragment else live_box_body


# ----------------------------------------------------------------- presenter steps
def step_question():
    st.markdown("<div class='big'>Can AI agents reproduce payment-framing bias?</div>", unsafe_allow_html=True)
    st.markdown("<div class='lead'>Two ways to word the same newsvendor game. People order more under one of them. Do AI agents?</div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Claude agents", D["claude"]["wide"].shape[0] if "claude" in D else "–"); b.metric("Llama agents", D["llama"]["wide"].shape[0] if "llama" in D else "–")
    c.metric("Rounds each", 25); d.metric("Human samples", 4)


def step_background():
    st.markdown("<div class='big'>Same payoffs, different framing</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown("<div class='card-o'><b>O · Own financing</b><br>Pay $1 per unit when you order. Receive $2 per unit sold.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card-c'><b>C · Customer financing</b><br>Receive $1 per unit when you order. Pay $2 per unit unsold.</div>", unsafe_allow_html=True)
    st.latex(r"\Delta_O = 2\min(q,D) - q \qquad \Delta_C = q - 2\max(q-D,0)")
    st.markdown("Demand is three dice, so the best order is **10 or 11**. **Mental accounting:** in O the cost is paid before demand is known, so it looms as a sure loss and people order more (Chen, Kök & Tong, 2013).")


def step_live():
    st.markdown("<div class='big'>Live demo: you are the participants</div>", unsafe_allow_html=True)
    st.markdown("Open the app address on your phone or laptop, go to **Live game**, and play. You are randomly given O or C.")
    live_box()
    st.caption("Share the Network URL that Streamlit prints when you start the app (same Wi-Fi), or deploy to Streamlit Community Cloud.")


def step_human():
    st.markdown("<div class='big'>Humans order more under O, every time</div>", unsafe_allow_html=True)
    _w(st.altair_chart, polish(dumbbell(rows_for(False)[:4])))
    S = D["samples"]
    st.markdown("Gap O − C: " + ", ".join(f"{s['diff']:+.2f}" for s in S[:4]) + " units. People are scattered: spread about 1.7 units.")


def step_design():
    st.markdown("<div class='big'>AI design: simulated participants</div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.markdown("**1 Persona**  \nAge 18–34, gender, major, degree level"); b.markdown("**2 Instructions**  \nOriginal O or C text, word for word")
    c.markdown("**3 Game engine**  \nValidates orders, rolls real dice after the order"); d.markdown("**4 Memory**  \nEach agent remembers all earlier days")
    st.markdown("Claude Sonnet sub-agents (n=50) and a **LangGraph** agent on a free local **Llama 3.1 8B** via Ollama (n=50). Same prompts, same engine.")
    st.code("decide (structured output: order 3–18)  →  play (engine rolls dice)  →  loop 25 days  →  debrief", language="text")


def step_replay_pointer():
    st.markdown("<div class='big'>Live demo: watch an AI agent play</div>", unsafe_allow_html=True)
    st.markdown("Open the **AI replay** tab and replay a real agent day by day, then read its own strategy statement.")


def step_compare():
    st.markdown("<div class='big'>Human versus AI</div>", unsafe_allow_html=True)
    _w(st.altair_chart, polish(dumbbell(rows_for(True))))
    a, b, c, d = st.columns(4)
    cw = D["claude"]["wide"]; lw = D["llama"]["wide"] if "llama" in D else None
    a.metric("Claude gap", f"{grp(cw,'O').avg.mean() - grp(cw,'C').avg.mean():+.2f}")
    if lw is not None:
        b.metric("Llama gap", f"{grp(lw,'O').avg.mean() - grp(lw,'C').avg.mean():+.2f}")
    cl = D["claude"]["long"]; c.metric("Claude orders in 10–11", f"{cl.order.isin([10, 11]).mean():.0%}")
    if "llama" in D:
        d.metric("Llama orders in 10–11", f"{D['llama']['long'].order.isin([10, 11]).mean():.0%}")


def step_why():
    st.markdown("<div class='big'>What drives each order?</div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    a.markdown("**Humans**  \nAnchor on their own last order (mean |Shapley| 1.18). Framing explains about 17% of explained variation. Cross-validated R² 0.37.")
    b.markdown("**Claude agents**  \nReact more to the frame (about 39%) and, with history, to last demand and profit. R² 0.50, up to 0.70 with history.")
    xf = ROOT / "analysis/fig_xai.png"
    if xf.exists():
        _w(st.image, str(xf))
    if "llama_checks" in D:
        st.markdown(f"**Llama:** {D['llama_checks']['comprehension_fail']} of 50 failed the comprehension check; orders drift upward through the game.")


def step_conclusion():
    st.markdown("<div class='big'>AI gets the direction, not the size</div>", unsafe_allow_html=True)
    st.markdown("Claude shows the O > C gap at a third to a half of human size, with far less variation. Llama shows a gap for the wrong reasons. Use AI to test whether a framing effect exists, then confirm with people.")
    st.markdown("**Limitations**\n- The 50 agents in a model are not independent people\n- Simple personas, no stakes, silent reasoning\n- Agents were told demand is the sum of three dice\n- Results depend on the model (Claude vs Llama)")
    st.subheader("Questions?")


STEPS = [("Question", 30, step_question), ("Background", 90, step_background), ("Live demo: class", 120, step_live), ("Human results", 60, step_human),
         ("AI design", 90, step_design), ("Live demo: AI", 90, step_replay_pointer), ("Comparison", 120, step_compare), ("Why", 60, step_why), ("Conclusion", 60, step_conclusion)]


def present_tab():
    ss = st.session_state
    ss.setdefault("step", 0); ss.setdefault("t0", None)
    total = sum(s[1] for s in STEPS)
    cols = st.columns(len(STEPS) + 2)
    for i, (name, _, _) in enumerate(STEPS):
        if btn(cols[i], str(i + 1), key=f"jump{i}", help=name, type="primary" if i == ss.step else "secondary"):
            ss.step = i; st.rerun()
    if btn(cols[-2], "⏱ Start" if ss.t0 is None else "⏱ Reset", key="timer"):
        ss.t0 = time.time() if ss.t0 is None else None
        st.rerun()
    if ss.t0:
        e = int(time.time() - ss.t0); cols[-1].markdown(f"**{e // 60}:{e % 60:02d}** / {total // 60}:{total % 60:02d}")
    else:
        cols[-1].caption(f"plan {total // 60}:{total % 60:02d}")
    name, sec, fn = STEPS[ss.step]
    with st.container(border=True):
        st.markdown(f"<div class='kick'>Step {ss.step + 1} of {len(STEPS)} · {name} · about {sec // 60}:{sec % 60:02d}</div>", unsafe_allow_html=True)
        fn()
    a, b, _ = st.columns([1, 1, 8])
    if btn(a, "← Back", disabled=ss.step == 0):
        ss.step -= 1; st.rerun()
    if btn(b, "Next →", type="primary", disabled=ss.step == len(STEPS) - 1):
        ss.step += 1; st.rerun()


# ----------------------------------------------------------------- live game
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
            st.markdown("You get a random pay scheme. Both schemes give the same profit for any order and any demand, but they are worded differently.")
            n = st.selectbox("Length", [25, 10], format_func=lambda x: "25 days (original)" if x == 25 else "10 days (quick)")
            if st.button("Start game", type="primary"):
                cond = forced if forced in ("O", "C") else RNG.choice(["O", "C"])
                ss.g = dict(cond=cond, N=n, day=1, cash=100, hist=[], last=None); st.rerun()
            st.caption("Presenter: add ?cond=O or ?cond=C to the address to force a scheme.")
        with c2:
            st.markdown("<div class='sec' style='font-size:1.5rem'>Class so far</div>", unsafe_allow_html=True); live_box()
        return
    c = g["cond"]
    left, right = st.columns([3, 2])
    with left:
        finished = len(g["hist"]) >= g["N"]
        h = st.columns(3); h[0].metric("Day", f"{min(g['day'], g['N'])} / {g['N']}"); h[1].metric("Play dollars", g["cash"]); h[2].markdown(f"**{'OWN FINANCING' if c == 'O' else 'CUSTOMER FINANCING'}**")
        if g["last"]:
            L = g["last"]; st.markdown(f"<div class='diceline'>{' '.join(FACES[x - 1] for x in L['d'])}</div>", unsafe_allow_html=True)
            st.info(outcome(c, L["q"], L["demand"]) + f"  \n**{L['change']:+d} play dollars**")
        if finished:
            qs = [r["q"] for r in g["hist"]]
            a, b, d = st.columns(3)
            a.metric("Your average order", f"{np.mean(qs):.1f}")
            hum = grp(D["human"], c).avg.mean(); a2 = grp(D["claude"]["wide"], c).avg.mean() if "claude" in D else None
            b.metric(f"Humans in {c}", f"{hum:.1f}"); d.metric(f"Claude agents in {c}", f"{a2:.1f}" if a2 else "–")
            st.markdown("Under both schemes, expected profit is highest at 10 or 11 units. " + ("You ordered above the optimum." if np.mean(qs) > 11.2 else "You ordered below the optimum." if np.mean(qs) < 9.8 else "You stayed close to the optimum."))
            other = "C" if c == "O" else "O"; lr = g["hist"][-1]
            with st.expander("Show your last day in the other wording"):
                st.write(outcome(other, lr["q"], lr["demand"])); st.write(f"Change: {lr['change']:+d} play dollars, identical to what you got.")
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
                save_play(ss.sid, c, [r["q"] for r in g["hist"]], len(g["hist"]) >= g["N"]); st.rerun()
    with right:
        st.markdown("**Instructions you were given**")
        for r in INSTR[c]:
            st.markdown(f"- {r}")
        st.caption("Start with 100 play dollars. Goal: finish with as many as possible.")
        if g["hist"]:
            d = pd.DataFrame(dict(day=[r["day"] for r in g["hist"]], order=[r["q"] for r in g["hist"]]))
            ch = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=45, color=O_COL if c == "O" else C_COL), color=O_COL if c == "O" else C_COL).encode(x=alt.X("day:Q", title="Day", scale=alt.Scale(domain=[1, g["N"]])), y=alt.Y("order:Q", title="Order", scale=alt.Scale(domain=[3, 18])))
            _w(st.altair_chart, polish(ch + alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y=alt.Y("y:Q", title="Order"))))


# ----------------------------------------------------------------- AI replay
def replay_tab():
    models = [m for m in ("claude", "llama") if m in D]
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Model", models, format_func=lambda m: "Claude (Sonnet sub-agents)" if m == "claude" else "Llama 3.1 8B (LangGraph + Ollama)")
    scheme = c2.selectbox("Scheme", ["O", "C"])
    P = D[model]["part"]; P = P[P.condition == scheme]
    agent = c3.selectbox("Agent", list(P.agent_id), format_func=lambda a: f"{a} (passed check)" if int(P[P.agent_id == a].comprehension_fail.iloc[0]) == 0 else f"{a} (failed check)")
    a = P[P.agent_id == agent].iloc[0]
    L = D[model]["long"]; L = L[L.agent_id == agent].sort_values("day")
    st.markdown(f"**Persona:** {a.persona}  ·  **{'Own financing' if scheme == 'O' else 'Customer financing'}**")
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
        box.caption("This agent failed the end-of-game comprehension question." if int(a.comprehension_fail) else "This agent passed the comprehension question.")


# ----------------------------------------------------------------- results / why / method
def card(title, caption=None):
    """Bordered card with a title; use as `with card("Title"):`."""
    box = st.container(border=True)
    box.markdown(f"<div class='sec'>{title}</div>", unsafe_allow_html=True)
    if caption:
        box.markdown(f"<div class='cap'>{caption}</div>", unsafe_allow_html=True)
    return box


def results_tab():
    V = D["verdict"]
    H = D["human"]; cw = D["claude"]["wide"]; lw = D["llama"]["wide"] if "llama" in D else None
    hg = grp(H, "O").avg.mean() - grp(H, "C").avg.mean()
    cg = grp(cw, "O").avg.mean() - grp(cw, "C").avg.mean()
    cs = class_stats()
    k = st.columns(4)
    k[0].metric("Humans O − C (pooled)", f"{hg:+.2f}")
    k[1].metric("Claude agents O − C", f"{cg:+.2f}")
    k[2].metric("Llama agents O − C", f"{grp(lw, 'O').avg.mean() - grp(lw, 'C').avg.mean():+.2f}" if lw is not None else "–")
    k[3].metric("This class O − C", f"{cs['O']['mean'] - cs['C']['mean']:+.2f}" if cs["O"]["n"] and cs["C"]["n"] else "waiting")
    if V:
        st.markdown(f"<div class='verdict'><b>{V.get('headline', '')}</b><br>{V.get('verdict', '')}</div>", unsafe_allow_html=True)

    box = card("Mean order by payment scheme", "Orange = own financing (O), teal = customer financing (C). Shaded band is the profit-maximizing 10 to 11. Right edge shows O minus C in the tooltip.")
    _w(box.altair_chart, polish(dumbbell(rows_for(True))))

    c1, c2 = st.columns(2)
    groups = [("Human O", "O", grp(H, "O").avg), ("Human C", "C", grp(H, "C").avg)]
    for m, nm in (("claude", "Claude"), ("llama", "Llama")):
        if m in D:
            groups += [(f"{nm} O", "O", grp(D[m]["wide"], "O").avg), (f"{nm} C", "C", grp(D[m]["wide"], "C").avg)]
    with c1:
        box = card("Each participant’s average order", "One dot per person or agent; the black bar is the group mean.")
        _w(box.altair_chart, polish(strips(groups)))
    with c2:
        box = card("This class (live)", "Everyone who plays the Live game is counted here.")
        if cs["total"]:
            _w(box.altair_chart, polish(strips([("Class O", "O", cs["O"]["avgs"]), ("Class C", "C", cs["C"]["avgs"])])))
        else:
            box.info("No class plays yet. Ask the room to play.")

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


def why_tab():
    a, b = st.columns(2)
    with a:
        box = card("The two schemes are the same game")
        box.markdown("Ordering **q** when demand is **D** gives")
        box.latex(r"\Delta_O = 2\min(q,D) - q \qquad \Delta_C = q - 2\max(q-D,0)")
        box.markdown("These are equal for every q and D. With demand from three dice, the best order is **10 or 11** under either scheme.")
    with b:
        box = card("Mental accounting")
        box.markdown("People keep separate mental accounts for payments made **before** and **after** demand is known, and underweight the first. In O the order-time payment is a cost, so orders rise. In C it is a gain and the refund risk feels larger, so orders fall.")
    box = card("Business link")
    box.markdown("Supply chains use these schemes: buyer-financed orders (O), trade credit from suppliers, and customer prepayment (C). The same economics can lead to different stocking decisions.")
    st.caption("Chen, L., Kök, A. G., & Tong, J. D. (2013). The effect of payment schemes on inventory decisions: The role of mental accounting. Management Science, 59(2), 436–451.")


def method_tab():
    V = D["verdict"]; k = D["checks"]
    a, b = st.columns(2)
    with a:
        box = card("Agents"); box.write(V.get("m_agents", ""))
        box = card("Data checks")
        box.write(f"{k['n_agents']} agents × 25 days = {k['n_rows']} orders. Out of range: {k['orders_out_of_range']}. Missing: {k['missing']}. Payoff-rule violations: {k['payoff_rule_violations']}. Demand mismatches: {k['demand_mismatch']}. Failed comprehension check: {k['comprehension_fail']}.")
    with b:
        box = card("Game engine"); box.write(V.get("m_engine", ""))
        box = card("Limits"); box.write(V.get("m_limits", ""))
    if "llama_checks" in D:
        box = card("Second model: Llama 3.1 8B")
        box.write(f"Same game, personas and instructions run through a LangGraph agent (decide, play, loop, debrief; structured output; message-history memory) on a free local Llama via Ollama. {D['llama_checks']['comprehension_fail']} of 50 agents failed the comprehension check, so its gap reflects weak task understanding more than human-like bias.")


# ----------------------------------------------------------------- layout
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class='hero'><h1>Payment Framing Lab</h1>
      <p>Same profit, different wording. Present the study, play the game live with the room, watch AI agents play it, and compare everyone.</p>
      <span class='chip o'>O · OWN FINANCING</span><span class='chip c'>C · CUSTOMER FINANCING</span><span class='chip n'>Chen, Kök &amp; Tong (2013) replication with AI agents</span></div>""", unsafe_allow_html=True)
    tabs = st.tabs(["Present", "Live game", "AI replay", "Humans vs AI", "Why it matters", "AI method"])
    funcs = [present_tab, game_tab, replay_tab, results_tab, why_tab, method_tab]
    for tab, fn in zip(tabs, funcs):
        with tab:
            fn()
    st.markdown("<div class='footer'>MBAN 5640 · Group Project 3 · Replication of Chen, Kök &amp; Tong (2013) with AI agents</div>", unsafe_allow_html=True)


main()
