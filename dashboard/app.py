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
    band = alt.Chart(pd.DataFrame({"a": [10], "b": [11]})).mark_rect(opacity=0.13, color=OPT).encode(x="a:Q", x2="b:Q")
    rule = alt.Chart(df).mark_rule(color="#5A6B7A", strokeWidth=2).encode(y=alt.Y("label:N", sort=order, title=None), x=alt.X("C:Q", scale=alt.Scale(domain=[8.5, 14]), title="Mean order (units)"), x2="O:Q")
    pts = alt.Chart(long).mark_circle(size=170, opacity=1).encode(
        y=alt.Y("label:N", sort=order, title=None), x="mean:Q",
        color=alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL]), legend=alt.Legend(title="Scheme")),
        tooltip=["label", "scheme", alt.Tooltip("mean:Q", format=".2f"), "gap"])
    return (band + rule + pts).properties(height=44 * len(df) + 40)


def by_day_chart(series):
    rows = []
    for name, df, cond in series:
        g = grp(df, cond)[OC].mean()
        for i, v in enumerate(g.values, 1):
            rows.append(dict(series=name, scheme=cond, day=i, mean=float(v)))
    d = pd.DataFrame(rows)
    return alt.Chart(d).mark_line(strokeWidth=2.2).encode(
        x=alt.X("day:Q", title="Day"), y=alt.Y("mean:Q", scale=alt.Scale(domain=[8, 15]), title="Mean order"),
        color=alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL])),
        strokeDash=alt.StrokeDash("series:N"), tooltip=["series", "scheme", "day", alt.Tooltip("mean:Q", format=".2f")]).properties(height=280)


def strips(groups):
    rows = [dict(group=g, scheme=c, avg=float(v)) for g, c, vals in groups for v in vals]
    d = pd.DataFrame(rows)
    order = [g for g, _, _ in groups]
    base = alt.Chart(d).encode(x=alt.X("group:N", sort=order, title=None))
    pts = base.mark_circle(size=40, opacity=0.55).encode(y=alt.Y("avg:Q", scale=alt.Scale(domain=[5, 17]), title="Participant average order"),
        color=alt.Color("scheme:N", scale=alt.Scale(domain=["O", "C"], range=[O_COL, C_COL])), xOffset="jitter:Q") \
        .transform_calculate(jitter="random()")
    means = base.mark_tick(color="black", thickness=3, size=60).encode(y="mean(avg):Q")
    rule = alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y="y:Q")
    return (pts + means + rule).properties(height=300)


def profit_curve():
    pm = np.zeros(19)
    for a in range(1, 7):
        for b in range(1, 7):
            for c in range(1, 7):
                pm[a + b + c] += 1 / 216
    ep = lambda q: sum(pm[d] * (2 * min(q, d) - q) for d in range(3, 19))
    d = pd.DataFrame(dict(q=range(3, 19), profit=[ep(q) for q in range(3, 19)]))
    return alt.Chart(d).mark_line(point=True, color="black").encode(x=alt.X("q:Q", title="Order quantity"), y=alt.Y("profit:Q", title="Expected profit per day"), tooltip=["q", alt.Tooltip("profit:Q", format=".2f")]).properties(height=240)


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
    return t.round(3)


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
    st.title("Can AI agents reproduce payment-framing bias?")
    st.markdown("##### Two ways to word the same newsvendor game. People order more under one of them. Do AI agents?")
    a, b, c, d = st.columns(4)
    a.metric("Claude agents", D["claude"]["wide"].shape[0] if "claude" in D else "–"); b.metric("Llama agents", D["llama"]["wide"].shape[0] if "llama" in D else "–")
    c.metric("Rounds each", 25); d.metric("Human samples", 4)


def step_background():
    st.header("Same payoffs, different framing")
    a, b = st.columns(2)
    a.info("**O · Own financing**  \nPay $1 per unit when you order. Receive $2 per unit sold.")
    b.success("**C · Customer financing**  \nReceive $1 per unit when you order. Pay $2 per unit unsold.")
    st.latex(r"\Delta_O = 2\min(q,D) - q \qquad \Delta_C = q - 2\max(q-D,0)")
    st.markdown("Demand is three dice, so the best order is **10 or 11**. **Mental accounting:** in O the cost is paid before demand is known, so it looms as a sure loss and people order more (Chen, Kök & Tong, 2013).")


def step_live():
    st.header("Live demo: you are the participants")
    st.markdown("Open the app address on your phone or laptop, go to **Live game**, and play. You are randomly given O or C.")
    live_box()
    st.caption("Share the Network URL that Streamlit prints when you start the app (same Wi-Fi), or deploy to Streamlit Community Cloud.")


def step_human():
    st.header("Humans order more under O, every time")
    st.altair_chart(dumbbell(rows_for(False)[:4]), use_container_width=True)
    S = D["samples"]
    st.markdown("Gap O − C: " + ", ".join(f"{s['diff']:+.2f}" for s in S[:4]) + " units. People are scattered: spread about 1.7 units.")


def step_design():
    st.header("AI design: simulated participants")
    a, b, c, d = st.columns(4)
    a.markdown("**1 Persona**  \nAge 18–34, gender, major, degree level"); b.markdown("**2 Instructions**  \nOriginal O or C text, word for word")
    c.markdown("**3 Game engine**  \nValidates orders, rolls real dice after the order"); d.markdown("**4 Memory**  \nEach agent remembers all earlier days")
    st.markdown("Claude Sonnet sub-agents (n=50) and a **LangGraph** agent on a free local **Llama 3.1 8B** via Ollama (n=50). Same prompts, same engine.")
    st.code("decide (structured output: order 3–18)  →  play (engine rolls dice)  →  loop 25 days  →  debrief", language="text")


def step_replay_pointer():
    st.header("Live demo: watch an AI agent play")
    st.markdown("Open the **AI replay** tab and replay a real agent day by day, then read its own strategy statement.")


def step_compare():
    st.header("Human versus AI")
    st.altair_chart(dumbbell(rows_for(True)), use_container_width=True)
    a, b, c, d = st.columns(4)
    cw = D["claude"]["wide"]; lw = D["llama"]["wide"] if "llama" in D else None
    a.metric("Claude gap", f"{grp(cw,'O').avg.mean() - grp(cw,'C').avg.mean():+.2f}")
    if lw is not None:
        b.metric("Llama gap", f"{grp(lw,'O').avg.mean() - grp(lw,'C').avg.mean():+.2f}")
    cl = D["claude"]["long"]; c.metric("Claude orders in 10–11", f"{cl.order.isin([10, 11]).mean():.0%}")
    if "llama" in D:
        d.metric("Llama orders in 10–11", f"{D['llama']['long'].order.isin([10, 11]).mean():.0%}")


def step_why():
    st.header("What drives each order?")
    a, b = st.columns(2)
    a.markdown("**Humans**  \nAnchor on their own last order (mean |Shapley| 1.18). Framing explains about 17% of explained variation. Cross-validated R² 0.37.")
    b.markdown("**Claude agents**  \nReact more to the frame (about 39%) and, with history, to last demand and profit. R² 0.50, up to 0.70 with history.")
    xf = ROOT / "analysis/fig_xai.png"
    if xf.exists():
        st.image(str(xf), use_container_width=True)
    if "llama_checks" in D:
        st.markdown(f"**Llama:** {D['llama_checks']['comprehension_fail']} of 50 failed the comprehension check; orders drift upward through the game.")


def step_conclusion():
    st.header("AI gets the direction, not the size")
    st.markdown("Claude shows the O > C gap at a third to a half of human size, with far less variation. Llama shows a gap for the wrong reasons. Use AI to test whether a framing effect exists, then confirm with people.")
    st.markdown("**Limitations**\n- The 50 agents in a model are not independent people\n- Simple personas, no stakes, silent reasoning\n- Agents were told demand is the sum of three dice\n- Results depend on the model (Claude vs Llama)")
    st.subheader("Questions?")


STEPS = [("Question", 30, step_question), ("Background", 90, step_background), ("Live demo: class", 120, step_live), ("Human results", 60, step_human),
         ("AI design", 90, step_design), ("Live demo: AI", 90, step_replay_pointer), ("Comparison", 120, step_compare), ("Why", 60, step_why), ("Conclusion", 60, step_conclusion)]


def present_tab():
    ss = st.session_state
    ss.setdefault("step", 0); ss.setdefault("t0", None)
    total = sum(s[1] for s in STEPS)
    top = st.columns([6, 2, 2])
    top[0].progress((ss.step + 1) / len(STEPS), text=f"Step {ss.step + 1} of {len(STEPS)}: {STEPS[ss.step][0]} (about {STEPS[ss.step][1] // 60}:{STEPS[ss.step][1] % 60:02d})")
    if top[1].button("Start timer" if ss.t0 is None else "Reset timer"):
        ss.t0 = time.time() if ss.t0 is None else None
    if ss.t0:
        e = int(time.time() - ss.t0); top[2].markdown(f"**{e // 60}:{e % 60:02d}** / {total // 60}:{total % 60:02d}")
    STEPS[ss.step][2]()
    st.divider()
    a, b, _ = st.columns([1, 1, 8])
    if a.button("Back", disabled=ss.step == 0):
        ss.step -= 1; st.rerun()
    if b.button("Next", type="primary", disabled=ss.step == len(STEPS) - 1):
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
            st.subheader("Run a widget shop")
            st.markdown("You get a random pay scheme. Both schemes give the same profit for any order and any demand, but they are worded differently.")
            n = st.selectbox("Length", [25, 10], format_func=lambda x: "25 days (original)" if x == 25 else "10 days (quick)")
            if st.button("Start game", type="primary"):
                cond = forced if forced in ("O", "C") else RNG.choice(["O", "C"])
                ss.g = dict(cond=cond, N=n, day=1, cash=100, hist=[], last=None); st.rerun()
            st.caption("Presenter: add ?cond=O or ?cond=C to the address to force a scheme.")
        with c2:
            st.subheader("Class so far"); live_box()
        return
    c = g["cond"]
    left, right = st.columns([3, 2])
    with left:
        finished = len(g["hist"]) >= g["N"]
        h = st.columns(3); h[0].metric("Day", f"{min(g['day'], g['N'])} / {g['N']}"); h[1].metric("Play dollars", g["cash"]); h[2].markdown(f"**{'OWN FINANCING' if c == 'O' else 'CUSTOMER FINANCING'}**")
        if g["last"]:
            L = g["last"]; st.markdown(f"<div style='font-size:56px'>{' '.join(FACES[x - 1] for x in L['d'])}</div>", unsafe_allow_html=True)
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
            ch = alt.Chart(d).mark_line(point=True, color=O_COL if c == "O" else C_COL).encode(x=alt.X("day:Q", scale=alt.Scale(domain=[1, g["N"]])), y=alt.Y("order:Q", scale=alt.Scale(domain=[3, 18])))
            st.altair_chart(ch + alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y="y:Q"), use_container_width=True)


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
    day = st.session_state.get("rday", 0)
    ph = st.empty(); chart_ph = st.empty()

    def draw(n):
        sub = L[L.day <= n]
        with ph.container():
            if n == 0:
                st.info("Press Play, or drag the slider, to step through the agent’s 25 days.")
            else:
                r = sub.iloc[-1]
                m = st.columns(4); m[0].metric("Day", f"{n} / 25"); m[1].metric("Play dollars", int(r.cash_end)); m[2].metric("Agent ordered", int(r.order)); m[3].metric("Change", f"{int(r.change):+d}")
                st.markdown(f"<div style='font-size:56px'>{' '.join(FACES[int(x) - 1] for x in (r.d1, r.d2, r.d3))}</div>", unsafe_allow_html=True)
                st.info(outcome(scheme, int(r.order), int(r.demand)))
        if n:
            d = sub[["day", "order"]]
            ch = alt.Chart(d).mark_line(point=True, color=O_COL if scheme == "O" else C_COL).encode(x=alt.X("day:Q", scale=alt.Scale(domain=[1, 25])), y=alt.Y("order:Q", scale=alt.Scale(domain=[3, 18])))
            chart_ph.altair_chart(ch + alt.Chart(pd.DataFrame({"y": [10.5]})).mark_rule(strokeDash=[4, 3], color=OPT).encode(y="y:Q"), use_container_width=True)
        else:
            chart_ph.empty()

    play = st.button("Play replay", type="primary")
    speed = st.select_slider("Speed", ["slow", "normal", "fast"], value="normal")
    delay = {"slow": 1.2, "normal": 0.7, "fast": 0.25}[speed]
    n = st.slider("Day", 0, 25, day, key="rday")
    draw(n)
    if play:
        for i in range(1, 26):
            draw(i); time.sleep(delay)
    if n == 25 or play:
        st.markdown("**In the agent’s own words:**")
        st.write(f"“{a.strategy}”")
        st.caption("This agent failed the end-of-game comprehension question." if int(a.comprehension_fail) else "This agent passed the comprehension question.")


# ----------------------------------------------------------------- results / why / method
def results_tab():
    V = D["verdict"]
    if V:
        st.success(f"**{V.get('headline', '')}**  \n{V.get('verdict', '')}")
    st.subheader("Mean order by scheme")
    st.altair_chart(dumbbell(rows_for(True)), use_container_width=True)
    c1, c2 = st.columns(2)
    H = D["human"]; groups = [("Human O", "O", grp(H, "O").avg), ("Human C", "C", grp(H, "C").avg)]
    for m, nm in (("claude", "Claude"), ("llama", "Llama")):
        if m in D:
            groups += [(f"{nm} O", "O", grp(D[m]["wide"], "O").avg), (f"{nm} C", "C", grp(D[m]["wide"], "C").avg)]
    c1.subheader("Each participant’s average"); c1.altair_chart(strips(groups), use_container_width=True)
    cs = class_stats(); c2.subheader("Class distribution")
    c2.altair_chart(strips([("Class O", "O", cs["O"]["avgs"] or [0]), ("Class C", "C", cs["C"]["avgs"] or [0])]), use_container_width=True) if cs["total"] else c2.info("No class plays yet. Ask the room to play.")
    st.subheader("Orders by day")
    ser = [("Humans", D["human"], "O"), ("Humans", D["human"], "C")]
    for m, nm in (("claude", "Claude"), ("llama", "Llama")):
        if m in D:
            ser += [(nm, D[m]["wide"], "O"), (nm, D[m]["wide"], "C")]
    st.altair_chart(by_day_chart(ser), use_container_width=True)
    st.subheader("What a wrong order costs"); st.altair_chart(profit_curve(), use_container_width=True)
    st.subheader("Statistics"); st.dataframe(stats_table(), use_container_width=True, hide_index=True)
    st.caption("Unit of analysis is the participant’s 25-day average (two-sample t-test). Original-study row is from the report.")


def why_tab():
    a, b = st.columns(2)
    a.subheader("The two schemes are the same game")
    a.markdown("Ordering q when demand is D gives 2·min(q,D) − q in O and q − 2·max(q−D,0) in C. These are equal for every q and D. With demand from three dice, the best order is 10 or 11 under either scheme.")
    b.subheader("Mental accounting")
    b.markdown("People split payments into “before demand” and “after demand” accounts and underweight the first. In O the order-time payment is a cost and orders rise. In C it is a gain and the refund risk feels larger, so orders fall.")
    st.subheader("Business link")
    st.markdown("Supply chains use these schemes: buyer-financed orders (O), trade credit, and customer prepayment (C). The same economics can lead to different stocking.")
    st.caption("Chen, L., Kök, A. G., & Tong, J. D. (2013). The effect of payment schemes on inventory decisions: The role of mental accounting. Management Science, 59(2), 436–451.")


def method_tab():
    V = D["verdict"]; a, b = st.columns(2)
    a.subheader("Agents"); a.write(V.get("m_agents", ""))
    b.subheader("Game engine"); b.write(V.get("m_engine", ""))
    k = D["checks"]
    a.subheader("Data checks")
    a.write(f"{k['n_agents']} agents × 25 days = {k['n_rows']} orders. Out of range: {k['orders_out_of_range']}. Missing: {k['missing']}. Payoff-rule violations: {k['payoff_rule_violations']}. Demand mismatches: {k['demand_mismatch']}. Failed comprehension check: {k['comprehension_fail']}.")
    b.subheader("Limits"); b.write(V.get("m_limits", ""))
    if "llama_checks" in D:
        st.subheader("Second model: Llama 3.1 8B")
        st.write(f"Same game, personas and instructions run through a LangGraph agent (decide, play, loop, debrief; structured output; message-history memory) on a free local Llama via Ollama. {D['llama_checks']['comprehension_fail']} of 50 agents failed the comprehension check, so its gap reflects weak task understanding more than human-like bias.")


# ----------------------------------------------------------------- layout
st.markdown("<h2 style='margin-bottom:0'>Payment Framing Lab</h2>", unsafe_allow_html=True)
t = st.tabs(["Present", "Live game", "AI replay", "Humans vs AI", "Why it matters", "AI method"])
with t[0]: present_tab()
with t[1]: game_tab()
with t[2]: replay_tab()
with t[3]: results_tab()
with t[4]: why_tab()
with t[5]: method_tab()