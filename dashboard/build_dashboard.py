#!/usr/bin/env python3
"""Embed human + AI-agent results into dashboard/template.html -> dashboard/index.html. Run after analysis/analyze.py."""
import json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT/"analysis/results.json").read_text())
tab = {r["sample"]: r for r in R["table"]}
def hum(fn):
    d = pd.read_csv(ROOT/"data/human"/fn, encoding="utf-8-sig"); d = d[d.ComprehensionFail==0]
    oc = [f"order{i}" for i in range(1,26)]; d = d.assign(avg=d[oc].mean(axis=1)); return d, oc
frames=[hum(f)[0] for f in ["ChenEtAl_OnlineReplication_Mich.csv","ChenEtAl_OnlineReplication_UTD.csv","ChenEtAl_InPersonReplication_Mich.csv"]]
oc=[f"order{i}" for i in range(1,26)]; H=pd.concat(frames)
def grp(df, cond, col="avg"):
    g=df[df.Condition==cond]; return dict(mean=float(g[col].mean()), avgs=[round(float(x),3) for x in g[col]], byDay=[round(float(x),3) for x in g[oc].mean()])
hp={c:grp(H,c) for c in "OC"}
A=pd.read_csv(ROOT/"data/ai_agent_wide.csv"); A=A.assign(avg=A[oc].mean(axis=1))
ai={"byCond":{c:dict(n=int((A.Condition==c).sum()), **grp(A,c)) for c in "OC"}}
def row(label,key):
    t=tab[key]; return dict(sample=label,n_O=int(t["n_O"]),n_C=int(t["n_C"]),mean_O=t["mean_O"],mean_C=t["mean_C"],diff=t["diff"],d=t.get("cohen_d"),p=t.get("p_ols"))
samples=[row("Original (Duke)","Original (Duke) *"),row("UM online","UM online"),row("UTD online","UTD online"),row("UM in person","UM in-person")]
aiKey="AI agents (all 50)"; aiRow=row("Claude agents",aiKey)

# ---- second model (Llama via LangGraph/Ollama) and replay samples ----
RL = json.loads((ROOT/"analysis/results_langchain.json").read_text())
LT = RL["table"][-1]
AL = pd.read_csv(ROOT/"data/ai_agent_wide_langchain.csv"); AL = AL.assign(avg=AL[oc].mean(axis=1))
llama = {"byCond":{c:dict(n=int((AL.Condition==c).sum()), **grp(AL,c)) for c in "OC"}}
llamaRow = dict(sample="Llama agents", n_O=int(LT["n_O"]), n_C=int(LT["n_C"]), mean_O=LT["mean_O"], mean_C=LT["mean_C"], diff=LT["diff"], d=LT.get("cohen_d"), p=LT.get("p_ols"))
llamaExtra = dict(fail=int(RL["checks"]["comprehension_fail"]), share1011=RL["extra"]["share_orders_equal_10_or_11"], early=[RL["extra"]["order_by_half"]["O_days1_5"],RL["extra"]["order_by_half"]["C_days1_5"]], late=[RL["extra"]["order_by_half"]["O_days6_25"],RL["extra"]["order_by_half"]["C_days6_25"]])
def replay(model, longf, partf, cond, passed_only):
    L = pd.read_csv(ROOT/"data"/longf); P = pd.read_csv(ROOT/"data"/partf)
    P = P[(P.condition==cond)]
    if passed_only: P = P[P.comprehension_fail==0]
    a = P.iloc[0]; r = L[L.agent_id==a.agent_id].sort_values("day")
    return dict(model=model, id=a.agent_id, cond=cond, persona=a.persona, strategy=str(a.strategy)[:260], comprehensionFail=int(a.comprehension_fail),
                rounds=[dict(q=int(x.order), d=[int(x.d1),int(x.d2),int(x.d3)], demand=int(x.demand), change=int(x.change), cash=int(x.cash_end)) for x in r.itertuples()])
replays = [replay("Claude","ai_agent_long.csv","ai_agent_participants.csv","O",True), replay("Claude","ai_agent_long.csv","ai_agent_participants.csv","C",True),
           replay("Llama","ai_agent_long_langchain.csv","ai_agent_participants_langchain.csv","O",False), replay("Llama","ai_agent_long_langchain.csv","ai_agent_participants_langchain.csv","C",False)]
v=(ROOT/"analysis/verdict.json"); V=json.loads(v.read_text())
chk=R["checks"]
DATA=dict(llama=llama,llamaRow=llamaRow,llamaExtra=llamaExtra,replays=replays,samples=samples,humanPooled=hp,ai=ai,aiRow=aiRow,headline=V["headline"],verdict=V["verdict"],
  method=dict(agents=V["m_agents"],engine=V["m_engine"],
    checks=f"{chk['n_agents']} agents x 25 days = {chk['n_rows']} orders. Orders out of range: {chk['orders_out_of_range']}. Missing values: {chk['missing']}. Payoff-rule violations: {chk['payoff_rule_violations']}. Demand mismatches: {chk['demand_mismatch']}. Failed comprehension check: {chk['comprehension_fail']}. Mean simulated demand {chk['dice_mean_demand']} (theory 10.5).",
    limits=V["m_limits"]))
t=(ROOT/"dashboard/template.html").read_text().replace("/*DATA*/null", json.dumps(DATA))
(ROOT/"dashboard/index.html").write_text(t); print("built", len(t)//1024, "KB")
