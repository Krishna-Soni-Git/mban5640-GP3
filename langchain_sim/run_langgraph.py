#!/usr/bin/env python3
"""LangChain + LangGraph version of the Chen, Kok & Tong (2013) newsvendor simulation, using the Claude API.

Concepts used
  ChatPromptTemplate       system prompt = persona + the study's instructions (word for word, from sim/setup_agents.py)
  with_structured_output   Pydantic Decision(order: int 3..18) -> validated decisions, automatic re-ask on bad output
  message history          every day's outcome is fed back, so the agent remembers earlier days
  LangGraph StateGraph     decide -> play -> (loop 25 days) -> debrief
  LangSmith tracing        set LANGSMITH_TRACING=true and LANGSMITH_API_KEY; each agent is one named, tagged run
Payoffs, dice and prompts come from sim/newsvendor_env.py, so rules are identical to the first run.

Setup:   pip install langgraph pydantic python-dotenv  + langchain-anthropic (API) or langchain-ollama (free, local)
         copy .env.example to .env and fill in your keys (or export them in the terminal)   (optional: LANGSMITH_TRACING=true LANGSMITH_API_KEY=... LANGSMITH_PROJECT=newsvendor-ai)
Run:     python langchain_sim/run_langgraph.py --n 50 --temperature 1.0 --workers 5                       (Claude API)
         python langchain_sim/run_langgraph.py --provider ollama --model llama3.1:8b --workers 2          (free, local)
Output:  data/state_langchain/A*.json  (same format as data/state)  -> then
         AI_STATE_DIR=data/state_langchain AI_TAG=_langchain python analysis/analyze.py
"""
import argparse, json, random, sys, concurrent.futures as cf
from pathlib import Path
try:                                   # load keys from <project>/.env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass
from typing import TypedDict, List
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sim"))
import newsvendor_env as env            # play(), prompt(), ROUNDS, QMIN, QMAX
import setup_agents as sa               # INSTR, MAJORS, LEVELS, GENDER
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

class Decision(BaseModel):
    order: int = Field(ge=env.QMIN, le=env.QMAX, description="Units to order today (whole number 3-18)")

class Debrief(BaseModel):
    check_answer: int = Field(description="Change in play dollars if you order 10 and demand is 7 (minus sign if it falls)")
    strategy: str = Field(description="One sentence on how you decided your orders")

SYSTEM = ChatPromptTemplate.from_messages([("system",
"""You are a participant in a paid decision-making experiment at a university behavioral lab ("Supply Chain Ordering Decisions"). You are {persona}. Take part the way that person genuinely would; you are a human participant, not an AI assistant, and nobody will help you with the task.

Thank you for participating in this experiment. This research is very important to us as researchers, so please try your best. You will receive $5 minimum for participating plus a bonus equal to 1 real dollar for every 50 play dollars you earn by the end of the game (rounded to the nearest dollar). You cannot receive a negative bonus, even if you lost money.

=== GAME INSTRUCTIONS ===
{instr}

The demand each day is the sum of three standard six-sided dice. The dice are rolled for you after you place your order. Your order each day can be any whole number from 3 to 18.
=== END OF INSTRUCTIONS ===
Answer each day with only your order quantity.""")])

class S(TypedDict):
    sim: dict          # engine state: id, cond, persona, cash, rounds, debrief
    messages: list     # conversation memory

def build_graph(llm, system_prompt_msgs):
    decider = llm.with_structured_output(Decision)
    debriefer = llm.with_structured_output(Debrief)

    def decide(s: S):
        d = decider.invoke(system_prompt_msgs(s["sim"]) + s["messages"])
        return {"messages": s["messages"] + [AIMessage(content=f"I order {d.order} units.")], "sim": {**s["sim"], "_q": d.order}}

    def play(s: S):
        sim = dict(s["sim"]); q = sim.pop("_q")
        text = env.play(sim, q)                       # rolls dice, applies O/C payoff, appends to sim["rounds"]
        nxt = env.prompt(sim)
        msg = text + "\n\n" + (nxt if nxt else f"The 25 days are over. You finished with {sim['cash']} play dollars.")
        return {"sim": sim, "messages": s["messages"] + [HumanMessage(content=msg)]}

    def debrief(s: S):
        q = HumanMessage(content='Final questions. (1) "On a day when you order 10 units and the demand turns out to be 7, by how many play dollars does your balance change that day?" (2) In one sentence, how did you decide your orders?')
        d = debriefer.invoke(system_prompt_msgs(s["sim"]) + s["messages"] + [q])
        sim = dict(s["sim"]); sim["debrief"] = dict(answer=str(d.check_answer), strategy=d.strategy)
        return {"sim": sim}

    g = StateGraph(S)
    g.add_node("decide", decide); g.add_node("play", play); g.add_node("debrief", debrief)
    g.set_entry_point("decide"); g.add_edge("decide", "play")
    g.add_conditional_edges("play", lambda s: "decide" if len(s["sim"]["rounds"]) < env.ROUNDS else "debrief")
    g.add_edge("debrief", END)
    return g.compile()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--provider", choices=["anthropic","ollama"], default="anthropic")
    ap.add_argument("--model", default=None, help="default: claude-sonnet-5-5 (anthropic) or llama3.1:8b (ollama)")
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default="base", help="instruction wording: base | v1 | v2 | v3 (see sim/prompt_variants.py)")
    a = ap.parse_args()
    from prompt_variants import get_instr, strip_bonus
    instr_map = get_instr(a.variant, sa.INSTR)
    out = Path(a.out or ROOT / "data" / ("state_langchain" + ("" if a.variant == "base" else "_" + a.variant))); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    conds = ["O"] * (a.n // 2) + ["C"] * (a.n - a.n // 2); random.shuffle(conds)
    a.model = a.model or ("llama3.1:8b" if a.provider == "ollama" else "claude-sonnet-5-5")
    if a.provider == "ollama":            # free, local: needs Ollama running and `ollama pull <model>`
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=a.model, temperature=a.temperature, num_predict=300)
    else:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=a.model, temperature=a.temperature, max_tokens=300)
    roster = []
    for i, cond in enumerate(conds, 1):
        persona = (f"a {random.randint(18,34)}-year-old {random.choice(sa.GENDER)} studying {random.choice(sa.MAJORS)} "
                   f"as {random.choice(sa.LEVELS)} student")
        roster.append(dict(id=f"A{i:02d}", cond=cond, persona=persona))

    def sys_msgs(sim):
        msgs = SYSTEM.format_messages(persona=sim["persona"], instr=instr_map[sim["cond"]])
        for m in msgs: m.content = strip_bonus(m.content, a.variant)
        return msgs

    graph = build_graph(llm, sys_msgs)

    def run_one(r):
        sim = dict(id=r["id"], cond=r["cond"], persona=r["persona"], cash=env.START_CASH, rounds=[], debrief=None)
        first = HumanMessage(content=env.prompt(sim))
        res = graph.invoke({"sim": sim, "messages": [first]},
            config={"run_name": f"newsvendor-{r['id']}-{r['cond']}", "tags": [r["cond"], a.model],
                    "metadata": dict(agent_id=r["id"], condition=r["cond"], model=a.model, temperature=a.temperature),
                    "recursion_limit": 200})
        (out / f"{r['id']}.json").write_text(json.dumps(res["sim"], indent=1)); return r["id"]

    (out / "roster.json").write_text(json.dumps(roster, indent=1))
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for done in ex.map(run_one, roster): print("finished", done, flush=True)

if __name__ == "__main__":
    main()
