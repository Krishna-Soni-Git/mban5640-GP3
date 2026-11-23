#!/usr/bin/env python3
"""LangChain + LangGraph version of the Chen, Kok & Tong (2013) newsvendor simulation, using the Claude API.

Concepts used
  ChatPromptTemplate       system prompt = the study's instructions (no persona) (word for word, from sim/setup_agents.py)
  with_structured_output   Pydantic Decision(order: int 3..18) -> validated decisions, automatic re-ask on bad output
  message history          every day's outcome is fed back, so the agent remembers earlier days
  LangGraph StateGraph     decide -> play -> (loop 25 days) -> debrief
  LangSmith tracing        set LANGSMITH_TRACING=true and LANGSMITH_API_KEY; each agent is one named, tagged run
Payoffs, dice and prompts come from sim/newsvendor_env.py, so rules are identical to the first run.

Setup:   pip install langgraph pydantic python-dotenv  + langchain-anthropic (API) or langchain-ollama (free, local)
         copy .env.example to .env and fill in your keys (or export them in the terminal)   (optional: LANGSMITH_TRACING=true LANGSMITH_API_KEY=... LANGSMITH_PROJECT=newsvendor-ai)
Run:     python langchain_sim/run_langgraph.py --limit 2                       (cheap test: 2 agents, Claude Haiku)
         python langchain_sim/run_langgraph.py --variant all --n 50 --workers 5   (full: 4 wordings x 50 agents)
         python langchain_sim/run_langgraph.py --provider ollama --model llama3.1:latest --workers 2 --variant all   (free, local)
Output:  data/state_api_haiku_5_5[_v1..v3]/A*.json (Claude)   data/state_langchain[_v1..v3]/ (Llama).  Re-run the same command to resume.
Then:    python analysis/variant_comparison.py haiku     (or llama)
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
    range_min: int = Field(description="Lowest possible demand on a day (three dice added up)")
    range_max: int = Field(description="Highest possible demand on a day (three dice added up)")
    strategy: str = Field(description="One sentence on how you decided your orders")

SYSTEM = ChatPromptTemplate.from_messages([("system",
"""You are a participant in a paid decision-making experiment at a university behavioral lab ("Supply Chain Ordering Decisions"). Take part the way a human participant genuinely would; you are not an AI assistant, and nobody will help you with the task.

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
        q = HumanMessage(content='Final questions. (1) "On a day when you order 10 units and the demand turns out to be 7, by how many play dollars does your balance change that day?" (2) What are the lowest and the highest possible demand on any day (three dice added up)? (3) In one sentence, how did you decide your orders?')
        d = debriefer.invoke(system_prompt_msgs(s["sim"]) + s["messages"] + [q])
        sim = dict(s["sim"]); sim["debrief"] = dict(answer=str(d.check_answer), range_min=str(d.range_min), range_max=str(d.range_max), strategy=d.strategy)
        return {"sim": sim}

    g = StateGraph(S)
    g.add_node("decide", decide); g.add_node("play", play); g.add_node("debrief", debrief)
    g.set_entry_point("decide"); g.add_edge("decide", "play")
    g.add_conditional_edges("play", lambda s: "decide" if len(s["sim"]["rounds"]) < env.ROUNDS else "debrief")
    g.add_edge("debrief", END)
    return g.compile()

def make_llm(a):
    if a.provider == "ollama":            # free, local: needs Ollama running and `ollama pull <model>`
        from langchain_ollama import ChatOllama
        return ChatOllama(model=a.model, temperature=a.temperature, num_predict=300)
    from langchain_anthropic import ChatAnthropic   # Claude API: needs ANTHROPIC_API_KEY in .env
    # newer Claude models reject the temperature parameter, so it is not sent (Llama/Ollama still uses it)
    return ChatAnthropic(model=a.model, max_tokens=300, max_retries=8, timeout=120)

def out_dir(a, variant):
    tag = "state_langchain" if a.provider == "ollama" else "state_api_" + a.model.split(":")[0].replace("claude-", "").replace("-", "_")
    return ROOT / "data" / (tag + ("" if variant == "base" else "_" + variant))

def run_variant(a, variant):
    from prompt_variants import get_instr, strip_bonus
    instr_map = get_instr(variant, sa.INSTR)
    out = Path(a.out) if a.out else out_dir(a, variant); out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)           # same O/C assignment per agent ID in every wording
    conds = ["O"] * (a.n // 2) + ["C"] * (a.n - a.n // 2); rnd.shuffle(conds)
    roster = [dict(id=f"A{i:02d}", cond=c, persona="none") for i, c in enumerate(conds, 1)]
    llm = make_llm(a)

    def sys_msgs(sim):
        msgs = SYSTEM.format_messages(instr=instr_map[sim["cond"]])
        for m in msgs: m.content = strip_bonus(m.content, variant)
        return msgs

    graph = build_graph(llm, sys_msgs)

    def run_one(r):
        f = out / f"{r['id']}.json"
        if f.exists() and json.loads(f.read_text()).get("debrief"):      # resume: skip finished agents
            return r["id"] + " (already done)"
        sim = dict(id=r["id"], cond=r["cond"], persona=r["persona"], cash=env.START_CASH, rounds=[], debrief=None)
        first = HumanMessage(content=env.prompt(sim))
        try:
            res = graph.invoke({"sim": sim, "messages": [first]},
                config={"run_name": f"newsvendor-{variant}-{r['id']}-{r['cond']}", "tags": [r["cond"], a.model, variant],
                        "metadata": dict(agent_id=r["id"], condition=r["cond"], model=a.model, variant=variant, temperature=a.temperature),
                        "recursion_limit": 200})
        except Exception as e:                                           # one bad agent must not stop the batch
            return f"{r['id']} FAILED: {type(e).__name__}: {str(e)[:150]}"
        f.write_text(json.dumps(res["sim"], indent=1)); return r["id"]

    (out / "roster.json").write_text(json.dumps(roster, indent=1))
    print(f"[{variant}] {a.provider}/{a.model}  n={a.n}  -> {out}", flush=True)
    todo = roster[:a.limit] if a.limit else roster
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for done in ex.map(run_one, todo): print("finished", done, flush=True)

def main():
    ap = argparse.ArgumentParser(description="Newsvendor O-vs-C experiment with LangGraph agents (Claude API or local Llama)")
    ap.add_argument("--n", type=int, default=50, help="agents per wording (half O, half C)")
    ap.add_argument("--provider", choices=["anthropic","ollama"], default="anthropic")
    ap.add_argument("--model", default=None, help="default: claude-haiku-5-5 (anthropic) or llama3.1:8b (ollama)")
    ap.add_argument("--temperature", type=float, default=1.0, help="Ollama/Llama only"); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--out", default=None)
    ap.add_argument("--variant", default="base", help="base | v1 | v2 | v3 | all  (wording, see sim/prompt_variants.py)")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N agents (use --limit 2 for a cheap test run)")
    a = ap.parse_args()
    a.model = a.model or ("llama3.1:8b" if a.provider == "ollama" else "claude-haiku-5-5")
    for v in (["base", "v1", "v2", "v3"] if a.variant == "all" else [a.variant]):
        run_variant(a, v)

if __name__ == "__main__":
    main()
