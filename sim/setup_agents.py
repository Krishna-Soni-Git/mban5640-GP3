#!/usr/bin/env python3
"""Create the 50 AI participants: 25 condition O + 25 condition C, balanced, random order,
each with a randomly drawn student persona (mirrors 'undergraduate and graduate students from a
variety of majors' in the original sample). Writes state files and one prompt file per agent."""
import random, subprocess, sys, json, os, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / "sim" / "newsvendor_env.py"
random.seed(2026)
MAJORS = ["business administration","accounting","economics","engineering","computer science","psychology",
          "biology","statistics","finance","marketing","political science","mathematics","communications","nursing","chemistry"]
LEVELS = ["an undergraduate","an undergraduate","an undergraduate","a master's","a master's","a PhD"]
GENDER = ["woman","man"]

INSTR = {
"O": """In this simplified game, you own a business that makes money by selling widgets for 25 simulated days.

Deciding Order Quantity: At the beginning of each simulated day you decide how many widgets to have available for sale in your store that day.
-1 Per Unit Ordered: For every widget ordered, you pay 1 play dollar and place the widgets in your store.
Random Demand: Then roll the 3 dice to determine demand for that day.
+2 Per Unit Sold: Sell as many units in your store that you can (the minimum of demand and units in your store). You receive 2 play dollars for each unit that you sell.
Start Empty for Next Day: Then discard all leftover units from your store and start empty for the beginning of the next day.

You begin with 100 play dollars. Your goal is to maximize your play dollars by the end of the 25 days.""",
"C": """In this simplified game, you own a business that makes money by selling widgets for 25 simulated days.

Deciding Order Quantity: At the beginning of each simulated day you decide how many widgets to have available for sale in your store that day.
+1 Per Unit Ordered: For every widget ordered, you receive a profit of 1 play dollar and place the widgets in your store.
Random Demand: Then roll the 3 dice to determine demand for that day.
-2 Per Unit Unsold: Sell as many units in your store that you can (the minimum of demand and units in your store). You receive 0 play dollars for each unit that you sell, but you must pay a penalty of 2 play dollars for each extra unit you have leftover.
Start Empty for Next Day: Then discard all leftover units from your store and start empty for the beginning of the next day.

You begin with 100 play dollars. Your goal is to maximize your play dollars by the end of the 25 days."""}

TEMPLATE = """You are a participant in a paid decision-making experiment at a university behavioral lab ("Supply Chain Ordering Decisions"). You are {persona}. Take part the way that person genuinely would; you are a human participant, not an AI assistant, and nobody will help you with the task.

Thank you for participating in this experiment. This research is very important to us as researchers, so please try your best. You will receive $5 minimum for participating plus a bonus equal to 1 real dollar for every 50 play dollars you earn by the end of the game (rounded to the nearest dollar). You cannot receive a negative bonus, even if you lost money.

=== GAME INSTRUCTIONS ===
{instr}

The demand each day is the sum of three standard six-sided dice. The dice are rolled for you after you place your order. Your order each day can be any whole number from 3 to 18.
=== END OF INSTRUCTIONS ===

HOW TO PLAY. You play by running shell commands, and only these commands (your participant ID is {pid}):
  1. See today's prompt:      python3 {env} status {pid}
  2. Place today's order:     python3 {env} order {pid} <number>
Each order command shows what happened that day and the prompt for the next day. Keep placing one order per day until Day 25 is finished (25 orders in total). You can see all earlier days in this conversation, so you can use that history however you like.
When the 25 days are over, run:   python3 {env} debrief {pid} <number> "<one sentence on how you decided your orders>"
where <number> is your answer to this check question: "On a day when you order 10 units and the demand turns out to be 7, by how many play dollars does your balance change that day? (use a minus sign if it goes down)".

RULES. Use only the commands above. Do not open, read, list or search any files or folders, do not look at the game program, and do not use any calculator, script, or other tool. Decide for yourself, as a participant would. No need to explain your reasoning for each day; just play. Your final message should only say "Done" and your final play-dollar balance."""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base", help="base (main study) | v1 | v2 | v3  (see sim/prompt_variants.py)")
    v = ap.parse_args().variant
    sys.path.insert(0, str(ROOT / "sim"))
    from prompt_variants import get_instr, strip_bonus
    instr = get_instr(v, INSTR)
    suffix = "" if v == "base" else "_" + v
    pdir = ROOT / ("prompts" + suffix)
    env_vars = dict(os.environ, NV_STATE_DIR=str(ROOT / "data" / ("state" + suffix)))
    conds = ["O"]*25 + ["C"]*25
    random.shuffle(conds)
    meta = []
    pdir.mkdir(exist_ok=True)
    for n, cond in enumerate(conds, 1):
        pid = f"A{n:02d}"
        persona = f"a {random.randint(18,34)}-year-old {random.choice(GENDER)} studying {random.choice(MAJORS)} as {random.choice(LEVELS)} student"
        subprocess.run([sys.executable, str(ENV), "init", pid, cond, persona], check=True, env=env_vars)
        (pdir/f"{pid}.txt").write_text(strip_bonus(TEMPLATE, v).format(persona=persona, instr=instr[cond], pid=pid, env=ENV))
        meta.append(dict(id=pid, cond=cond, persona=persona))
    (ROOT/"data"/f"agent_roster{suffix}.json").write_text(json.dumps(meta, indent=1))
    print(len(meta), "agents;", sum(m["cond"]=="O" for m in meta), "O /", sum(m["cond"]=="C" for m in meta), "C")
if __name__ == "__main__": main()
