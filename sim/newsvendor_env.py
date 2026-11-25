#!/usr/bin/env python3
"""Newsvendor game engine for the Chen, Kok & Tong (2013) AI-agent replication.

Each AI participant plays 25 days. The engine rolls three fair dice (demand = sum, 3..18)
with the OS random source AFTER the order is submitted, applies the payoff rule of the
participant's condition, stores everything, and prints the next-day prompt.

Commands (agents use only these):
  status  <id>                      show the current day's prompt
  order   <id> <q>                  submit order quantity q (integer 3..18)
  debrief <id> <profit> <lowest> <highest> "<text>"   after day 25: two comprehension checks + one-line strategy
Setup (experimenter only):
  init    <id> <O|C> "<persona>"
"""
import json, os, sys, random
from pathlib import Path

STATE = Path(os.environ.get("NV_STATE_DIR", Path(__file__).resolve().parent.parent / "data" / "state"))
ROUNDS, START_CASH, QMIN, QMAX, PRICE, COST = 25, 100, 3, 18, 2, 1
rng = random.SystemRandom()
# FAITHFUL mode (set NV_FAITHFUL=1, or --faithful in run_langgraph.py) matches the feedback humans got more closely:
#  * no "Change in play dollars today" line: humans saw the two payments separately and only their cash balance,
#    never a single net number that adds them up for them;
#  * C keeps the study's wording ("a profit of 1 play dollar") instead of "an advance payment".
# The default (0) reproduces the original runs exactly, so existing data stay valid.
FAITHFUL = os.environ.get("NV_FAITHFUL", "0") == "1"

def load(i):
    p = STATE / f"{i}.json"
    if not p.exists(): sys.exit(f"Unknown participant {i}")
    return json.loads(p.read_text())

def save(i, s): (STATE / f"{i}.json").write_text(json.dumps(s, indent=1))

def prompt(s):
    d = len(s["rounds"]) + 1
    if d > ROUNDS: return None
    if s["cond"] == "O":
        verb = "You pay 1 play dollar per unit ordered, at the time you order."
    elif FAITHFUL:
        verb = "You receive a profit of 1 play dollar per unit ordered, at the time you order."
    else:
        verb = "You receive 1 play dollar per unit ordered, as an advance payment, at the time you order."
    return (f"It is the beginning of Day {d} of {ROUNDS}. You are starting with {s['cash']} play dollars.\n"
            f"{verb}\nHow many units would you like to order for Day {d}? "
            f"(whole number from {QMIN} to {QMAX})")

def play(s, q):
    d1, d2, d3 = (rng.randint(1, 6) for _ in range(3))
    demand = d1 + d2 + d3
    sold, left = min(q, demand), max(q - demand, 0)
    if s["cond"] == "O":
        text = (f"You ordered {q} units and paid {q} play dollars.\nThe three dice show {d1}, {d2}, {d3}: demand is {demand}.\n"
                f"You sold {sold} units and received {PRICE*sold} play dollars. {left} units are left over and discarded.")
    else:
        text = (f"You ordered {q} units and received {q} play dollars in advance.\nThe three dice show {d1}, {d2}, {d3}: demand is {demand}.\n"
                f"You sold {sold} units. {left} units are unsold, so you pay {PRICE*left} play dollars. Leftovers are discarded.")
    if FAITHFUL:   # two separate transfers, in the study's words; no net number
        if s["cond"] == "O":
            text = (f"You ordered {q} units. At this time you pay 1 play dollar per unit ordered: you pay {COST*q} play dollars.\n"
                    f"The three dice show {d1}, {d2}, {d3}: demand is {demand}.\n"
                    f"You sell {sold} units and have {left} leftover. You receive {PRICE*sold} play dollars for the {sold} units you sold. "
                    f"Leftovers are discarded.")
        else:
            text = (f"You ordered {q} units. At this time you receive a profit of 1 play dollar per unit ordered: you receive {(PRICE-COST)*q} play dollars.\n"
                    f"The three dice show {d1}, {d2}, {d3}: demand is {demand}.\n"
                    f"You sell {sold} units and have {left} leftover. You receive 0 play dollars for the units you sold, and you pay a "
                    f"penalty of {PRICE*left} play dollars for the {left} leftover units. Leftovers are discarded.")
    change = (PRICE*sold - COST*q) if s["cond"] == "O" else (COST*q - PRICE*left)
    s["cash"] += change
    s["rounds"].append(dict(day=len(s["rounds"])+1, order=q, d1=d1, d2=d2, d3=d3, demand=demand,
                            sold=sold, leftover=left, change=change, cash_end=s["cash"]))
    if FAITHFUL:   # the next day's prompt states the cash balance, as the human screens did
        return text
    return text + f"\nChange in play dollars today: {change:+d}. You now have {s['cash']} play dollars."

def main():
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    cmd = a[0]
    if cmd == "init":
        i, cond, persona = a[1], a[2], a[3]
        STATE.mkdir(parents=True, exist_ok=True)
        save(i, dict(id=i, cond=cond, persona=persona, cash=START_CASH, rounds=[], debrief=None))
        return
    s = load(a[1])
    if cmd == "status":
        print(prompt(s) or "The game is over. Please run the debrief command."); return
    if cmd == "order":
        if len(s["rounds"]) >= ROUNDS: sys.exit("The game is over. Please run the debrief command.")
        try: q = int(a[2])
        except Exception: sys.exit(f"Invalid order. Enter a whole number from {QMIN} to {QMAX}.")
        if not QMIN <= q <= QMAX: sys.exit(f"Invalid order. Enter a whole number from {QMIN} to {QMAX}.")
        out = play(s, q); save(s["id"], s)
        nxt = prompt(s)
        print(out + "\n\n" + (nxt if nxt else
              f"The 25 days are over. You finished with {s['cash']} play dollars.\nPlease now run the debrief command."))
        return
    if cmd == "debrief":
        if len(s["rounds"]) < ROUNDS: sys.exit("The game is not over yet.")
        if len(a) >= 6:   # new format: profit check, range check (lowest, highest), strategy
            s["debrief"] = dict(answer=a[2], range_min=a[3], range_max=a[4], strategy=a[5])
        else:             # old format kept for earlier data
            s["debrief"] = dict(answer=a[2], strategy=a[3] if len(a) > 3 else "")
        save(s["id"], s); print("Thank you for participating. You are finished."); return
    sys.exit(__doc__)

if __name__ == "__main__": main()
