#!/usr/bin/env python3
"""Quick look at a pilot (or any run): mean order by scheme, O-C gap, share of orders at 10-11, check results.

Usage:  python analysis/quick_check.py data/state_langchain_v3
"""
import json, sys, statistics as st
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/state_langchain_v3")
files = sorted(p for p in folder.glob("A*.json"))
if not files:
    sys.exit(f"No agent files found in {folder}")

by = {"O": [], "C": []}
rows = []
for p in files:
    s = json.loads(p.read_text())
    orders = [r["order"] for r in s["rounds"]]
    if not orders:
        continue
    d = s.get("debrief") or {}
    ans = d.get("check_answer", d.get("answer"))
    # check question: order 10, demand 7. O: 2*7-10 = +4.  C: 10-2*3 = +4.  (both +4)
    passed = ans is not None and int(ans) == 4
    by[s["cond"]].append(st.mean(orders))
    rows.append((s["id"], s["cond"], len(orders), round(st.mean(orders), 2),
                 round(100 * sum(o in (10, 11) for o in orders) / len(orders)), ans, "pass" if passed else "FAIL"))

print(f"{'agent':6}{'cond':5}{'days':5}{'avg':>7}{'%10-11':>8}  check")
for r in rows:
    print(f"{r[0]:6}{r[1]:5}{r[2]:<5}{r[3]:>7}{r[4]:>8}  {r[5]} {r[6]}")

print()
for c in "OC":
    v = by[c]
    print(f"Scheme {c}: n={len(v)}  mean order = {st.mean(v):.2f}" if v else f"Scheme {c}: no agents yet")
if by["O"] and by["C"]:
    print(f"O - C gap = {st.mean(by['O']) - st.mean(by['C']):.2f}   (humans in the paper: 1.98)")
allo = [o for p in files for o in (r["order"] for r in json.loads(p.read_text())["rounds"])]
print(f"Orders at 10-11 overall: {100 * sum(o in (10, 11) for o in allo) / len(allo):.0f}%   (best answer is 10 or 11)")
print(f"Failed comprehension check: {sum(r[6] == 'FAIL' for r in rows)} of {len(rows)}")
