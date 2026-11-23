#!/usr/bin/env python3
"""Compare each prompt wording with the BASE wording WITHIN the same payment scheme, for one model.

  python analysis/variant_comparison.py claude     # data/state, state_v1, state_v2, state_v3
  python analysis/variant_comparison.py haiku      # data/state_api_haiku_5_5, _v1.._v3 (Claude API)
  python analysis/variant_comparison.py llama      # data/state_langchain, _v1, _v2, _v3

Rows:  O-vs-O  (variant O vs base O),  C-vs-C (variant C vs base C),  and each wording's own O-C gap.
A variant C is never compared with the base O (or the reverse).
Agents that failed either comprehension check are reported but kept in; add --passed-only to drop them."""
import json, sys, glob, math, statistics as st
from pathlib import Path
from scipy import stats
ROOT = Path(__file__).resolve().parent.parent
model = sys.argv[1] if len(sys.argv) > 1 else "claude"
passed_only = "--passed-only" in sys.argv
DIRS = {"claude": {"base": "state", "v1": "state_v1", "v2": "state_v2", "v3": "state_v3"},
        "haiku": {"base": "state_api_haiku_5_5", "v1": "state_api_haiku_5_5_v1", "v2": "state_api_haiku_5_5_v2", "v3": "state_api_haiku_5_5_v3"},
        "llama": {"base": "state_langchain", "v1": "state_langchain_v1", "v2": "state_langchain_v2", "v3": "state_langchain_v3"}}[model]

def num(x):
    try: return int(float(str(x).replace("+", "").strip()))
    except Exception: return None

def load(folder):
    out = {"O": [], "C": []}; fails = 0; n = 0
    for f in sorted(glob.glob(str(ROOT / "data" / folder / "A*.json"))):
        s = json.load(open(f)); o = [r["order"] for r in s["rounds"]]
        if len(o) < 25 or not s.get("debrief"): continue
        d = s["debrief"]; n += 1
        ok = num(d.get("answer")) == 4 and ((num(d.get("range_min")) == 3 and num(d.get("range_max")) == 18) if "range_min" in d else True)
        fails += (not ok)
        if passed_only and not ok: continue
        out[s["cond"]].append(st.mean(o))
    return out, n, fails

def cmp(a, b):
    if len(a) < 2 or len(b) < 2: return None
    t = stats.ttest_ind(a, b, equal_var=False)
    sp = math.sqrt(((len(a)-1)*st.variance(a) + (len(b)-1)*st.variance(b)) / (len(a)+len(b)-2))
    return st.mean(a) - st.mean(b), t.pvalue, (st.mean(a)-st.mean(b)) / sp if sp else float("nan")

D = {k: load(v) for k, v in DIRS.items()}
print(f"MODEL: {model}   ({'passed both checks only' if passed_only else 'all completed agents'})\n")
print(f"{'wording':8}{'n':>4}{'meanO':>8}{'meanC':>8}{'O-C':>8}{'p':>8}{'fail checks':>13}")
for k, (g, n, f) in D.items():
    r = cmp(g["O"], g["C"])
    print(f"{k:8}{n:>4}{(st.mean(g['O']) if g['O'] else float('nan')):8.2f}{(st.mean(g['C']) if g['C'] else float('nan')):8.2f}" +
          (f"{r[0]:8.2f}{r[1]:8.3f}" if r else f"{'-':>8}{'-':>8}") + f"{f:>10}/{n}")
print("\nWithin-scheme comparison with BASE (variant minus base):")
print(f"{'wording':8}{'scheme':8}{'diff':>8}{'p':>8}{'d':>8}")
for k in ["v1", "v2", "v3"]:
    for c in "OC":
        r = cmp(D[k][0][c], D["base"][0][c])
        print(f"{k:8}{c + ' vs base ' + c:8}" + (f"{r[0]:8.2f}{r[1]:8.3f}{r[2]:8.2f}" if r else "  (not enough agents)"))
