"""Writes analysis/results_summary.csv (O vs C per model/wording) and results_vs_base.csv (variant vs base, same scheme).
Run: python analysis/export_results_csv.py"""
import json, glob, csv
import numpy as np
from pathlib import Path
from scipy import stats
ROOT = Path(__file__).resolve().parent.parent
DIRS = {"Claude Haiku 5.5": "state_api_haiku_5_5", "Llama 3.1": "state_langchain"}
WORDINGS = ["base", "v1", "v2", "v3"]

def load(pre, w):
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / (pre + ("" if w == "base" else "_" + w)) / "A*.json"))):
        d = json.load(open(f)); b = d.get("debrief")
        if not b: continue
        m = np.mean([r["order"] for r in d["rounds"]])
        pf = str(b["answer"]).strip() != "4"
        rf = not (str(b["range_min"]).strip() == "3" and str(b["range_max"]).strip() == "18")
        rows.append(dict(id=d["id"], cond=d["cond"], mean_order=m, profit_fail=pf, range_fail=rf))
    return rows

def welch(a, b):
    if len(a) < 2 or len(b) < 2: return None, None
    t = stats.ttest_ind(a, b, equal_var=False)
    sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float(t.pvalue), (float((np.mean(a) - np.mean(b)) / sd) if sd > 0 else None)

def r(x, k=3): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)

data = {(m, w): load(p, w) for m, p in DIRS.items() for w in WORDINGS}
out1, out2 = [], []
for (m, w), rows in data.items():
    for subset in ["all agents", "passed both checks"]:
        sel = [x for x in rows if subset == "all agents" or not (x["profit_fail"] or x["range_fail"])]
        O = [x["mean_order"] for x in sel if x["cond"] == "O"]; C = [x["mean_order"] for x in sel if x["cond"] == "C"]
        p, d = welch(O, C)
        out1.append(dict(model=m, wording=w, subset=subset, n_agents=len(sel), n_O=len(O), n_C=len(C),
                         mean_order_O=r(np.mean(O)) if O else "", mean_order_C=r(np.mean(C)) if C else "",
                         gap_O_minus_C=r(np.mean(O) - np.mean(C)) if O and C else "", p_value=r(p, 4), cohens_d=r(d, 2),
                         profit_check_fails=sum(x["profit_fail"] for x in rows), range_check_fails=sum(x["range_fail"] for x in rows), agents_total=len(rows)))
for m in DIRS:
    for w in WORDINGS[1:]:
        for sc in ["O", "C"]:
            a = [x["mean_order"] for x in data[(m, w)] if x["cond"] == sc]; b = [x["mean_order"] for x in data[(m, "base")] if x["cond"] == sc]
            p, d = welch(a, b)
            out2.append(dict(model=m, comparison=f"{w} {sc} vs base {sc}", n_variant=len(a), n_base=len(b),
                             mean_variant=r(np.mean(a)), mean_base=r(np.mean(b)), difference=r(np.mean(a) - np.mean(b)), p_value=r(p, 4), cohens_d=r(d, 2)))
for name, rows in [("results_summary.csv", out1), ("results_vs_base.csv", out2)]:
    with open(ROOT / "analysis" / name, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    print("wrote", name, len(rows), "rows")
