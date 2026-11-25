#!/usr/bin/env python3
"""Why did the agents order what they ordered?  Codes the agents' own words into reason categories.

Two sources of text:
  * strategy   the one-sentence strategy every agent gives after day 25 (exists in every run)
  * reason     a sentence with EVERY order, only in runs made with --reasons

Usage
  python analysis/reason_coding.py                                    # strategy sentences of all Haiku/Llama runs
  python analysis/reason_coding.py data/state_api_haiku_5_5_faithful_reasons   # one run (uses daily reasons if present)

Method: simple, transparent keyword rules (below). A text can fall into several categories. The rules are a first
pass: read the examples printed for each category and adjust the patterns if something is misclassified.
Key question for the study: do agents mention the money that moves AT ORDER TIME (the payment the paper says
people underweight), and does that differ between O and C?  Or do they reason only from average demand?

Outputs  analysis/reason_codes_strategy.csv  (or reason_codes_<run>.csv)  and a summary table printed to screen.
"""
import json, glob, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATS = {
    "average demand (10.5 / expected)": r"\baverage|\bmean\b|expected|10\.5|typical|cent(er|re)|middle|most likely|most common",
    "order-time payment":               r"per unit ordered|(when|as soon as) (i )?(order|ordered|place)|at (the time of )?order(ing)?|order[- ]time|upfront|up front|in advance|advance payment|paid (for|to) (order|stock)|profit (of 1 )?(per|for each) (unit )?order",
    "leftover / penalty risk":          r"leftover|left over|unsold|penalt|waste|discard|over-?order|excess|overstock",
    "lost sales / running short":       r"lost sales|lose sales|missed|miss(ing)? (out|sales)|short|stock-?out|under-?order|not enough|sell out|sold out|run out|unmet",
    "balance / trade-off":              r"balanc|trade-?off|between|both risks|hedg",
    "recent demand / adjusting":        r"\blast\b|previous|yesterday|recent|adjust|trend|observ|lately|after (a )?(day|days)",
    "explicit profit calculation":      r"expected (profit|value)|margin|critical|newsvendor|probabilit|break-?even|marginal",
}
RX = {k: re.compile(v, re.I) for k, v in CATS.items()}
RUNS = {"Claude Haiku 5.5": "state_api_haiku_5_5", "Llama 3.1": "state_langchain"}


def code(text):
    t = str(text or "")
    return {k: int(bool(rx.search(t))) for k, rx in RX.items()}


def rows_from_dir(d, model, wording):
    out = []
    for f in sorted(glob.glob(str(Path(d) / "A*.json"))):
        s = json.loads(Path(f).read_text())
        if not s.get("debrief") or len(s.get("rounds", [])) < 25:
            continue
        daily = [r for r in s["rounds"] if r.get("reason")]
        if daily:                                     # --reasons run: one row per decision
            for r in daily:
                out.append(dict(model=model, wording=wording, agent_id=s["id"], condition=s["cond"], day=r["day"],
                                order=r["order"], source="daily reason", text=r["reason"], **code(r["reason"])))
        st = s["debrief"].get("strategy", "")
        out.append(dict(model=model, wording=wording, agent_id=s["id"], condition=s["cond"], day=None,
                        order=sum(r["order"] for r in s["rounds"]) / 25, source="end-of-game strategy", text=st, **code(st)))
    return out


def main():
    if len(sys.argv) > 1:                             # one named run folder
        d = Path(sys.argv[1]); d = d if d.is_absolute() else ROOT / d
        rows = rows_from_dir(d, d.name, "-"); name = d.name
    else:                                             # strategy sentences of every standard run
        rows = []
        for model, pre in RUNS.items():
            for d in sorted(glob.glob(str(ROOT / "data" / (pre + "*")))):
                w = Path(d).name[len(pre):].strip("_") or "base"
                rows += rows_from_dir(d, model, w)
        name = "strategy"
    if not rows:
        raise SystemExit("no finished agents found")
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "analysis" / f"reason_codes_{name}.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 140)
    for src, g in D.groupby("source"):
        print(f"\n=== {src}: % of texts mentioning each reason, by model / wording / scheme ===")
        tab = (g.groupby(["model", "wording", "condition"])[list(CATS)].mean() * 100).round(0)
        tab.insert(0, "n", g.groupby(["model", "wording", "condition"]).size())
        print(tab.to_string())
    print("\n=== examples per category (first 2) ===")
    for k in CATS:
        ex = D[D[k] == 1]["text"].head(2).tolist()
        print(f"- {k}:"); [print("     ", e) for e in ex]
    none = D[D[list(CATS)].sum(axis=1) == 0]["text"].head(5).tolist()
    print("- (no category):"); [print("     ", e) for e in none]
    print(f"\nwrote analysis/reason_codes_{name}.csv")


if __name__ == "__main__":
    main()
