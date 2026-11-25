#!/usr/bin/env python3
"""Deliverable B: one clean AI dataset for the whole study (both models, all four wordings).
Run (after analyze.py has been run for every folder):  python analysis/build_master_dataset.py
Writes data/ai_dataset_all_long.csv (one row per decision) and data/ai_dataset_all_participants.csv (one row per agent)."""
import pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MODELS = {"Claude Haiku 5.5": "", "Llama 3.1": "_langchain"}      # file tag used by analyze.py
WORDINGS = ["base", "v1", "v2", "v3", "faithful"]   # "faithful" = base wording + human-like feedback (only if that run exists)
longs, parts = [], []
for model, mtag in MODELS.items():
    for w in WORDINGS:
        tag = mtag + ("" if w == "base" else "_" + w)
        if not (ROOT / "data" / f"ai_agent_long{tag}.csv").exists():
            if w == "faithful": continue            # optional run
            raise SystemExit(f"missing data/ai_agent_long{tag}.csv: run analysis/analyze.py for that folder first")
        l = pd.read_csv(ROOT / "data" / f"ai_agent_long{tag}.csv").drop(columns=["persona"], errors="ignore")
        p = pd.read_csv(ROOT / "data" / f"ai_agent_participants{tag}.csv").drop(columns=["persona"], errors="ignore")
        for df, out in ((l, longs), (p, parts)):
            df.insert(0, "wording", w); df.insert(0, "model", model); out.append(df)
L = pd.concat(longs, ignore_index=True); P = pd.concat(parts, ignore_index=True)
L.to_csv(ROOT / "data/ai_dataset_all_long.csv", index=False); P.to_csv(ROOT / "data/ai_dataset_all_participants.csv", index=False)
print("long", L.shape, "participants", P.shape); print(P.groupby(["model", "wording"]).agg(agents=("agent_id", "count"), failed_profit=("fail_profit_check", "sum"), failed_range=("fail_range_check", "sum")))
