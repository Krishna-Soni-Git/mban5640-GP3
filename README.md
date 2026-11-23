# AI-agent replication of Chen, Kök & Tong (2013): payment framing in the newsvendor game

**Question:** do AI agents show the human bias of ordering more under own financing (O) than under customer financing (C), when the two schemes pay out exactly the same?

Own financing (O): pay 1 per unit ordered, earn 2 per unit sold. Customer financing (C): receive 1 per unit ordered, pay 2 per unit unsold. Best order is 10 or 11. 25 rounds, 100 play dollars to start, demand = sum of three dice (3–18).

## Design (final version)
- **Two models, reported separately:** Claude Haiku 5.5 (Claude API through LangGraph) and Llama 3.1 8B (free, local, Ollama through LangGraph).
- **Four prompt wordings per model:** `base` (the study's instructions word for word) and three alternative wordings `v1`, `v2`, `v3` (see `sim/prompt_variants.py`; v3 adds a token-spend incentive instead of the human $5 bonus paragraph).
- **50 agents per model per wording** (25 O + 25 C, the same O/C assignment for each agent ID in every wording, seed 2026) → 8 runs × 50 agents × 25 days = **10,000 decisions**.
- **No agent backgrounds.** Agents get exactly the study's instructions and nothing else (no age, gender, major or degree).
- **Two comprehension checks after day 25:** (1) order 10, demand 7 → profit change must be **+4**; (2) lowest and highest possible demand must be **3 and 18**. An agent "fails" if either is wrong. Results are shown for all agents and for agents that passed both.
- **Comparisons are within the same scheme:** variant C against base C and variant O against base O (never variant C against base O), plus O − C inside each wording.
- Dice are rolled by the engine after each order, so agents cannot influence them.

## Main results (mean order per agent; two-sample t-test)
| Sample | n O/C | Mean O | Mean C | O − C |
|---|---|---|---|---|
| Original human study (Duke) | 29/21 | 11.73 | 9.75 | 1.98 |
| Human replications (UM online / UTD online / UM in-person) | ~25/25 each | 10.86 / 11.28 / 10.96 | 9.97 / 9.87 / 9.83 | 0.90 / 1.42 / 1.13 |
| Claude Haiku 5.5, base wording | 25/25 | 10.56 | 10.54 | **0.03** (p = 0.81) |
| Llama 3.1, base wording | 25/25 | 13.23 | 13.26 | **−0.03** (p = 0.91) |

Both models show **no framing effect in the base wording**. Full tables: `analysis/results_summary.csv` (every model × wording, all agents and passed-both-checks) and `analysis/results_vs_base.csv` (variant against base within the same scheme).

| Model / wording | O − C gap | p | Failed profit check | Failed range check |
|---|---|---|---|---|
| Haiku base / v1 / v2 / v3 | +0.03 / +0.36 / +0.65 / +0.12 | 0.81 / <0.001 / <0.001 / 0.28 | 28 / 33 / 36 / 32 of 50 | 0 / 0 / 0 / 0 |
| Llama base / v1 / v2 / v3 | −0.03 / −0.37 / +0.20 / +0.41 | 0.91 / 0.10 / 0.57 / 0.09 | 46 / 47 / 47 / 49 of 50 | 6 / 7 / 12 / 7 |

Reading the results: Haiku orders almost always at 10–11 (about 93% of orders) and shows a small gap in the human direction in some wordings; Llama over-orders (about 12.3–14 units on average, 8% at 10–11) and its gap changes sign with the wording. Most agents of both models fail the profit check, so the agents that pass are few and unevenly split between O and C (for Haiku base 17 O vs 5 C); treat the passed-only numbers as suggestive. The 50 agents of one model are not 50 independent people.
Haiku 5.5 does not accept a temperature setting, so it ran at its default sampling; Llama ran at temperature 1.0.

## Layout
- `sim/newsvendor_env.py` game engine (real dice, O/C payoffs, validation, logging, the two comprehension checks)
- `sim/setup_agents.py` writes `prompts/` (A01–A50) and `prompts_v1/_v2/_v3/` (exact briefing per agent, no backgrounds)
- `sim/prompt_variants.py` the v1, v2, v3 instruction wordings
- `langchain_sim/run_langgraph.py` LangGraph runner: decide → play → loop 25 days → debrief; Claude API or Ollama
- `data/state_api_haiku_5_5[_v1|_v2|_v3]/` raw per-agent logs, Claude Haiku 5.5
- `data/state_langchain[_v1|_v2|_v3]/` raw per-agent logs, Llama 3.1
- `data/ai_dataset_all_long.csv` **Deliverable B**: all 10,000 decisions (columns `model`, `wording`, ...)
- `data/ai_dataset_all_participants.csv` per-agent summary, comprehension checks, strategy sentence (400 agents)
- `data/ai_agent_*{tag}.csv` the same data per run (tag `""` = Haiku base, `_v1` … Haiku variants, `_langchain`, `_langchain_v1` … Llama)
- `data/human/` human data from the replication package
- `data/archive_pre_rework/` the first run (Claude Sonnet sub-agents with student personas, one comprehension check) kept for the record; not used in the results
- `analysis/analyze.py` compiles, validates, tests and plots one run; `analysis/variant_comparison.py` tables for a model; `analysis/export_results_csv.py` writes the two results CSVs; `analysis/build_master_dataset.py` writes the master dataset; `analysis/xai_analysis.py` regression and Shapley step
- `streamlit_app/` presentation dashboard; `DATA_DICTIONARY.md` explains every variable

## Setup
```
pip install -r requirements.txt
```
Copy `.env.example` to `.env` and put your key in it: `ANTHROPIC_API_KEY=sk-ant-...` (never upload `.env`). In PowerShell you can instead run `$env:ANTHROPIC_API_KEY="sk-ant-..."` for that window.

## Re-run the agents
```
python langchain_sim\run_langgraph.py --limit 2                                        # 2-agent test, Claude Haiku
python langchain_sim\run_langgraph.py --variant all --n 50 --workers 5                 # Claude Haiku, 4 wordings
python langchain_sim\run_langgraph.py --provider ollama --model llama3.1:latest --workers 2 --variant all --n 50   # Llama
```
Re-running the same command resumes and skips finished agents. A single wording: `--variant base|v1|v2|v3`.

## Re-run the analysis (PowerShell)
```
$runs = @(
 @("data\state_api_haiku_5_5","") , @("data\state_api_haiku_5_5_v1","_v1"), @("data\state_api_haiku_5_5_v2","_v2"), @("data\state_api_haiku_5_5_v3","_v3"),
 @("data\state_langchain","_langchain"), @("data\state_langchain_v1","_langchain_v1"), @("data\state_langchain_v2","_langchain_v2"), @("data\state_langchain_v3","_langchain_v3"))
foreach ($r in $runs) { $env:AI_STATE_DIR=$r[0]; $env:AI_TAG=$r[1]; python analysis\analyze.py }
Remove-Item Env:AI_STATE_DIR, Env:AI_TAG
python analysis\build_master_dataset.py
python analysis\export_results_csv.py
python analysis\xai_analysis.py            # Claude Haiku;  $env:AI_TAG="_langchain"; python analysis\xai_analysis.py  for Llama
python analysis\variant_comparison.py haiku
python analysis\variant_comparison.py llama
```
Quick per-agent check of one run: `python analysis\quick_check.py data\state_api_haiku_5_5`.

## XAI step (Lecture Note 03)
`xai_analysis.py` fits an interpretable OLS model and a gradient-boosting model of the next order and explains it with exact Shapley values. Human files have no dice data, so the like-for-like comparison uses condition, day and previous order. Result: for people, the payment scheme explains about 17% of the Shapley importance and the previous order most of the rest; for Haiku the scheme explains about 2–3% (it follows its own previous order even more strongly), and for Llama about 6%.

## Streamlit app (presentation dashboard)
```
python -m streamlit run streamlit_app\app.py
```
Tabs: Present (10-step guided talk, timer, notes), Game (the newsvendor game for the presenter), AI system, AI replay (any saved Claude or Llama agent), Results (humans vs both models), Prompt wording (both models, four wordings, within-scheme comparisons), Q&A, Checklist. No API key is needed. Add `?presenter=1` to the address on your own laptop for notes and the timer.

## Limits
Agents of one model are not independent observations; agents have no real stakes; the alternative O texts for v1 and v2 were drafted by us from the C text supplied by the team; most agents fail the profit comprehension check, especially Llama; Haiku ran at default sampling.
