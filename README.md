# AI-agent replication of Chen, Kök & Tong (2013): payment framing in the newsvendor game

Question: do AI agents show the human bias of ordering more under own financing (O) than under customer financing (C), when the two schemes are mathematically identical?

## Layout
- `sim/newsvendor_env.py`  game engine (real dice, O/C payoffs, validation, logging, comprehension check)
- `sim/setup_agents.py`    creates 50 agents (25 O, 25 C, random order, random student persona) and their prompts
- `prompts/A01.txt … A50.txt`  exact briefing each agent received (instructions copied word for word from the study materials)
- `data/state/*.json`      raw per-agent logs (every day: order, dice, demand, sold, leftover, change, cash)
- `data/ai_agent_long.csv` one row per agent-day (1,250 rows)
- `data/ai_agent_wide.csv` one row per agent, order1…order25 (same layout as the human files)
- `data/ai_agent_participants.csv` per-agent summary, comprehension check, one-sentence strategy
- `data/human/`            human data from the replication package
- `analysis/analyze.py`    compiles, validates, tests and plots; outputs `results.json`, `summary_table.csv`, `fig_*.png`
- `dashboard/`             class dashboard source (`template.html`, `build_dashboard.py`)

## Variables (ai_agent_long.csv)
agent_id, condition (O/C), persona, day (1–25), order (3–18), d1,d2,d3 (dice), demand (sum), sold = min(order, demand), leftover = max(order − demand, 0), change (play-dollar change that day), cash_end.

## Procedure
Each agent is a Claude sub-agent (Sonnet) in its own conversation, so it remembers all earlier days. It sees the study's instructions for its condition, a note that demand is the sum of three standard dice (as humans were told), and the same day-by-day prompts as the Qualtrics version. It plays only through the engine commands. After day 25 it answers a comprehension question (correct answer: +4) and gives a one-line strategy. Nothing was tuned to match human results.

## Re-run
`python3 sim/setup_agents.py` (resets state), run each prompt in `prompts/` with an LLM agent that can execute shell commands, then `python3 analysis/analyze.py` and `python3 dashboard/build_dashboard.py`.

## Main result (participant-average order, two-sample t-test)
| Sample | n O/C | Mean O | Mean C | O − C |
|---|---|---|---|---|
| Original (Duke) | 29/21 | 11.73 | 9.75 | 1.98 |
| UM online | 27/25 | 10.86 | 9.96 | 0.90 |
| UTD online | 25/25 | 11.28 | 9.87 | 1.42 |
| UM in person | 25/27 | 10.96 | 9.83 | 1.13 |
| Claude agents | 25/25 | 10.63 | 10.04 | 0.60 |

Agents: p < 0.001 (clustered-SE p < 0.001), Cohen d ≈ 2.0 because between-agent SD is 0.24 (O) and 0.35 (C) against about 1.7 for humans. 99% of O orders are 10 or 11; 95% of C orders are 10 or 11. Excluding the one agent that failed the comprehension check changes nothing (O − C = 0.60).
Caveat: the agents share one model, so they are not independent observations and the p-value overstates the evidence.

## XAI step (Lecture Note 03)
`python3 analysis/xai_analysis.py` fits an interpretable OLS model and a gradient-boosting model of the next order, then explains it with exact Shapley values (all feature subsets, as in the lecture's house-price example). Outputs `analysis/xai_results.json`, `analysis/fig_xai.png`. Human files have no dice data, so the like-for-like comparison uses cond_O, day and previous order only.

## Streamlit app (alternative to the HTML dashboard)
```
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```
Tabs: Present (9-step guided talk with timer), Live game (class plays; shared through `data/class_plays.json`), AI replay (Claude or Llama agent day by day), Humans vs AI, Why it matters, AI method. Class members open the Network URL that Streamlit prints (same Wi-Fi), or deploy the repo to Streamlit Community Cloud. Add `?cond=O` or `?cond=C` to the address to force a scheme.
