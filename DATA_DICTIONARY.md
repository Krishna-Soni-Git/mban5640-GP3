# Data dictionary (Deliverable B)

Study design: 2 models (Claude Haiku 5.5, Llama 3.1) × 4 prompt wordings (base, v1, v2, v3) × 50 agents (25 O, 25 C) × 25 days, plus one Claude Haiku run in the base wording with human-like feedback (`faithful`): 9 runs, 450 agents, 11,250 decisions. Agents have no personas or backgrounds.

## How the data were collected
Each agent is an independent LangGraph session (`langchain_sim/run_langgraph.py`) that receives the study's instructions as its system prompt (exact text in `prompts_final/`). Each day it returns an order through a validated structured output (whole number 3–18; invalid answers are re-asked). The game engine (`sim/newsvendor_env.py`) then rolls three dice, applies the O or C payoff rule and writes the day to the agent's JSON log. After day 25 the agent answers the debrief questions. `analysis/analyze.py` compiles the logs into the CSV files below and checks every row (orders out of range, missing values, demand ≠ sum of dice, payoff-rule violations: 0 in every run). `analysis/build_master_dataset.py` stacks all runs into the two master files.

Model settings: Claude Haiku 5.5 through the Claude API, no temperature sent (the model does not accept one; the API default is 1.0), at most 300 tokens per answer. Llama 3.1 8B through Ollama, temperature 1.0, at most 300 tokens. Scheme assignment uses seed 2026, so an agent ID has the same scheme in every wording.

## data/ai_dataset_all_long.csv (11,250 rows: one row per agent-day)
| Variable | Meaning |
|---|---|
| model | Claude Haiku 5.5 or Llama 3.1 |
| wording | Prompt version: base (the study's text), v1, v2, v3 (see `sim/prompt_variants.py`), or faithful (base text with human-like feedback: no daily net-change line, the study's C wording "a profit"; Claude only) |
| agent_id | A01–A50 within a model and wording. The same ID has the same O/C condition in every wording |
| condition | O (own financing) or C (customer financing) |
| day | Round 1–25 |
| order | Units ordered (whole number 3–18) |
| d1, d2, d3 | The three dice rolled by the engine after the order |
| demand | d1 + d2 + d3 (3–18) |
| sold | min(order, demand) |
| leftover | max(order − demand, 0) |
| change | Play-dollar change that day (same in O and C): 2·sold − order |
| cash_end | Cash at the end of the round (start 100) |

## data/ai_dataset_all_participants.csv (450 rows: one row per agent)
| Variable | Meaning |
|---|---|
| model, wording, agent_id, condition | As above |
| comprehension_answer | Answer to our profit check: order 10 and demand 7, by how many play dollars does the balance change? Correct answer is +4 in both schemes |
| range_min_answer, range_max_answer | Answers to the original study's check: lowest and highest possible demand (three dice). Correct answers are 3 and 18 |
| fail_profit_check | 1 if the profit answer is not 4 |
| fail_range_check | 1 if the range answer is not 3 and 18. **Primary exclusion rule** (as in the original study): agents with 1 here are left out of the main analysis |
| comprehension_fail | 1 if either check failed, else 0 (secondary analysis only) |
| final_cash | Cash after round 25 |
| strategy | The agent's own one-sentence description of how it decided |
| avg_order | Mean of its 25 orders |
| followup_order | Answer to the original follow-up question "If you could play again choosing only one order quantity, what would you choose?" (faithful run only; blank elsewhere) |

## Per-run files (same variables, one run each)
`data/ai_agent_long{tag}.csv`, `ai_agent_participants{tag}.csv` (also has a `persona` column that is always `none`) and `ai_agent_wide{tag}.csv` (one row per agent, columns order1–order25, same layout as the human files). Tags: `""` Haiku base, `_v1`, `_v2`, `_v3` Haiku variants, `_faithful` Haiku with human-like feedback; `_langchain`, `_langchain_v1`, `_langchain_v2`, `_langchain_v3` Llama.

## analysis/human_vs_ai_table.csv (one row per human or AI sample)
| Variable | Meaning |
|---|---|
| sample, source | Sample name and Human / model |
| n_O, n_C, mean_O, mean_C | Participants per scheme and mean of their 25-day averages (AI: dice-range failures excluded) |
| gap, ci_low, ci_high, p_gap | O − C gap, its 95% confidence interval (Welch) and pooled-variance t-test p-value |
| round1_gap, p_round1 | O − C gap on day 1 only |
| followup_gap, p_followup | O − C gap in the follow-up question (original study and faithful run) |
| within_sd, between_sd, pct_10_11 | Average spread within a participant, spread between participants, % of orders at 10–11 |
| OxAI, OxAI_se, OxAI_p | AI gap minus human gap, from the pooled regression order ~ O + AI + O×AI with standard errors clustered by participant |
| AI_shift_in_C, AI_shift_p | Level difference AI − humans under C |
| excluded_range_fail | AI agents excluded for the dice-range check |

## analysis/results_summary.csv (one row per model × wording × subset)
| Variable | Meaning |
|---|---|
| subset | "all agents", "passed range check (original rule)" (primary) or "passed both checks" (secondary) |
| n_agents, n_O, n_C | Agents in the subset and per scheme |
| mean_order_O, mean_order_C | Mean of the agents' average orders |
| gap_O_minus_C | Mean O minus mean C |
| p_value, cohens_d | Welch t-test p-value and effect size (blank if a group has fewer than 2 agents) |
| profit_check_fails, range_check_fails, agents_total | Failure counts for the whole run (not only the subset) |

## analysis/results_vs_base.csv (variant against base, same scheme)
Columns: model, comparison (for example "v2 C vs base C", "faithful O vs base O"), n_variant, n_base, mean_variant, mean_base, difference, p_value, cohens_d.

## data/state_*/A##.json (raw logs)
`id`, `cond`, `persona` (always none), `cash`, `rounds` (list of the 25 days with the fields above, plus `reason` in `--reasons` runs), `debrief` (follow-up answer where asked, the check answers and strategy). Faithful runs also carry `faithful: true`.

## data/human/
Three human replication datasets (UM online, UTD online, UM in-person): one row per participant, order1–order25, Condition, ComprehensionFail.
