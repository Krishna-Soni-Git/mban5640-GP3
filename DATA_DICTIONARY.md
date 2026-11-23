# Data dictionary (Deliverable B)

Study design: 2 models (Claude Haiku 5.5, Llama 3.1) × 4 prompt wordings (base, v1, v2, v3) × 50 agents (25 O, 25 C) × 25 days = 10,000 decisions. Agents have no personas or backgrounds.

## data/ai_dataset_all_long.csv (10,000 rows: one row per agent-day)
| Variable | Meaning |
|---|---|
| model | Claude Haiku 5.5 or Llama 3.1 |
| wording | Instruction wording: base (the study's text), v1, v2, v3 (see sim/prompt_variants.py) |
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

## data/ai_dataset_all_participants.csv (400 rows: one row per agent)
| Variable | Meaning |
|---|---|
| model, wording, agent_id, condition | As above |
| comprehension_answer | Answer to check 1: order 10 and demand 7, by how many play dollars does the balance change? Correct answer is +4 in both schemes |
| range_min_answer, range_max_answer | Answers to check 2: lowest and highest possible demand (three dice). Correct answers are 3 and 18 |
| fail_profit_check | 1 if the check 1 answer is not 4 |
| fail_range_check | 1 if check 2 is not 3 and 18 |
| comprehension_fail | 1 if either check failed, else 0 |
| final_cash | Cash after round 25 |
| strategy | The agent's own one-sentence description of how it decided |
| avg_order | Mean of its 25 orders |

## Per-run files (same variables, one run each)
`data/ai_agent_long{tag}.csv`, `ai_agent_participants{tag}.csv` (also has a `persona` column that is always `none`) and `ai_agent_wide{tag}.csv` (one row per agent, columns order1–order25, same layout as the human files). Tags: `""` Haiku base, `_v1`, `_v2`, `_v3` Haiku variants; `_langchain`, `_langchain_v1`, `_langchain_v2`, `_langchain_v3` Llama.

## analysis/results_summary.csv (one row per model × wording × subset)
| Variable | Meaning |
|---|---|
| subset | "all agents" or "passed both checks" |
| n_agents, n_O, n_C | Agents in the subset and per scheme |
| mean_order_O, mean_order_C | Mean of the agents' average orders |
| gap_O_minus_C | Mean O minus mean C |
| p_value, cohens_d | Welch t-test p-value and effect size (blank if a group has fewer than 2 agents) |
| profit_check_fails, range_check_fails, agents_total | Failure counts for the whole run (not only the subset) |

## analysis/results_vs_base.csv (variant against base, same scheme)
Columns: model, comparison (for example "v2 C vs base C"), n_variant, n_base, mean_variant, mean_base, difference, p_value, cohens_d.

## data/state_*/A##.json (raw logs)
`id`, `cond`, `persona` (always none), `cash`, `rounds` (list of the 25 days with the fields above), `debrief` (the check answers and strategy).

## data/human/
Three human replication datasets (UM online, UTD online, UM in-person): one row per participant, order1–order25, Condition, ComprehensionFail.
