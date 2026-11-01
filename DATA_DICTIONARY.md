# Data dictionary (Deliverable B)

## data/ai_agent_long.csv (1,250 rows: 50 agents x 25 rounds)
| Variable | Meaning |
|---|---|
| agent_id | A01-A50 |
| condition | O (own financing) or C (customer financing) |
| persona | Age, gender, major and degree level given to the agent |
| day | Round 1-25 |
| order | Units ordered (3-18) |
| d1, d2, d3 | The three dice rolled by the engine after the order |
| demand | d1 + d2 + d3 |
| sold | min(order, demand) |
| leftover | max(order - demand, 0) |
| change | Cash change that round (same in O and C): 2*sold - order |
| cash_end | Cash at end of round (start 100) |

## data/ai_agent_participants.csv (50 rows)
| Variable | Meaning |
|---|---|
| comprehension_answer | Agent's answer to the end-of-game comprehension question |
| comprehension_fail | 1 if the answer was wrong, else 0 |
| final_cash | Cash after round 25 |
| strategy | Agent's own description of its strategy |
| avg_order | Mean of its 25 orders |

## data/ai_agent_wide.csv
One row per agent, one column per day of orders.

## data/human/
Three human replication datasets (UM online, UTD online, UM in-person).
