# Full speech: dashboard steps 1–9 (about 10 minutes)

Text in brackets is a cue: what to click or point at. Don't say it out loud.
The handover lines (Krishna → Michael → Dale) can be dropped if one person presents.

---

## Step 1 · The business problem (45 s) — Krishna

**[Dashboard on step 1]**

Hi everyone, we're Krishna, Michael and Dale.

Let me start with a decision every store manager faces. You have to order stock *before* you know how many customers will show up. Order too little and you lose sales. Order too much and you're stuck with leftovers. That's the classic newsvendor problem.

There's a detail that usually gets ignored: *when* the money moves. Sometimes you pay for stock up front. Sometimes a customer or partner pays you first, and you pay back for whatever doesn't sell. If the profit is exactly the same either way, a rational manager should order the same amount.

So our question is: **people turn out to care about that timing. Do AI agents too?** It matters because companies and researchers are starting to use AI agents as cheap stand-ins for human participants, and that only works if the agents make the same mistakes we do.

## Step 2 · The original experiment (75 s)

**[Click Next]**

We replicate a 2013 Management Science paper by Chen, Kök and Tong.

They had 99 students at Duke run a small widget shop for 25 days, starting with 100 play dollars. Every day they picked an order, then rolled three real dice, and the total was that day's demand: anywhere from 3 to 18, about 10.5 on average. They played with poker chips and were paid real dollars based on how well they did.

The difference was how the money moved. Under **own financing, O**, you pay one dollar per unit when you order and get two dollars for every unit you sell. *[pause]* Under **customer financing, C**, you *receive* a dollar per unit when you order and pay two dollars for every unit you don't sell.

**[Point to the formula]**

This line is the key: the math is identical. Any order, any dice roll, same profit. The best order under both is 10 or 11. The paper also had a neutral scheme where all money moves after demand, but the replications, and we, compare O and C.

## Step 3 · What people did, and why (60 s)

**[Click Next]**

And yet people didn't behave the same. In the original study, students ordered about **11.7 units under O and 9.7 under C**: a two-unit gap from payment timing alone.

**[Hover over the Duke row, then point down the replication rows]**

In 2023, researchers at Michigan and UT Dallas ran it again three times. They found gaps of 0.9, 1.4 and 1.1: smaller than the original, but the same direction every time. Pooled, the gap is **1.15 units**. Those 154 people are the human benchmark for our AI agents.

Why does it happen? The authors call it **prospective accounting**. People give less weight to money that changes hands when they order, before they know demand, about 70 to 80 percent of its real value. Under O that money is a cost, so ordering feels cheaper and people order more. Under C it's income, so ordering feels less rewarding and people order less.

## Step 4 · Same task, same profit (30 s)

**[Click Next]**

One example day. You order 11 units, and the dice give a demand of 9.

**[Point to the left card]** Under own financing, you pay 11 up front and get 18 back for the 9 you sell. Plus 7.

**[Point to the right card]** Under customer financing, you receive 11 up front and pay a 4-dollar penalty for the 2 you didn't sell. Also plus 7.

Same order, same dice, same profit. Only the timing differs. This is exactly the task we gave our AI agents, and Michael will show you how we built them.

**[Click Next and hand over to Michael]**

## Step 5 · How we built the AI participants (90 s) — Michael

Thanks, Krishna.

We built the AI participants to be as close to the human ones as we could. **[Point across the four tiles]**

First, **instructions only**. Each agent reads the study's O or C instructions word for word, plus the bonus rules. We gave them no persona: no age, no gender, no background, because the study didn't describe its participants and inventing them would add a variable we can't justify.

Second, a **game engine** in code. The agent never rolls its own dice. The engine checks every order is a whole number from 3 to 18, rolls real dice *after* the order, and applies the payoff rule.

Third, **memory**. Each agent sees all of its earlier days, just like a person with a record sheet.

Fourth, the same **comprehension checks** as the study, so we can drop agents that didn't understand, using the original rule: anyone who gets the dice range wrong is excluded.

**[Point to the lower left tile]**

We ran two models: **Claude Haiku 5.5** through the Claude API and **Llama 3.1**, running for free on a laptop, both orchestrated with LangGraph. Fifty agents per run, 25 per scheme, the same size as the original sample. Each model played the study's wording plus three alternative wordings.

**[Point to the lower right tile]**

We also checked ourselves. Our first runs showed the agents a daily net change in play dollars, while humans saw the two payments separately. That net number could hide the timing. So we re-ran Claude with **human-like feedback**, showing the payments the way people saw them. In total: 450 agents and over 11,000 decisions.

## Step 6 · Results: AI versus people (160 s)

**[Click Next. "Averages" view is showing]**

Here's the main result. Each row is one group. The orange dot is the average order under O, the teal dot is C, and the line between them is the gap. The purple band is the best order, 10 to 11.

**[Run a finger down the top four rows]**

For people, the orange dot sits clearly to the right of the teal one in every study.

**[Point to the Claude and Llama rows]**

For the AIs, the dots sit almost on top of each other. Claude's gap is **+0.03 units**. With human-like feedback it's +0.19, and that's not significant. Llama's gap is +0.11.

**[Click "C" in the legend to highlight it, then click it again to clear]**

Notice also *where* they sit. Claude orders right inside the best band. **[Point to the last tile]** 93 percent of Claude's orders are exactly 10 or 11. Llama sits far to the right, around 13, so it over-orders under both schemes.

**[Click "Day by day"]**

This view shows the same thing over the 25 days. **[Point to Humans]** For people, orange stays above teal almost all game: the bias doesn't fade with experience. **[Point to Claude]** Claude starts a bit lower and within two or three days settles on 10 to 11 and stays there, whatever the scheme. **[Point to Llama]** Llama keeps drifting up.

**[Drag the "Days shown" slider back to 1, then slowly to 25]**

If we replay it day by day, you can watch Claude lock onto the answer while Llama climbs.

**[Point to the paragraph under the tiles]**

To make sure this isn't just noise, we pooled the human and AI data in one regression. It confirms the AI gap is significantly **smaller** than the human gap: by 1.12 units for Claude, 1.03 for Llama, and 0.95 for Claude with human-like feedback. So with the study's own wording, **neither model shows the human bias**.

Dale will show you whether the wording changes that.

**[Click Next and hand over to Dale]**

## Step 7 · Does the prompt wording matter? (40 s) — Dale

Thanks, Michael.

Each dot here is a gap, O minus C, with its 95 percent confidence interval. Zero means no effect. **[Point to the top two rows]** These are the people: clearly to the right.

**[Click "Claude" in the legend]**

For Claude, the wording does move the gap. With a customer story it's +0.36, and with bankruptcy warnings it rises to +0.65, the human direction. But it never reaches the human 1.15.

**[Click "Llama" in the legend]**

Llama is all over the place: from minus 0.47 to plus 0.54, sometimes the *wrong* direction.

**[Click the legend again to clear]**

So the wording matters, which is a warning for anyone using AI in research. We kept the study's wording for our main results and report the others openly.

## Step 8 · What explains the difference? (40 s)

**[Click Next]**

Why don't the agents show the bias? We modelled every decision with machine learning and used Shapley values to see what drives each next order. **[Click "Payment scheme" in the legend]** For people, the payment scheme explains about a sixth of it. For Claude and Llama it barely registers. Claude mostly repeats its last order, and Llama mostly follows the day of the game, which is that upward drift.

**[Scroll to the reasons chart]**

At the end, each agent wrote one sentence on how it decided. The top reasons are leftover risk and average demand. **[Click the orange "Lost sales" bar]** And here's what they actually wrote: "I settled on 11 units as a level that rarely left me short." Interesting: Claude under O talks about lost sales more, 52 percent versus 28 percent. So the scheme changes what Claude *says*, but not what it *orders*.

**[Point to the last line]**

And only **5 of 450** agents ever mention the money that moves at order time, the exact thing the theory is about. The agents simply solve it as a math problem.

## Step 9 · Conclusions and limits (60 s)

**[Click Next]**

**[Point to the left tile]** What we learned: neither model reproduced the human framing gap, even with human-like feedback, and the difference from people is statistically significant. Claude behaves like a textbook optimizer; Llama over-orders and drifts. Neither behaves like a person.

**[Point to the middle tile]** Our limits: agents of one model aren't truly independent people; there's no real money at stake; and results depend on the model and the wording. The same prompt even gave different first-day orders on different run dates.

**[Point to the right tile]** So our advice to researchers: AI agents are useful for screening ideas quickly, but confirm with real people. Always report the model, the prompt and the settings, and test whether the result survives a change in wording.

To answer our question: **people care about when the money moves. AI agents, so far, don't.**

Thank you. We're happy to take questions.

---

## Delivery tips

- Practise the clicks once. Wait about a second after each click before the next, or the page can jump back to step 1.
- Step 6 is the longest step (160 s). If time is short, skip the slider replay.
- Land the anchor numbers clearly: **11.7 vs 9.7**, **1.15 pooled**, **+0.03 Claude**, **93 percent at 10–11**, **5 of 450**.

## Questions that may come up

- **Why no neutral scheme S?** The 2023 replication compared only O and C, and we follow it.
- **Why no personas?** The study's instructions don't describe participants; invented backgrounds would add an untested variable.
- **Why does Claude start lower under O on day 1?** Before it has any data it guesses cautiously; within two or three days it settles on 10–11 under both schemes, so the 25-day average gap is near zero.
- **Why are fewer Llama agents in the comparison?** Following the original rule, agents that got the dice range (3 to 18) wrong are excluded. Six Llama C agents failed it; no Claude agent did.
- **Is +0.19 with human-like feedback a real effect?** It's not significant (the confidence interval crosses zero), and it is still significantly smaller than the human gap.
- **Can we see an agent playing?** Every agent's 25 days are saved in `data/state_*/`, and the game rules are in `sim/newsvendor_env.py`.
