# Speech: Krishna, dashboard steps 1–4 (about 3.5 minutes)

Bracketed lines are cues, not words to say.

---

## Step 1 · The business problem

**[Dashboard on step 1]**

Hi everyone, we're Krishna, Michael and Dale.

Let me start with a decision every store manager faces. You have to order stock *before* you know how many customers will show up. Order too little and you lose sales. Order too much and you're stuck with leftovers. That's the classic newsvendor problem.

But there's a detail that usually gets ignored: *when* the money moves. Sometimes you pay for the stock up front. Sometimes a customer or partner pays you first, and you pay back for whatever doesn't sell. If the profit works out exactly the same either way, it shouldn't change anything. A rational manager would order the same amount.

So our question is: **people turn out to care about that timing — do AI agents too?** That matters because companies and researchers are starting to use AI agents as cheap stand-ins for human participants, and that only works if the agents make the same mistakes we do.

## Step 2 · The original experiment

**[Click Next]**

Our project replicates a 2013 Management Science paper by Chen, Kök and Tong.

They had 99 students at Duke run a small widget shop for 25 days, starting with 100 play dollars. Every day they picked an order, then rolled three real dice, and the total was that day's demand: anywhere from 3 to 18, about 10.5 on average. They used poker chips and paper money, and got paid real dollars based on how well they did.

The key difference was how the money moved. Under **own financing, O**, you pay one dollar per unit when you order, and get two dollars for every unit you sell. Under **customer financing, C**, you *receive* a dollar per unit when you order, and pay two dollars for every unit you don't sell.

**[Point to the formula]**

This line shows the important part: the math is identical. Any order, any dice roll, same profit. The best order under both is 10 or 11. The paper also had a neutral scheme where all the money moved after demand, but the replications, and we, compare O and C.

## Step 3 · What people did, and why

**[Click Next]**

And yet people didn't behave the same. In the original study, students ordered about 11.7 units under O and 9.7 under C: a two-unit gap from payment timing alone.

**[Point to the four replication numbers]**

In 2023, researchers at Michigan and UT Dallas ran the study again three times. They found gaps of 0.9, 1.4 and 1.1: smaller than the original, but the same direction every time. Pooled, the gap is **1.15 units**. Those 154 people are the human benchmark we compare our AI agents against.

Why does it happen? The authors call it **prospective accounting**. People give less weight to money that changes hands at the moment they order, before they know what demand will be. They estimate people count only about 70 to 80 percent of it. Under O that money is a cost, so ordering feels cheaper, and people order more. Under C it's income, so ordering feels less rewarding, and people order less. The neutral scheme landed right at the optimum, and the authors ruled out other explanations like loss aversion.

## Step 4 · Same task, same profit

**[Click Next]**

Here's one example day. You order 11 units, and the dice give a demand of 9.

**[Point to the left card]** Under own financing, you pay 11 up front and get 18 back for the 9 you sell. That's plus 7.

**[Point to the right card]** Under customer financing, you receive 11 up front, and pay a 4-dollar penalty for the 2 you didn't sell. Also plus 7.

Same order, same dice, same profit. Only the timing of the money differs. This is exactly the task the participants had, and the task we gave our AI agents. Michael's going to show you how we built them.

**[Click Next and hand over to Michael]**

---

## Delivery tips

- **Step 2:** slow down on the two schemes and pause after each one. It's the one part everyone has to follow.
- **Step 3:** land two numbers clearly, "11.7 versus 9.7" and "1.15 pooled". Michael and Dale compare everything against them.
- **Step 4:** point at each card as you say its number, so the audience sees both reach +7.

## Questions that may come to you

- **Why not test the neutral scheme S?** The 2023 replication compared only O and C, and we follow the replication.
- **Why no personas for the agents?** The study's prompt doesn't describe participants, and invented backgrounds would add a variable we couldn't justify.
- **Can we see the game or an agent playing?** We kept the talk to the results. The game rules are in `sim/newsvendor_env.py`, and every agent's 25 days are saved in `data/state_*/`, so any agent can be shown afterwards.
