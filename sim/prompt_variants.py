"""Alternative game-instruction wordings for a prompt-sensitivity test.

"base" is the wording used in the main study (sim/setup_agents.py INSTR). The two variants below
are the instructor-style wordings supplied by the team. Only the C text was given; the matching O text is
DRAFTED here by flipping the payment direction and keeping every other sentence the same. Check the O drafts
against the course materials before relying on them.

Only one typo from the pasted text was fixed ("self" -> "sell" in v2-C). Everything else is verbatim.
The payoff engine is unchanged in every variant (bankruptcy is described in v2 but not enforced, because a
25-day game starting at 100 play dollars practically cannot reach it).
"""

TAIL = "\n\nYou begin with 100 play dollars. Your goal is to maximize your play dollars by the end of the 25 days."

V1 = {
"C": """In this simplified game, you own a business that makes money by selling widgets for 25 simulated days.

* Deciding Order Quantity: At the beginning of each simulated day you have to choose how many widgets to order.
* +1 Per Unit Ordered: For every widget ordered, each customer will pay you 1 play dollar for a widget. However, you pay back 2 dollars at the end of each day for each unsold widget. On the other hand, if you don’t stock enough widgets to satisfy your customers, they will leave disappointed and you will miss out on the money you could have made for you.
* Random Demand: Then roll the 3 dice to determine demand for that day.
* Start Empty for Next Day: Then discard all leftover units from your store and start empty for the beginning of the next day. Any customers you didn’t have enough widgets for will not be back the next day and demand will be rerolled with a new three dice.""" + TAIL,
# DRAFTED counterpart (flip of the payment direction only)
"O": """In this simplified game, you own a business that makes money by selling widgets for 25 simulated days.

* Deciding Order Quantity: At the beginning of each simulated day you have to choose how many widgets to order.
* -1 Per Unit Ordered: For every widget ordered, you pay 1 play dollar for a widget. However, you receive 2 dollars at the end of each day for each sold widget. On the other hand, if you don’t stock enough widgets to satisfy your customers, they will leave disappointed and you will miss out on the money you could have made for you.
* Random Demand: Then roll the 3 dice to determine demand for that day.
* Start Empty for Next Day: Then discard all leftover units from your store and start empty for the beginning of the next day. Any customers you didn’t have enough widgets for will not be back the next day and demand will be rerolled with a new three dice.""" + TAIL,
}

V2_GOAL = ("\n\nYou begin with 100 play dollars. Your goal is to maximize your play dollars by the end of the 25 days. ")
V2 = {
"C": """In this simplified game, you own a business that makes money by selling widgets for 25 simulated days.

* Deciding Order Quantity: At the beginning of each simulated day you have to carefully determine how many widgets to have available for sale in your store that day.
* +1 Per Unit Ordered: For every widget ordered, you receive a profit of 1 play dollar and place the widgets in your store. However, you MUST pay a 2 dollar penalty at the end of each day for each unsold widget at the end of each day. If you cannot pay it back you will be BANKRUPT and unable to order more widgets. If this happens, the experiment will end in FAILURE.
* Random Demand: Then roll the 3 dice to determine demand for that day.
* Start Empty for Next Day: Then discard all leftover units from your store and start empty for the beginning of the next day. Any widgets you failed to sell because you over estimated demand will be wasted and you will be penalized.""" + V2_GOAL +
"This means you cannot over order widgets or you could potentially end up with less dollars than you started with.",
# DRAFTED counterpart (flip of the payment direction only; the closing warning is kept parallel)
"O": """In this simplified game, you own a business that makes money by selling widgets for 25 simulated days.

* Deciding Order Quantity: At the beginning of each simulated day you have to carefully determine how many widgets to have available for sale in your store that day.
* -1 Per Unit Ordered: For every widget ordered, you pay 1 play dollar and place the widgets in your store. However, you receive 2 dollars at the end of each day for each sold widget. If you cannot afford to pay for the widgets you order you will be BANKRUPT and unable to order more widgets. If this happens, the experiment will end in FAILURE.
* Random Demand: Then roll the 3 dice to determine demand for that day.
* Start Empty for Next Day: Then discard all leftover units from your store and start empty for the beginning of the next day. Any widgets you failed to sell because you over estimated demand will be wasted and you will have paid for them.""" + V2_GOAL +
"This means you cannot over order widgets or you could potentially end up with less dollars than you started with.",
}

VARIANTS = {"v1": V1, "v2": V2}

# v3 = the main-study wording with ONE change: the incentive sentence pays in API token spend instead of real
# dollars (supplied by the team). C is verbatim from the pasted text; O is the same text with the O payoffs
# from the main study. Because the incentive is now inside the instructions, the human "$5 + bonus" paragraph
# is dropped from the prompt for this variant (see NO_HUMAN_BONUS).
V3_GOAL_OLD = "Your goal is to maximize your play dollars by the end of the 25 days."
V3_GOAL_NEW = (V3_GOAL_OLD + " For every $10 play dollars over $100 you make at the end of 25 rounds we will allocate "
               "and additional $0.01 in token spend to your session API.")
NO_HUMAN_BONUS = {"v3"}
HUMAN_BONUS = ("You will receive $5 minimum for participating plus a bonus equal to 1 real dollar for every 50 play dollars "
               "you earn by the end of the game (rounded to the nearest dollar). You cannot receive a negative bonus, "
               "even if you lost money.")
BULLETS = ("Deciding Order Quantity:", "+1 Per Unit Ordered:", "-1 Per Unit Ordered:", "Random Demand:",
           "-2 Per Unit Unsold:", "+2 Per Unit Sold:", "Start Empty for Next Day:")


def _v3(base):
    out = {}
    for cond, text in base.items():
        lines = ["* " + l if l.startswith(BULLETS) else l for l in text.split("\n")]
        out[cond] = "\n".join(lines).replace(V3_GOAL_OLD, V3_GOAL_NEW)
    return out


def strip_bonus(text, variant):
    """Remove the human-participant bonus sentence for variants that carry their own incentive."""
    return text.replace(" " + HUMAN_BONUS, "") if variant in NO_HUMAN_BONUS else text


def get_instr(variant, base):
    """Return the {'O':..., 'C':...} instruction dict for a variant name ('base' = main study wording)."""
    if variant in (None, "", "base"):
        return base
    if variant == "v3":
        return _v3(base)
    if variant not in VARIANTS:
        raise SystemExit(f"Unknown variant {variant!r}; choose base, v3, " + ", ".join(VARIANTS))
    return VARIANTS[variant]
