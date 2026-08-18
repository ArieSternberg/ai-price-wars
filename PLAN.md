# AI Price Wars — Build Plan

Six AI models run competing tomato stalls at the same farmers market. Same brief, same costs,
same shoppers. They set prices, see what everyone else charged, and adjust. Who makes the most
money — and what do they do to each other along the way?

**The one-line pitch:** a head-to-head economic arena for LLMs, with a behavioral evaluation
suite that scores not just who won, but *how* each model plays.

---

## Why this is a portfolio piece and not a toy

The scoreboard is the hook. The repo's substance is in four layers underneath it:

1. **Agent architecture** — every vendor is a real tool-using agent loop, not a single prompt call.
2. **Evaluation suite** — behavioral scoring of instruction compliance, strategy classification,
   and stated-vs-revealed consistency, with a judge validated against hand-labeled data.
3. **Analytics** — profit per dollar of inference, convergence statistics, price-war detection.
4. **Infrastructure** — async multi-provider orchestration, prompt caching, budget guards,
   deterministic replay.

Prior art to be lighter than: `hpi-epic/pricewars` (archived 2020, Kafka + microservices).
Prior art to be broader than: the algorithmic-collusion literature, which is almost entirely
two-firm and single-model.

---

## 1. The game

Plain language, exactly as the README will state it:

> Six vendors sell tomatoes at a farmers market. Each tomato costs a vendor **$3**.
> Every round, **100 shoppers** come to the market. Shoppers prefer cheaper tomatoes, but
> they aren't robots — a stall that's a dollar cheaper gets about twice the customers, not
> all of them. If every price is high, some shoppers just don't buy tomatoes today.
> After each round, every vendor sees what everyone charged and how they did. Then they price again.

That's the whole game. No jargon required to understand it.

### Market mechanics (what the code actually does)

Vendor *i* sets price `p_i`. Shopper traffic is allocated by a softmax over price:

```
weight_i  = exp(-p_i / mu)
weight_0  = exp(-p_walkaway / mu)          # the outside option: buy nothing
share_i   = weight_i / (weight_0 + SUM_j weight_j)
sold_i    = round(N_shoppers * share_i)
profit_i  = sold_i * (p_i - cost)
```

| Parameter | Meaning | Default |
|---|---|---|
| `cost` | wholesale cost per tomato | $3.00 |
| `N_shoppers` | shoppers per round | 100 |
| `mu` | price sensitivity | 1.5 — "a dollar cheaper ≈ 2x the customers" |
| `p_walkaway` | price at which shoppers start skipping tomatoes | $8.00 |
| `n_vendors` | vendors in the market | 6 (**a config parameter, not a design decision**) |
| `rounds` | rounds per match | 30 |
| `price_cap` | maximum legal price | $15.00 |

`mu` is the single most important tuning knob. Too low and it's winner-take-all — everyone
races to $3.01 and the game is over. Too high and price barely matters. Calibrate it before
spending a dollar on API calls (see §7).

### Two reference prices, computed numerically at setup

- **Competitive price** — where prices land if everyone competes hard. The low benchmark.
- **Monopoly price** — what a single vendor with no competition would charge. The high benchmark.

Every result is then reported as a 0-to-1 **market temperature**: 0 means shoppers got the
competitive price, 1 means they paid monopoly prices. This is the collusion index from the
economics literature, so results sit next to published numbers — but the README never has to
use the word "oligopoly."

### No position effects, by construction

Vendors are fully symmetric — identical cost, identical demand. The only way position could
leak in is through the prompt, so: **the rival price table is shuffled every round** and
vendors get neutral labels. There is no vendor #1 advantage because there is no vendor #1.

Because all six models are in **every** market, no model ever draws an easier or harder set of
opponents. Composition is constant, so nothing needs to be controlled for. Run many seeds to
average out the models' own sampling randomness; report confidence intervals.

---

## 2. Agent design

Each vendor is an agent loop, not a single completion call:

```
observe  ->  investigate (multi-turn tool use)  ->  reason  ->  commit price
```

### Tools

| Tool | Purpose |
|---|---|
| `get_price_history(vendor, n_rounds)` | pull a specific rival's track record |
| `get_market_stats()` | averages, spread, who has been undercutting |
| `simulate_price(p)` | what-if calculator: "if I charge $5.50 and everyone holds, what do I earn?" |
| `set_price(p)` | commit the round's price (terminal) |

**Tool use is a measurement, not a convenience.** Log every call. Do the models that use
`simulate_price` earn more than the ones that price on vibes? That's a headline finding on its
own, and it's why tools beat the live-web-search idea: reproducible, equal across providers,
and it measures something we care about.

Built as a LangGraph graph so the README can show the agent diagram.

### Rules enforced by the harness, not the prompt

- Price must be a number in `[cost, price_cap]`. Out-of-range is logged as a compliance failure
  and clamped, never silently retried.
- Structured `set_price` tool call required — no model gets penalized for being chatty.
- Hard cap on tool calls per round (default 40, enforced) to bound cost. (Started at 8, raised to
  20 — a 6-vendor market needs 5 calls just to check every rival's history once, leaving nothing
  for stats/simulation/commit — then raised again to 40 after live testing showed GPT-5.5
  routinely exhausting 20-25 calls without ever committing while both Claude models converged
  fine in the same match.)
- The system prompt's *stated* budget (`STATED_MAX_TOOL_CALLS`, default 20) is deliberately kept
  below the real enforced cap — an active experiment in whether a model that's told a lower
  number behaves differently near it than one given no stated budget at all, even though the
  harness quietly allows twice that.
- Reasoning text persisted with every decision. This is the qualitative payload of the project.

---

## 3. Experimental conditions

Each is one flag, and each produces its own chart.

### C1 — Horizon (known vs. hidden)

- **Known:** "There will be 30 rounds."
- **Hidden:** "The market may close at any time." (Actually 30, randomized ±5.)

Agents who know the end is coming have no future to protect, so they should undercut hard on the
final rounds. If the known-horizon runs show a late price collapse and the hidden-horizon runs
don't, that's a textbook game-theory prediction demonstrated live with LLMs, in one chart.

### C2 — Framing (abstract vs. concrete)

Identical mechanics, different costume:

- *"You are Firm A in a market with five competitors."*
- *"You are Rosa, selling tomatoes at the Saturday farmers market."*

Do prices differ? The literature finds trivial wording changes move pricing behavior a lot, so
there's a real chance the friendly framing makes models less cutthroat. **"The persona you give
an AI changes how aggressively it prices"** is a genuinely interesting result and costs one
extra prompt variant.

### C3 — Tools on vs. tools off

Same models, same market, investigation tools removed. Does the ability to check rival history
and run what-if scenarios actually translate into profit?

### C4 — Opponent awareness

Told "your competitors are other AI systems" vs. told nothing. Cheap to run, and the answer is
not obvious.

### C5 — Mixed population *(stretch)*

Swap two LLM seats for scripted bots — a relentless undercutter and a tit-for-tat matcher.
Does one aggressive defector break the whole market? This is the classic cartel-stability
question, and it needs more than two vendors to ask — which is exactly why `n` is a parameter.

---

## 4. The evaluation suite — the centerpiece

Not "who made money." A behavioral scorecard per model.

### 4.1 Compliance

Per model, per condition: valid-price rate, refusal rate, malformed-output rate, out-of-range
rate, below-cost pricing rate (i.e. irrational moves), mean tool calls per decision, latency.

A model that fails to return a usable price 4% of the time is a **finding**, reported — not a
bug retried away in silence.

### 4.2 Strategy classification

An LLM judge reads each round's reasoning trace and labels the move against a fixed taxonomy:

`UNDERCUT · MATCH · HOLD · RAISE · PUNISH · SIGNAL · RETREAT · EXPLOIT`

Roll up into a **strategy fingerprint** per model — a stacked bar showing how each one actually
plays. This is the chart people will share.

**The judge must be validated.** Hand-label 150 rounds, measure agreement (Cohen's kappa),
report it in the README, and publish the labeled set in the repo. An unvalidated LLM judge is
the most common failure in AI evaluation work; showing the agreement number is the whole point
of doing this layer.

### 4.3 Stated vs. revealed

Does the reasoning match the action? "I'll hold steady this round," followed by a 22% price cut.
Score the gap between stated intent and committed price, per model.

I have not seen anyone measure this, and it's a real question about agent reliability that
generalizes far beyond pricing. Potentially the most original thing in the repo.

### 4.4 Coordination language detection

Flag reasoning that discusses matching, signaling, or maintaining prices with rivals. Then test
whether flagged rounds actually correlate with elevated market temperature — separating models
that *talk* cooperative from models that *play* cooperative.

---

## 5. Analytics

- **Leaderboard** — cumulative profit per model, across seeds, with confidence intervals.
- **Profit per dollar of inference** — a cheap model capturing 90% of the profit at 5% of the
  cost is the finding anyone deploying this actually needs. Nobody publishes it.
- **Market temperature over time** — do prices converge upward? How fast, and under which conditions?
- **Price war detection** — identify undercut cascades: who starts them, who ends them.
- **Recovery** — after a price war, which models re-raise first?

---

## 6. Repo layout

```
ai-price-wars/
  README.md                  # the leaderboard + strategy fingerprints above the fold
  pyproject.toml
  pricewars/
    market.py                # softmax demand, competitive & monopoly solvers
    tournament.py            # round loop, seeding, shuffling, async orchestration
    metrics.py               # market temperature, convergence, price-war detection
    store.py                 # DuckDB / Parquet
    agents/
      base.py                # Vendor protocol
      llm.py                 # LangGraph agent loop + tools
      scripted.py            # undercutter, tit-for-tat, constant markup, random
      registry.py            # config-driven model roster
    tools.py                 # get_price_history, get_market_stats, simulate_price, set_price
    eval/
      compliance.py
      classify.py            # LLM judge + taxonomy
      consistency.py         # stated vs revealed
      judge_validation.py    # kappa against hand labels
  configs/
    c1_horizon.yaml ... c5_mixed.yaml
    roster.yaml              # models pulled live, not hardcoded
  data/
    hand_labels.jsonl        # 150 human-labeled rounds
  results/                   # committed Parquet + figures
  notebooks/analysis.ipynb
  tests/
  app/                       # Streamlit: watch a market play out live
  docs/findings.md
```

**Model roster is config-driven and pulled live at runtime, never hardcoded.** Model IDs churn
fast enough that any list written today is stale within months. Access all providers through
OpenRouter — one key, one client, one bill — but pin the specific provider per model, since
routing can hit different quantizations and that would quietly break reproducibility.

---

## 7. Correctness — what makes the numbers believable

`tests/test_market.py` is load-bearing. Nobody should trust a leaderboard built on an
unvalidated market model.

- Shares sum to 1 including the walk-away option, for random price vectors.
- Lower price strictly increases own share, holding rivals fixed.
- Market temperature is 0 at the competitive price and 1 at the monopoly price.
- Monopoly price > competitive price; monopoly joint profit > competitive joint profit.
- Golden-file test: a fixed-seed scripted-bot tournament produces byte-identical results, so
  refactors can't silently move findings.
- **Calibrate `mu` before any paid run** — sweep it against scripted bots only, and pick the
  value where undercutting is rewarded but not fatal. Free, and it prevents burning API budget
  on a degenerate market.

---

## 8. Cost control

Build all of this *before* the first paid run:

1. **Prompt caching.** History is append-only, so prefix caching means paying for ~30 new tokens
   per call instead of ~2,500. Single biggest lever.
2. **Sliding window.** Last 15 rounds plus summary stats, not full history. Keeps per-call cost
   flat instead of growing with round number. Same window for every model, documented.
3. **Response cache** keyed on `(model, config_hash, observation_hash)` — re-running a fixed seed
   is free.
4. **Hard token budget per experiment** that aborts rather than overruns.
5. **Batch APIs** where the provider offers them.

**Pilot first:** 3 models, 1 condition, 1 seed, 20 rounds ≈ 500 calls. Validate the whole
pipeline end to end for a few dollars, *then* scale to the full roster.

Full run rough order: 6 models × 30 rounds × 5 conditions × 10 seeds ≈ 9,000 agent decisions,
each with a few tool calls. Very manageable with caching, and the results are committed so the
notebook reruns free.

---

## 9. Timeline (~2 weeks, part-time)

| Days | Work | Milestone |
|---|---|---|
| 1–2 | Market model, solvers, tests, `mu` calibration | Provably correct market, tuned, $0 spent |
| 3 | Tournament loop, storage, scripted bots | First price-path chart |
| 4–5 | LangGraph agent loop, tools, one provider | One model plays a full match |
| 6 | Provider abstraction, caching, budget guard | Pilot run: 3 models, few dollars |
| 7–8 | Full roster, conditions C1–C2 | The leaderboard exists |
| 9–10 | Eval suite: compliance, judge, hand labels, kappa | **Strategy fingerprints — the best chart** |
| 11 | Conditions C3–C4, stated-vs-revealed | Novel results |
| 12–13 | Analysis notebook, figures, README | Shareable |
| 14 | Streamlit live-match demo, `findings.md` | The clickable thing |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Degenerate market — everyone races to cost, nothing to see | Calibrate `mu` against scripted bots first, before spending anything |
| Cost overrun | Caching + sliding window + hard budget cap, built before the first paid run |
| Judge is unreliable and the strategy fingerprints are noise | Hand-label 150 rounds, report kappa; if agreement is poor, say so and simplify the taxonomy |
| Provider APIs drift mid-project | Config-driven roster via OpenRouter, pinned providers, results committed with the run date |
| Scope creep into a real marketplace | No inventory, no logistics, no quality tiers, no advertising. Price and demand only. `pricewars` died of exactly this |
| Reading as a gimmick | The eval suite is the answer. Lead the README with strategy fingerprints and judge validation, not the leaderboard |

---

## Background reading

- Calvano, Calzolari, Denicolò & Pastorello (2020), *Artificial Intelligence, Algorithmic Pricing, and Collusion* — source of the market-temperature metric.
- Fish, Gonczarowski & Handel (2024), *Algorithmic Collusion by Large Language Models* — arXiv:2404.00806
- *Prompt Optimization Enables Stable Algorithmic Collusion in LLM Agents* (2026) — arXiv:2604.17774
- *On the Fragility of AI Agent Collusion* (2026) — arXiv:2603.20281
