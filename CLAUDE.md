# AI Price Wars — Project Context

Six AI models run competing tomato stalls at the same farmers market. Same brief, same costs,
same shoppers. They set prices, see what everyone charged, and adjust.

**Read `PLAN.md` first** — it is the full spec. This file records *why* the decisions in it were
made, so they don't get re-litigated.

**Author context:** this is a portfolio project for Arie Sternberg (AI Engineer / AI PM /
Analyst roles). Code quality, test coverage, and README polish matter as much as the results —
a hiring manager gives a repo about 90 seconds. Every phase should end in something committable
and demoable.

---

## Settled decisions — do not re-open without a reason

### Framing

**Farmers market, not "firms."** The mechanics are standard differentiated-Bertrand logit
demand, but "oligopoly" and "firm" are inherited jargon from the economics papers and make the
README unreadable to most people. The code can be precise; the README says "shoppers prefer
cheaper tomatoes but aren't robots." Keep the two registers separate.

**"Market temperature" is the collusion index.** Same 0-to-1 metric as Calvano et al., renamed.
0 = shoppers got a competitive price, 1 = they paid monopoly prices. Renaming keeps
comparability to published numbers without the jargon.

### Market design

**`n_vendors` is a config parameter, never hardcoded.** The logit model generalizes to any n for
free. Making it a parameter turns "2 vendors vs 6" into an experiment instead of an
architectural commitment, and it's what enables the mixed-population condition (C5).

**All models play in every market, with the rival table shuffled each round.** This eliminates
both artifacts at once: no model ever draws an easier set of opponents (composition is constant
by construction), and no model benefits from prompt position. Vendors are fully symmetric —
identical cost, identical demand.

**Do NOT build plus-minus / Elo attribution.** It was considered and rejected as unnecessary
given the fixed roster. It only earns its place if the roster later outgrows what fits in one
market, or if market size is varied within a single comparison. Back pocket only.

**`mu` calibration comes before any paid API run.** If price sensitivity is too high the market
is degenerate — everyone races to $3.01 and there is no game. This is the single most likely way
the project fails. Sweep `mu` against scripted bots, which is free, in the first two days.

### Agent design

**Never give the agents a strategy menu.** Handing them a list of strategies measures the menu,
not the model. What each model *invents* is the result, and comparing invented strategies across
models is the interesting part.

**Tools query the simulation, not the web.** Live web search was explicitly rejected:
non-reproducible (the web changes between runs), unequal across providers, slow, and it mostly
retrieves generic pricing boilerplate. Simulation tools (`get_price_history`, `get_market_stats`,
`simulate_price`) are reproducible, identical for every model, and let us measure whether
investigation actually correlates with profit.

**Compliance failures are logged and reported, never silently retried.** A model that fails to
return a valid price 4% of the time is a finding. Out-of-range prices get clamped *and* recorded.

### Rounds and horizon

**30 rounds, not 5.** Five rounds measures nothing — round 1 is a blind guess, 2–4 are noise, and
round 5 is a knife fight. Retaliation, price wars, and convergence all need repetition to emerge.

**Known-horizon endgame defection is a feature, not a bug.** Agents told "there will be 30 rounds"
should undercut hard at the end because there's no future left to protect. That's the point of
condition C1 — comparing it against a hidden horizon demonstrates a textbook game-theory
prediction with live LLMs.

### Evaluation

**The eval suite is the portfolio centerpiece, not the leaderboard.** Lead the README with
strategy fingerprints and judge validation. The leaderboard is the hook; the eval layer is what
makes this a credible piece of work rather than a gimmick.

**The judge must be validated or the eval layer is worthless.** Hand-label 150 rounds, report
Cohen's kappa in the README, commit the labeled set. An unvalidated LLM judge is the most common
failure in AI eval work — showing the agreement number is the entire reason this layer exists.
If agreement is poor, say so and simplify the taxonomy rather than hiding it.

### Infrastructure

**Model roster is config-driven and resolved at runtime.** Model IDs churn fast; any hardcoded
list is stale within months. Route through OpenRouter (one key, one client, one bill) but pin the
specific provider per model — routing can hit different quantizations, which would quietly break
reproducibility.

**Caching and budget guards ship before the first paid run, not after.** History is append-only,
so prefix caching means ~30 new tokens per call instead of ~2,500. Sliding window of 15 rounds
keeps per-call cost flat. Hard token budget aborts rather than overruns.

**Pilot before scaling.** 3 models, 1 condition, 1 seed, 20 rounds ≈ 500 calls. Validate the
whole pipeline for a few dollars before touching the full roster.

---

## Scope guards

`hpi-epic/pricewars` (the archived 2020 prior art) died of scope creep. Do not add:

- inventory, restocking, or supply constraints
- logistics, shipping, or delivery
- product quality tiers or differentiation beyond price
- advertising or promotions
- Kafka, microservices, or any message broker

**Price and demand only.** If a feature doesn't serve a chart in `docs/findings.md`, it doesn't
ship.

---

## Build order

Do not start agents before the market model is tested and calibrated.

1. **Market model** — `market.py`, softmax demand, competitive/monopoly solvers, full test suite,
   `mu` calibration sweep. No AI, no API spend.
2. **Tournament loop** — round orchestration, seeding, shuffling, Parquet storage, scripted bots.
   First price-path chart.
3. **Agent loop** — LangGraph graph, tools, one provider end to end.
4. **Provider abstraction** — roster config, caching, budget guard. Pilot run.
5. **Full roster + conditions C1–C2** — the leaderboard exists.
6. **Eval suite** — compliance, judge, hand labels, kappa. Strategy fingerprints.
7. **Conditions C3–C4**, stated-vs-revealed.
8. **Analysis notebook, README, Streamlit demo, `findings.md`.**

## Conventions

- Python 3.11+, `pyproject.toml`, `pytest`.
- DuckDB + Parquet for run logs. **Results are committed** so the analysis notebook reruns
  without re-spending money.
- LangGraph for the agent loop (also gives the README an architecture diagram).
- Streamlit for the live-match demo.
- Every run is reproducible from a seed. Golden-file test guards against silent result drift.
- Secrets in `.env`, never committed.
