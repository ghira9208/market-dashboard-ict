# market-dashboard-ict — the research engine

Where strategies are found, tested and proven (or thrown out). The live trading platform is the sibling repo
`~/dev/edge-pipeline`; it imports a handful of modules from here and relies on the leaderboard this repo produces.
Nothing in this repo places an order: `ctrader_bridge.py` only relays the platform's orders to cTrader.

## Map

```
experiments.py          detectors' event builders, _simulate_fixed_rr, the permutation/holdout/locked-slice machinery, the trial ledger
detectors.py, indicators.py, data.py, news.py, institutional.py     detection, indicators, OHLCV fetch + caches, calendar, derivatives data
ticker_behavior.py      builds ticker_behavior.db: the leaderboard, lifecycle states, auto-retire, fail-closed identity
strategy_identity.py    strategy_id = hash of everything that changes P&L
observatory.py          per-combo trade detail, regime matrix, rolling stability, parameter grid (read-only)
ema_strategies.py, pattern_strategies.py, shock_strategies.py, tsmom_backtest.py   other strategy families (none promoted so far)
revalidate_combos.py    weekly decay check -> .cache/decay_flags.json           [scheduled]
tsmom_forward.py, ema_lead_forward.py     frozen-spec forward tests, never trade  [scheduled]
validate_target_variants.py     per-combo target-distance validation (writes to the main ledger)
overlay_tools.py        shared helpers for overlay studies and the forward test
universe.py             the researched universe and the data-fetch recipe
ctrader_bridge.py, ctrader_tokens.py, ctrader_auth.py, tick_recorder.py, tick_backfill.py   broker connection, tokens, ticks   [bridge scheduled]
studies/                one-off research studies; each writes its result to .cache/<name>.json
sweeps/                 discovery sweeps (isolated ledgers) and their report scripts
config/                 small tracked settings (tick_symbols.json)
docs/                   PIPELINE.md (how evidence is judged), RESEARCH_LOG.md (what was tried and what it showed)
tests/                  pytest, ~240 tests
archive/                retired code (eToro client, early pages)
app.py, crypto_app.py, news_app.py, institutional_app.py, ...   the legacy Streamlit dashboards (below)
```

Run anything from the repo root. Scripts in `studies/` and `sweeps/` find the repo root themselves and import the core
modules by name; run them as `python3 studies/<name>.py`.

## Rules this repo lives by

- **Isolated ledgers.** Exploratory work writes to its own ledger (`.cache/<name>_trials.jsonl`), never the main
  `experiment_trials.jsonl`: adding hundreds of exploratory trials to the main ledger would raise its
  Benjamini-Hochberg denominator and could demote strategies that are already live.
- **Freeze before looking.** A forward test writes its spec (rules, universe, verdict bar) with a hash first; the script
  refuses to run if the spec is edited afterwards.
- **Out-of-sample or it did not happen.** Rules are fixed in advance and judged on later slices; anything found by
  looking at all the data is a lead, not evidence.
- **Changing P&L-relevant logic changes `strategy_id`** and the live bot halts that strategy until it is re-validated.

## Data policy

| Tracked in git | Not tracked |
|---|---|
| main ledger, leaderboard DB, decay flags, study results, isolated sweep ledgers, forward-test specs and ledgers, small history logs | OHLCV caches, archived broker candles, OAuth tokens, backups, logs, ticks (`~/dev/tick-data`) |

Old backups live in `~/dev/edge-pipeline/backups/research-archive/`. The GitHub Action `collect.yml` commits fresh
snapshots of three history files and the intraday futures archive to `origin/main` every 6 hours.

## Tests

```bash
python3 -m pytest tests -q
```

`tests/conftest.py` puts `studies/` and `sweeps/` on the import path so tests can import those scripts by name.

## The Streamlit dashboards (legacy, not running)

Two Streamlit dashboards for ICT (Inner Circle Trader) chart analysis —
auto-detected Fair Value Gaps, Order Blocks, Swing Points, Equal Highs/Lows,
Market Structure, Premium/Discount, and Liquidity levels on a
TradingView-style interactive candlestick chart, plus an on-chart "best
trade right now" Entry/Stop/Target read.

- **Markets** (`app.py`, port 8501) — highly liquid markets only: forex
  majors, major indices, commodities.
- **Crypto** (`crypto_app.py`, port 8506) — Bitcoin/crypto, with a live
  Binance WebSocket tick feed on top of the usual polling.

Both are independent Streamlit scripts that share the same underlying
modules (`detectors.py`, `theme.py`, `data.py`, `recommender.py`, `ict_chart/`)
and differ only in which curated symbol list their own ticker picker
shows.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501          # Markets
streamlit run crypto_app.py --server.port 8506    # Crypto
```

Or via the Claude Code launch config: `.claude/launch.json`.

## Data sources

Yahoo Finance (via `yfinance`) for OHLCV, no API key required. Binance's
public WebSocket feed adds live ticks on top for crypto tickers (Crypto
dashboard). `data.py` also has an optional Twelve Data/Tiingo fallback
chain for when Yahoo is unavailable.

## Files

- `app.py` / `crypto_app.py` — the two pages: controls, ICT detection
  wiring, and the custom interactive chart component.
- `detectors.py` — FVG/Order Block/Swing/Structure/Liquidity/Equilibrium
  detection, shared by both pages.
- `recommender.py` — ranks currently-open zones into the on-chart "best
  trade right now" Entry/Stop/Target lines, reusing `research/setups.py`'s
  setup math and Edge Lab's validation lookup where it exists.
- `theme.py` — the neon dark-mode styling + nav.
- `data.py` — OHLCV fetch/cache/provider chain.
- `ict_chart/` — the custom lightweight-charts Streamlit component.
- `research/`, `edge_lab/` — backing library code `recommender.py` depends
  on (event-study math, Edge Lab's validated-edge lookup); not runnable
  pages on their own anymore, see `archive/`.

## Archive

`archive/` holds the project's earlier pages (Home, Research Lab, Edge
Lab, Setups) and their supporting scheduled tasks — removed from active
use, kept in case any of it's wanted again. Not wired into the running
apps; nothing currently imports from `archive/`.
