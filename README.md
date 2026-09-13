# ICT Terminal

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
