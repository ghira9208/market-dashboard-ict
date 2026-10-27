# Research log

What was asked, what it showed, and where the evidence is. Newest first. Results are as of the date given; numbers
come from the files named in the last column (all under `.cache/` unless noted).

| Date | Question | Answer | Evidence |
|---|---|---|---|
| 2026-10-27 | Does stacking confirming layers (news-clear, higher-timeframe trend, calm volatility, EMA aligned) make trades safer? | No. Losing trades rise from 28% (0 layers) to 44% (4 layers); about +0.1R per extra layer. Calm volatility +0.63R, EMA aligned -0.62R, news-clear about 0, trend +0.2R | `layer_confluence.json`, `studies/layer_confluence_study.py` |
| 2026-10-27 | Can we get broker ticks? | Yes: live spot feed and at least 90 days of bid/ask history; EURUSD ticks about once a second; quotes only, no sizes. Recording to `~/dev/tick-data` | `ctrader_bridge.py`, `tick_recorder.py`, `tick_backfill.py` |
| 2026-10-21 | Do the detectors read the future? | Barely: expansion 100% of backtest touches reproduce with only live-visible bars, retracement-resumption 90%, expansion-liquidity 66%; the touches live would miss are the weaker ones | `causal_probe.csv`, `studies/causal_probe.py` |
| 2026-10-21 | Enter only after EMAs align / the streak is longer? | Worse. Longer streak -0.15R; EMA20/50 aligned -0.62R (CI -0.84..-0.42, every slice); waiting 3 bars for alignment -0.55R. Aligned-and-near-EMA about 0 | `streak_polish.json`, `studies/streak_polish_study.py` |
| 2026-10-21 | Is the opposite (EMA not aligned is better) real? | Forward test running: frozen spec, judged once at 400 forward trades with 100+ per group. Interim values swing (diff -0.12R at 82 trades, +0.49R at 182) and mean nothing yet | `ema_lead_forward_status.json`, `ema_lead_forward.py` |
| 2026-10-16 | Why did live trail the backtest? | Signals from the broker feed were acted on one bar late (the feed holds only completed bars). Fixed with a synthetic forming bar and bar-aligned caching. ATR-at-entry lookahead is real but tiny (+1.263R -> +1.252R) | memory note; `experiments.py` `atr_lag` |
| 2026-10-13 | Where does the live-vs-backtest gap come from? | The best combos could not trade at an $800 account (minimum lot); scale-out legs rounded up to minimum lots and doubled risk; about 0.35R per trade lost to spread and slippage | `edge_weights.py`, dashboard split view |
| 2026-10-09 | More strategy families | Four EMA families (3,772 scored): 0 survivors. Squeeze and round-number (1,801): one survivor, not tradable. 180-day time-series momentum: positive in all three slices but never significant, so it became a forward test | `ema_sweep_trials*.jsonl`, `pattern_sweep_trials*.jsonl`, `tsmom_*` |
| 2026-10-09 | Entry hour and higher-timeframe filters on the live trades | 05:00 UTC stronger in all three slices (now a size boost with an auto-revert audit); 16:00 UTC weaker but skipping it loses about 68R; higher-timeframe trend filter: no consistent effect | `overlay_study.json` |
| by 2026-10-20 | Would standing aside through scheduled news help? | No. Calendar blackout loses 420R (5m) and 2,270R (1h); even a perfect-hindsight shock blackout loses | `news_blackout_study.json` |
| by 2026-10-18 | Do quiet-volatility entries do better? | Backtest +0.82R per trade in both halves and both timeframes; the first 197 real trades (all entered a bar late) showed the opposite. A weekly audit reports at 40 and 100 quiet trades since the timing fix | `quiet_entry_audit.py` |
| by 2026-10-20 | Can a timing rule beat plain weekly buying for accumulation? | Five timing rules tested; none beat a fixed weekly basket | `accumulation_study.json`, `studies/accumulation*_study.py` |
| by 2026-10-14 | Is the backtest cost realistic? | Index CFDs: backtested R understates real cost by up to 80x because of per-side minimum commissions (flat 2.0 bps assumed). FX and crypto are close to proportional | `studies/validate_cost_model.py` |

Other study outputs without a headline above: `generic_pattern_screen.json`, `hourly_vol_profile.json`,
`scalp_winrate_study.json`, `news_positioning_study.json`, `target_variant_results.json`, `shock_report.txt`.
