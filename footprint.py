"""
Real footprint chart (price-by-volume, per candle, from actual Binance
trade-by-trade data) — Crypto only, since it needs Binance's public trade
feed, not a Yahoo OHLCV bar. Split out of backtest_ui.py, which this used
to share a file with purely for historical reasons (both got rendered
inside the same ☰ drawer at different points): this has nothing to do with
backtesting or the FVG/OB sweep engine — no shared state, no shared
imports beyond the basics, just two unrelated features that happened to
live in one file.

Called from app.py / crypto_app.py's own bottom-of-page "Footprint chart"
expander (a different location entirely from the ☰ drawer's Strategy/
Backtest tabs that backtest_ui.py still owns).
"""
import json
import uuid

import pandas as pd
import requests
import streamlit as st


def _binance_symbol(yahoo_ticker):
    return yahoo_ticker.replace("-USD", "").upper() + "USDT"


@st.cache_resource(show_spinner=False)
def _binance_session():
    """One pooled HTTP connection reused across every Binance call in this
    server process. Plain requests.get() opens a fresh TCP+TLS connection
    per call -- for a 150-call aggTrades pagination run that overhead alone
    measured out to well over a minute of pure connection setup, on top of
    the actual data transfer. A Session reuses the connection instead."""
    s = requests.Session()
    s.headers.update({"Connection": "keep-alive"})
    return s


@st.cache_data(ttl=20, show_spinner=False)
def _fetch_klines(symbol, interval, n_candles):
    resp = _binance_session().get("https://api.binance.com/api/v3/klines",
                                   params={"symbol": symbol, "interval": interval, "limit": n_candles}, timeout=10)
    resp.raise_for_status()
    return [{"open_time": r[0], "close_time": r[6], "open": float(r[1]), "high": float(r[2]),
              "low": float(r[3]), "close": float(r[4])} for r in resp.json()]


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_agg_trades(symbol, start_ms, end_ms, max_calls):
    """Paginates Binance's aggTrades via fromId, 1000 trades/call. A quiet
    pair (DOT: ~10 trades/candle-minute, measured directly) finishes in one
    call regardless of window size; a busy one (BTC: ~310 trades/candle-
    minute, measured directly — 150 1m-candles took 47 calls / 46.5k
    trades / 21.7s for real) can hit max_calls before reaching end_ms.
    Returns (trades, truncated) rather than silently handing back a
    partial window that LOOKS complete — the caller must tell the user."""
    session = _binance_session()
    all_trades = []
    resp = session.get("https://api.binance.com/api/v3/aggTrades",
                        params={"symbol": symbol, "startTime": start_ms, "limit": 1000}, timeout=10)
    resp.raise_for_status()
    batch = resp.json()
    all_trades.extend(batch)
    calls = 1
    truncated = False
    while batch and batch[-1]["T"] < end_ms:
        if calls >= max_calls:
            truncated = True
            break
        resp = session.get("https://api.binance.com/api/v3/aggTrades",
                            params={"symbol": symbol, "fromId": batch[-1]["a"] + 1, "limit": 1000}, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_trades.extend(batch)
        calls += 1
    return [t for t in all_trades if t["T"] <= end_ms], truncated


def _compute_value_area(totals_by_idx, poc_idx, va_pct=0.70):
    """Standard market-profile value-area algorithm: start at the point of
    control and expand outward one tick at a time, always adding whichever
    neighboring row (above or below the current range) has more volume,
    until the accumulated volume covers va_pct of the candle's total."""
    total = sum(totals_by_idx.values())
    if total <= 0:
        return poc_idx, poc_idx
    target = total * va_pct
    idxs = sorted(totals_by_idx.keys())
    pos = idxs.index(poc_idx)
    lo, hi = pos, pos
    acc = totals_by_idx[poc_idx]
    while acc < target and (lo > 0 or hi < len(idxs) - 1):
        vol_below = totals_by_idx[idxs[lo - 1]] if lo > 0 else -1
        vol_above = totals_by_idx[idxs[hi + 1]] if hi < len(idxs) - 1 else -1
        if vol_above >= vol_below:
            hi += 1
            acc += totals_by_idx[idxs[hi]]
        else:
            lo -= 1
            acc += totals_by_idx[idxs[lo]]
    return idxs[lo], idxs[hi]


def _detect_imbalance_stacks(levels_asc, ratio=3.0, min_vol=0.02):
    """Diagonal bid/ask imbalance, the standard order-flow convention: a
    price level's BUY volume is compared to the SELL volume one tick BELOW
    it (a buyer lifting the offer at this price vs. a seller who was
    passively offering one tick down), and vice versa for SELL imbalance
    one tick above. A level clearing `ratio` in either direction is
    "imbalanced"; three or more consecutive imbalanced levels on the same
    side is a "stack" — the classic footprint absorption/exhaustion
    signal. levels_asc: list of (tick_idx, buy, sell) sorted ascending."""
    n = len(levels_asc)
    buy_flag = [False] * n
    sell_flag = [False] * n
    for i, (_, buy, sell) in enumerate(levels_asc):
        if i > 0:
            sell_below = levels_asc[i - 1][2]
            if buy >= min_vol and buy > sell_below * ratio:
                buy_flag[i] = True
        if i < n - 1:
            buy_above = levels_asc[i + 1][1]
            if sell >= min_vol and sell > buy_above * ratio:
                sell_flag[i] = True

    def stacks(flags):
        out = [False] * n
        i = 0
        while i < n:
            if flags[i]:
                j = i
                while j < n and flags[j]:
                    j += 1
                if j - i >= 3:
                    for k in range(i, j):
                        out[k] = True
                i = j
            else:
                i += 1
        return out

    return stacks(buy_flag), stacks(sell_flag)


def _build_footprint_payload(symbol, interval, klines, trades, tick, imbalance_ratio):
    def tick_idx(p):
        return round(p / tick)

    trades_sorted = sorted(trades, key=lambda t: t["T"])
    ti = 0
    candles_out = []
    profile_totals = {}  # tick_idx -> {"buy","sell"}
    cum_delta = 0.0

    for k in klines:
        cell = {}
        while ti < len(trades_sorted) and trades_sorted[ti]["T"] <= k["close_time"]:
            t = trades_sorted[ti]
            if t["T"] >= k["open_time"]:
                idx = tick_idx(float(t["p"]))
                r = cell.setdefault(idx, {"buy": 0.0, "sell": 0.0})
                qty = float(t["q"])
                if t["m"]:
                    r["sell"] += qty
                else:
                    r["buy"] += qty
                pr = profile_totals.setdefault(idx, {"buy": 0.0, "sell": 0.0})
                if t["m"]:
                    pr["sell"] += qty
                else:
                    pr["buy"] += qty
            ti += 1

        base = {
            "time": pd.to_datetime(k["open_time"], unit="ms").strftime("%H:%M"),
            "open": k["open"], "high": k["high"], "low": k["low"], "close": k["close"],
        }
        if not cell:
            candles_out.append({**base, "levels": [], "poc_idx": None, "va_low_idx": None,
                                 "va_high_idx": None, "total_buy": 0.0, "total_sell": 0.0,
                                 "delta": 0.0, "cum_delta": round(cum_delta, 4)})
            continue

        totals = {idx: v["buy"] + v["sell"] for idx, v in cell.items()}
        poc_idx = max(totals.items(), key=lambda kv: kv[1])[0]
        va_low_idx, va_high_idx = _compute_value_area(totals, poc_idx)
        idxs_asc = sorted(cell.keys())
        levels_asc = [(i, cell[i]["buy"], cell[i]["sell"]) for i in idxs_asc]
        buy_stacks, sell_stacks = _detect_imbalance_stacks(levels_asc, ratio=imbalance_ratio)

        levels_out = []
        for pos, (idx, buy, sell) in enumerate(levels_asc):
            levels_out.append({
                "idx": idx, "price": round(idx * tick, 8), "buy": round(buy, 4), "sell": round(sell, 4),
                "buy_imb": buy_stacks[pos], "sell_imb": sell_stacks[pos],
            })
        total_buy = sum(v["buy"] for v in cell.values())
        total_sell = sum(v["sell"] for v in cell.values())
        delta = total_buy - total_sell
        cum_delta += delta
        candles_out.append({
            **base, "levels": levels_out, "poc_idx": poc_idx,
            "va_low_idx": va_low_idx, "va_high_idx": va_high_idx,
            "total_buy": round(total_buy, 4), "total_sell": round(total_sell, 4),
            "delta": round(delta, 4), "cum_delta": round(cum_delta, 4),
        })

    profile_out = sorted(
        [{"idx": idx, "price": round(idx * tick, 8), "buy": round(v["buy"], 4), "sell": round(v["sell"], 4)}
         for idx, v in profile_totals.items()],
        key=lambda r: r["idx"],
    )
    session_poc_idx = (max(profile_totals.items(), key=lambda kv: kv[1]["buy"] + kv[1]["sell"])[0]
                        if profile_totals else None)
    highs = [c["high"] for c in candles_out]
    lows = [c["low"] for c in candles_out]
    session = {
        "high": max(highs) if highs else None, "low": min(lows) if lows else None,
        "last": candles_out[-1]["close"] if candles_out else None,
        "total_buy": round(sum(c["total_buy"] for c in candles_out), 3),
        "total_sell": round(sum(c["total_sell"] for c in candles_out), 3),
        "total_delta": round(sum(c["delta"] for c in candles_out), 3),
        "poc_idx": session_poc_idx,
    }
    return {"symbol": symbol, "interval": interval, "tick": tick, "candles": candles_out,
            "profile": profile_out, "session": session}


_FOOTPRINT_JS = r"""
<div id="__WRAP_ID__" style="overflow-x:auto; overflow-y:hidden; border-radius:10px; border:1px solid rgba(255,255,255,0.08);">
  <canvas id="__CANVAS_ID__"></canvas>
</div>
<div id="__TOOLTIP_ID__" style="position:fixed; display:none; pointer-events:none; z-index:9999;
  background:#2c2c2e; color:#f5f5f7; border:1px solid rgba(255,255,255,0.12); border-radius:8px;
  padding:8px 10px; font:12px -apple-system,'SF Pro Display',sans-serif; box-shadow:0 8px 24px rgba(0,0,0,0.4);
  min-width:150px;"></div>
<script>
(function() {
  const payload = __PAYLOAD_JSON__;
  const candles = payload.candles;
  const profile = payload.profile;
  const session = payload.session;
  const tick = payload.tick;

  const C = {
    bg: '#1c1c1e', sep: 'rgba(255,255,255,0.09)', sepSoft: 'rgba(255,255,255,0.05)',
    text: '#f5f5f7', dim: '#98989d', buy: '#0A84FF', sell: '#FF453A',
    green: '#30D158', amber: '#FF9F0A', neutral: 'rgba(255,255,255,0.045)',
    va: 'rgba(255,255,255,0.045)',
  };
  const FONT_UI = '-apple-system, "SF Pro Display", "Segoe UI", sans-serif';
  const FONT_MONO = '"SF Mono", Menlo, Monaco, Consolas, monospace';

  function hexToRgb(hex) {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
  }
  const BUY_RGB = hexToRgb(C.buy), SELL_RGB = hexToRgb(C.sell);
  function rgba(rgb, a) { return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + a + ')'; }
  function fmt(n, d) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    return n.toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
  }
  function priceFmt(n) {
    const d = tick < 0.01 ? 4 : (tick < 0.1 ? 3 : (tick < 1 ? 2 : (tick < 10 ? 1 : 0)));
    return fmt(n, d);
  }
  // Volume/delta numbers, not price: a low-unit-price coin (DOT, DOGE...)
  // trades in raw-quantity amounts that run into the thousands even for
  // one price level, so these get k/M-compressed instead of full digits
  // -- shorter text means a narrower column, which is the whole point:
  // more candles fit on screen before you need to scroll.
  function volFmt(n) {
    if (n === null || n === undefined || !isFinite(n)) return '—';
    const sign = n < 0 ? '-' : '';
    const a = Math.abs(n);
    if (a >= 1000000) return sign + (a / 1000000).toFixed(a >= 10000000 ? 0 : 1) + 'M';
    if (a >= 1000) return sign + (a / 1000).toFixed(a >= 10000 ? 0 : 1) + 'k';
    if (a >= 1) return sign + a.toFixed(1);
    return sign + a.toFixed(a >= 0.1 ? 2 : 3);
  }

  if (!candles.length) return;

  // ---- global price axis (tick-index rows spanning the whole session) ----
  let hiIdx = -Infinity, loIdx = Infinity;
  candles.forEach(c => {
    if (c.high == null) return;
    hiIdx = Math.max(hiIdx, Math.round(c.high / tick));
    loIdx = Math.min(loIdx, Math.round(c.low / tick));
  });
  if (!isFinite(hiIdx)) { hiIdx = 1; loIdx = 0; }
  const rowIdxs = [];
  for (let i = hiIdx; i >= loIdx; i--) rowIdxs.push(i);
  const rowPos = new Map(rowIdxs.map((idx, i) => [idx, i]));

  // ---- layout ----
  const rowH = 20, candleLaneW = 11, numsW = 78, colW = candleLaneW + numsW;
  const marginLeft = 74, profileW = 130, headerH = 66, deltaH = 108, footerH = 30, pad = 14;
  const gridW = candles.length * colW;
  const gridH = rowIdxs.length * rowH;
  const totalW = marginLeft + gridW + profileW + pad * 2;
  const totalH = headerH + gridH + deltaH + footerH + pad;

  const wrap = document.getElementById('__WRAP_ID__');
  const canvas = document.getElementById('__CANVAS_ID__');
  const tooltip = document.getElementById('__TOOLTIP_ID__');
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = totalW + 'px';
  canvas.style.height = totalH + 'px';
  canvas.width = Math.round(totalW * dpr);
  canvas.height = Math.round(totalH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.textBaseline = 'alphabetic';

  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, totalW, totalH);

  // ---- header: symbol + session stats ----
  ctx.fillStyle = C.text;
  ctx.font = '600 17px ' + FONT_UI;
  ctx.fillText(payload.symbol, pad, 27);
  ctx.font = '12px ' + FONT_UI;
  ctx.fillStyle = C.dim;
  ctx.fillText(payload.interval + ' candles · footprint', pad, 45);

  function stat(x, label, value, color) {
    ctx.font = '10px ' + FONT_UI;
    ctx.fillStyle = C.dim;
    ctx.textAlign = 'right';
    ctx.fillText(label, x, 22);
    ctx.font = '600 14px ' + FONT_MONO;
    ctx.fillStyle = color || C.text;
    ctx.fillText(value, x, 40);
    ctx.textAlign = 'left';
  }
  const deltaStr = (session.total_delta >= 0 ? '+' : '') + volFmt(session.total_delta);
  const statVals = [
    ['LAST', priceFmt(session.last), C.text],
    ['HIGH', priceFmt(session.high), C.buy],
    ['LOW', priceFmt(session.low), C.sell],
    ['VOLUME', volFmt(session.total_buy + session.total_sell), C.text],
    ['DELTA', deltaStr, session.total_delta >= 0 ? C.buy : C.sell],
  ];
  const gap = 108;
  let xr = totalW - pad;
  for (let i = statVals.length - 1; i >= 0; i--) {
    stat(xr, statVals[i][0], statVals[i][1], statVals[i][2]);
    xr -= gap;
  }

  ctx.strokeStyle = C.sep;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, headerH - 0.5); ctx.lineTo(totalW, headerH - 0.5); ctx.stroke();

  // ---- price axis (left margin) ----
  ctx.font = '11px ' + FONT_MONO;
  ctx.fillStyle = C.dim;
  ctx.textAlign = 'right';
  rowIdxs.forEach((idx, i) => {
    if (i % 1 === 0) {
      const y = headerH + i * rowH + rowH / 2 + 4;
      ctx.fillText(priceFmt(idx * tick), marginLeft - 10, y);
    }
  });
  ctx.textAlign = 'left';

  // ---- candle grid ----
  candles.forEach((c, ci) => {
    const x0 = marginLeft + ci * colW;
    const laneX = x0, numX = x0 + candleLaneW;

    // value area tint (drawn first, under everything)
    if (c.va_low_idx != null) {
      const yTop = headerH + rowPos.get(c.va_high_idx) * rowH;
      const yBot = headerH + (rowPos.get(c.va_low_idx) + 1) * rowH;
      ctx.fillStyle = C.va;
      ctx.fillRect(numX, yTop, numsW, yBot - yTop);
    }

    // per-candle max cell volume, for cell alpha scaling
    let maxCell = 0;
    c.levels.forEach(l => { maxCell = Math.max(maxCell, l.buy, l.sell); });
    if (maxCell <= 0) maxCell = 1;

    // neutral fill for in-range-but-untraded rows
    const hiI = Math.round(c.high / tick), loI = Math.round(c.low / tick);
    for (let idx = loI; idx <= hiI; idx++) {
      const pos = rowPos.get(idx);
      if (pos === undefined) continue;
      ctx.fillStyle = C.neutral;
      ctx.fillRect(numX, headerH + pos * rowH, numsW, rowH - 1);
    }

    c.levels.forEach(l => {
      const pos = rowPos.get(l.idx);
      if (pos === undefined) return;
      const y = headerH + pos * rowH;
      const total = l.buy + l.sell;
      const dominant = l.buy >= l.sell;
      const alpha = 0.16 + 0.62 * Math.min(total / maxCell, 1);
      ctx.fillStyle = rgba(dominant ? BUY_RGB : SELL_RGB, alpha);
      ctx.fillRect(numX, y, numsW, rowH - 1);

      if (l.idx === c.poc_idx) {
        ctx.strokeStyle = C.amber;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(numX + 0.75, y + 0.75, numsW - 1.5, rowH - 2.5);
      }

      ctx.font = '10px ' + FONT_MONO;
      ctx.fillStyle = '#f5f5f7';
      ctx.textAlign = 'right';
      ctx.fillText(volFmt(l.buy), numX + numsW * 0.46, y + rowH - 6);
      ctx.fillStyle = C.dim;
      ctx.font = '9px ' + FONT_MONO;
      ctx.fillText('x', numX + numsW * 0.54, y + rowH - 6);
      ctx.textAlign = 'left';
      ctx.font = '10px ' + FONT_MONO;
      ctx.fillStyle = '#f5f5f7';
      ctx.fillText(volFmt(l.sell), numX + numsW * 0.58, y + rowH - 6);

      if (l.buy_imb) {
        ctx.fillStyle = C.buy;
        ctx.beginPath();
        ctx.moveTo(numX + numsW - 1, y + 3); ctx.lineTo(numX + numsW - 7, y + rowH/2); ctx.lineTo(numX + numsW - 1, y + rowH - 4);
        ctx.fill();
      }
      if (l.sell_imb) {
        ctx.fillStyle = C.sell;
        ctx.beginPath();
        ctx.moveTo(numX + 1, y + 3); ctx.lineTo(numX + 7, y + rowH/2); ctx.lineTo(numX + 1, y + rowH - 4);
        ctx.fill();
      }
    });

    // candle silhouette lane
    function yForPrice(p) {
      const idx = p / tick;
      // linear interpolation against the integer row grid
      const topIdx = rowIdxs[0];
      return headerH + (topIdx - idx) * rowH + rowH / 2;
    }
    const yHigh = yForPrice(c.high), yLow = yForPrice(c.low);
    const yOpen = yForPrice(c.open), yClose = yForPrice(c.close);
    const up = c.close >= c.open;
    ctx.strokeStyle = up ? C.buy : C.sell;
    ctx.fillStyle = up ? C.buy : C.sell;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.5;
    const cx = laneX + candleLaneW / 2;
    ctx.beginPath(); ctx.moveTo(cx, yHigh); ctx.lineTo(cx, yLow); ctx.stroke();
    const bodyTop = Math.min(yOpen, yClose), bodyBot = Math.max(yOpen, yClose);
    ctx.fillRect(laneX + 2, bodyTop, candleLaneW - 4, Math.max(bodyBot - bodyTop, 1.5));
    ctx.globalAlpha = 1;

    // column separator + time label
    ctx.strokeStyle = C.sepSoft;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0 + colW - 0.5, headerH); ctx.lineTo(x0 + colW - 0.5, headerH + gridH); ctx.stroke();
    ctx.font = '10px ' + FONT_MONO;
    ctx.fillStyle = C.dim;
    ctx.textAlign = 'center';
    ctx.fillText(c.time, x0 + colW / 2, headerH + gridH + 16);
    ctx.textAlign = 'left';
  });

  // outer grid border
  ctx.strokeStyle = C.sep;
  ctx.strokeRect(marginLeft + 0.5, headerH + 0.5, gridW - 1, gridH - 1);

  // ---- right margin: session volume profile ----
  const profX0 = marginLeft + gridW + 10;
  const profMaxW = profileW - 20;
  let maxProfTotal = 0;
  profile.forEach(r => { maxProfTotal = Math.max(maxProfTotal, r.buy + r.sell); });
  if (maxProfTotal <= 0) maxProfTotal = 1;
  const profByIdx = new Map(profile.map(r => [r.idx, r]));
  rowIdxs.forEach((idx, i) => {
    const r = profByIdx.get(idx);
    if (!r) return;
    const y = headerH + i * rowH;
    const buyW = (r.buy / maxProfTotal) * profMaxW;
    const sellW = (r.sell / maxProfTotal) * profMaxW;
    ctx.fillStyle = rgba(BUY_RGB, 0.55);
    ctx.fillRect(profX0, y + 2, buyW, rowH - 5);
    ctx.fillStyle = rgba(SELL_RGB, 0.55);
    ctx.fillRect(profX0 + buyW, y + 2, sellW, rowH - 5);
    if (idx === session.poc_idx) {
      ctx.fillStyle = C.amber;
      ctx.fillRect(profX0 - 4, y + 1, 3, rowH - 3);
    }
  });
  ctx.font = '10px ' + FONT_UI;
  ctx.fillStyle = C.dim;
  ctx.fillText('SESSION VOLUME PROFILE', profX0, headerH - 8);

  // ---- cumulative delta subplot ----
  const dY0 = headerH + gridH + 24;
  const dPlotH = deltaH - 30;
  let maxAbsCum = 0;
  candles.forEach(c => { maxAbsCum = Math.max(maxAbsCum, Math.abs(c.cum_delta)); });
  if (maxAbsCum <= 0) maxAbsCum = 1;
  const zeroY = dY0 + dPlotH / 2;

  ctx.strokeStyle = C.sepSoft;
  ctx.beginPath(); ctx.moveTo(marginLeft, dY0); ctx.lineTo(marginLeft, dY0 + dPlotH); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(marginLeft, zeroY); ctx.lineTo(marginLeft + gridW, zeroY); ctx.stroke();
  ctx.font = '10px ' + FONT_UI;
  ctx.fillStyle = C.dim;
  ctx.textAlign = 'right';
  ctx.fillText('CUM. DELTA', marginLeft - 10, dY0 + 10);
  ctx.textAlign = 'left';

  ctx.beginPath();
  candles.forEach((c, ci) => {
    const x = marginLeft + ci * colW + colW / 2;
    const y = zeroY - (c.cum_delta / maxAbsCum) * (dPlotH / 2 - 4);
    if (ci === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = session.total_delta >= 0 ? C.buy : C.sell;
  ctx.lineWidth = 2;
  ctx.stroke();
  candles.forEach((c, ci) => {
    const x = marginLeft + ci * colW + colW / 2;
    const y = zeroY - (c.cum_delta / maxAbsCum) * (dPlotH / 2 - 4);
    ctx.fillStyle = c.delta >= 0 ? C.buy : C.sell;
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, 7); ctx.fill();
  });

  // ---- footer legend ----
  const fY = headerH + gridH + deltaH + 18;
  ctx.font = '10.5px ' + FONT_UI;
  function legendDot(x, color, label) {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x, fY - 3, 4, 0, 7); ctx.fill();
    ctx.fillStyle = C.dim;
    ctx.fillText(label, x + 10, fY);
    return x + 10 + ctx.measureText(label).width + 22;
  }
  let lx = pad;
  lx = legendDot(lx, C.buy, 'buyers dominant');
  lx = legendDot(lx, C.sell, 'sellers dominant');
  lx = legendDot(lx, C.amber, 'point of control');
  ctx.fillStyle = C.dim;
  ctx.fillText('▶ ◀ stacked imbalance (' + payload.imbalance_ratio + ':1, 3+ levels) · shaded band = value area (70% of volume)', lx, fY);

  // ---- hover tooltip ----
  // Built via DOM APIs (createElement/textContent), never HTML-string
  // concatenation -- an open-angle-bracket tag pattern inside this
  // script's own JS text is enough to make Streamlit's sanitizer drop the
  // whole script block silently, confirmed directly.
  while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
  function ttLine(text, color, extraStyle) {
    const div = document.createElement('div');
    div.textContent = text;
    if (color) div.style.color = color;
    if (extraStyle) Object.assign(div.style, extraStyle);
    tooltip.appendChild(div);
    return div;
  }
  const ttTitle = ttLine('', null, {fontWeight: '600', marginBottom: '4px'});
  const ttBuy = ttLine('', C.buy);
  const ttSell = ttLine('', C.sell);
  const ttDelta = ttLine('', C.dim, {marginTop: '4px'});
  const ttPoc = ttLine('point of control', C.amber, {marginTop: '2px'});
  ttPoc.style.display = 'none';

  const levelByPos = new Map();
  candles.forEach((c, ci) => {
    c.levels.forEach(l => {
      const pos = rowPos.get(l.idx);
      if (pos !== undefined) levelByPos.set(ci + ':' + pos, {c, l});
    });
  });
  canvas.addEventListener('mousemove', (ev) => {
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
    if (mx < marginLeft || mx > marginLeft + gridW || my < headerH || my > headerH + gridH) {
      tooltip.style.display = 'none';
      return;
    }
    const ci = Math.floor((mx - marginLeft) / colW);
    const pos = Math.floor((my - headerH) / rowH);
    const hit = levelByPos.get(ci + ':' + pos);
    if (!hit) { tooltip.style.display = 'none'; return; }
    const { c, l } = hit;
    const total = l.buy + l.sell;
    const pctOfCandle = c.total_buy + c.total_sell > 0 ? (total / (c.total_buy + c.total_sell) * 100) : 0;
    const delta = l.buy - l.sell;
    ttTitle.textContent = c.time + '   ' + priceFmt(l.price);
    ttBuy.textContent = 'buy    ' + l.buy.toFixed(2);
    ttSell.textContent = 'sell   ' + l.sell.toFixed(2);
    ttDelta.textContent = 'delta ' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + '  ·  ' + pctOfCandle.toFixed(0) + '% of candle';
    ttPoc.style.display = (l.idx === c.poc_idx) ? 'block' : 'none';
    tooltip.style.display = 'block';
    tooltip.style.left = (ev.clientX + 14) + 'px';
    tooltip.style.top = (ev.clientY + 14) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });
})();
</script>
"""


def _footprint_html(payload, imbalance_ratio):
    uid = uuid.uuid4().hex[:8]
    payload = {**payload, "imbalance_ratio": imbalance_ratio}
    html = (_FOOTPRINT_JS
            .replace("__WRAP_ID__", f"fp-wrap-{uid}")
            .replace("__CANVAS_ID__", f"fp-canvas-{uid}")
            .replace("__TOOLTIP_ID__", f"fp-tooltip-{uid}")
            .replace("__PAYLOAD_JSON__", json.dumps(payload)))
    return html


def render_footprint_tab(ticker_info, sync_ticker=None, sync_interval=None):
    """Crypto only, always — a Markets-scoped caller (no crypto in its own
    ticker_info) gets an honest 'not available' rather than being called
    at all; app.py's Strategy tab skips calling this entirely instead, since
    the reason (Yahoo-only tickers can't do this) isn't specific to what
    happens to be in ticker_info at the moment.

    sync_ticker/sync_interval: when given, the Ticker/Candle size pickers
    below default to — and re-follow, the moment either one actually
    changes — whatever the caller's own main chart is currently showing,
    instead of always starting at BTC-USD/1m. Picking something ELSE here
    by hand still works and stays picked (this only overwrites the
    picker's own session_state on a genuine CHANGE upstream, same
    "owner tracking" pattern the shared TF radio uses elsewhere — not on
    every render, which would fight a manual override right back to
    whatever the main chart happens to be on). sync_interval is only
    honored when it's one of THIS tab's own supported candle sizes
    (1m-4h) — a main chart sitting on 1D+ leaves the candle-size picker
    at whatever it was last explicitly set to."""
    crypto_tickers = sorted(t for t, (_, cat) in ticker_info.items() if cat == "Crypto")
    st.subheader("Footprint chart")
    if not crypto_tickers:
        st.info("No crypto tickers available.")
    else:
        if sync_ticker in crypto_tickers:
            st.session_state.setdefault("_fp_synced_ticker", sync_ticker)
            if st.session_state["_fp_synced_ticker"] != sync_ticker:
                st.session_state["bt_fp_ticker"] = sync_ticker
                st.session_state["_fp_synced_ticker"] = sync_ticker
            else:
                st.session_state.setdefault("bt_fp_ticker", sync_ticker)
        if sync_interval in ("1m", "5m", "15m", "30m", "1h", "4h"):
            st.session_state.setdefault("_fp_synced_interval", sync_interval)
            if st.session_state["_fp_synced_interval"] != sync_interval:
                st.session_state["bt_fp_interval"] = sync_interval
                st.session_state["_fp_synced_interval"] = sync_interval
            else:
                st.session_state.setdefault("bt_fp_interval", sync_interval)

        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            _default_fp = "BTC-USD" if "BTC-USD" in crypto_tickers else crypto_tickers[0]
            fp_ticker = st.selectbox("Ticker", crypto_tickers, index=crypto_tickers.index(_default_fp),
                                      key="bt_fp_ticker", format_func=lambda t: f"{t} — {ticker_info[t][0]}")
        with f2:
            fp_interval = st.selectbox("Candle size", ["1m", "5m", "15m", "30m", "1h", "4h"],
                                        key="bt_fp_interval",
                                        help="Same intraday tiers as the main chart's timeframe picker. Stops at "
                                             "4h on purpose — a footprint chart reads individual trades inside "
                                             "each candle, and a daily-or-slower candle on a busy pair can hold "
                                             "hundreds of thousands of them, past what's practical to fetch.")
        with f3:
            fp_n = st.slider("Candles", 5, 150, 50, key="bt_fp_n",
                              help="Binance allows up to 1000 candles per call, but a busy pair (BTC, ETH) trades "
                                   "~300 times a minute — fetching their real trade history gets slow past a couple "
                                   "hundred candles. A quiet pair (DOT, and most smaller alts) trades far less "
                                   "often, so the same candle count loads almost instantly for those. Defaulted "
                                   "higher than the safe minimum on purpose — push it further if this ticker/"
                                   "candle-size combo loads quickly; the safety cap below kicks in and warns you "
                                   "instead of hanging if you push it too far for a busy pair.")
        with f4:
            fp_tick = st.number_input("Price bucket size ($)", 0.001, 5000.0, 5.0, step=0.01, key="bt_fp_tick",
                                       help="How finely to group trades by price. Smaller = more rows, more detail. "
                                            "A coin under $10 (like DOT) usually wants 0.01 or smaller; BTC wants 5-50.")
        with f5:
            fp_ratio = st.number_input("Imbalance ratio", 1.5, 10.0, 3.0, step=0.5, key="bt_fp_ratio",
                                        help="How much bigger one side needs to be than the diagonal level on the "
                                             "other side to count as an imbalance. 3.0 is the common default.")

        load = st.button("Load footprint", key="bt_fp_load")
        if load:
            st.session_state["bt_fp_loaded"] = True

        if st.session_state.get("bt_fp_loaded"):
            symbol = _binance_symbol(fp_ticker)
            try:
                with st.spinner(f"Fetching real trades for {symbol} from Binance..."):
                    klines = _fetch_klines(symbol, fp_interval, fp_n)
                    if not klines:
                        st.warning(f"No data back from Binance for {symbol}.")
                        trades, payload = [], None
                    else:
                        # Scales with candle SIZE too, not just count -- a
                        # 4h candle holds ~240x the trades a 1m one does
                        # (same underlying trade rate, just a wider window),
                        # so the same candle count needs proportionally more
                        # pagination headroom the bigger the interval is.
                        interval_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
                        cost = fp_n * interval_minutes.get(fp_interval, 1)
                        max_calls = min(150, max(40, cost // 2))
                        trades, truncated = _fetch_agg_trades(
                            symbol, klines[0]["open_time"], klines[-1]["close_time"], max_calls)
                        payload = _build_footprint_payload(symbol, fp_interval, klines, trades, fp_tick, fp_ratio)
                if payload and any(c["levels"] for c in payload["candles"]):
                    if truncated:
                        st.warning(f"{symbol} trades fast enough that fetching all of it for {fp_n} × {fp_interval} "
                                   f"candles hit the safety limit ({max_calls} API calls) — the earliest candles "
                                   f"shown may be missing trades. Try fewer candles or a smaller candle size for a "
                                   f"complete picture.")
                    st.html(_footprint_html(payload, fp_ratio), unsafe_allow_javascript=True)
                    st.caption(f"{symbol} · {len(trades):,} real trades from Binance's public feed, bucketed into "
                               f"${fp_tick:g} price levels. Hover any cell for detail.")
                elif klines:
                    st.info("No trades came back for this window — try a larger candle size or fewer candles.")
            except requests.RequestException as e:
                st.error(f"Couldn't reach Binance: {e}")
        else:
            st.caption("Pick a ticker and click Load footprint — fetches real trades from Binance, takes a few seconds.")
