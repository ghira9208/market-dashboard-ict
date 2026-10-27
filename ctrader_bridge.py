"""Persistent local bridge to FxPro/cTrader's Open API — the ONE process
in this whole platform allowed to import ctrader_open_api, because that
package pins protobuf==3.20.1, which conflicts with Streamlit's own
protobuf>=7 requirement in the main edge-pipeline environment (same
conflict noted in ctrader_auth.py). Runs under the isolated .venv_ctrader
interpreter, holds the single long-lived TCP connection to cTrader, and
exposes a plain local HTTP/JSON API — the same shape brokers/etoro.py
already talks to eToro's own real REST API in, just proxied here instead
of hitting a remote host directly. brokers/ctrader.py (in the main
edge-pipeline environment) is the only intended caller.

Run: .venv_ctrader/bin/python3 ctrader_bridge.py

DEMO ACCOUNT ONLY — the OAuth token this reads (.cache/ctrader_tokens.json,
"trading" scope) can see every trading account under this cTrader ID,
live and demo alike. On startup this asks cTrader which accounts exist
and picks the demo one explicitly (isLive == False) — if none is found,
this refuses to start rather than guess. It also hardcodes the DEMO
protobuf host (EndPoints.PROTOBUF_DEMO_HOST), never the live one. Same
non-negotiable guarantee brokers/etoro.py already enforces for eToro.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import crochet
crochet.setup()

import ctrader_tokens
import tick_recorder

from ctrader_open_api import Client, EndPoints, Protobuf, TcpProtocol, Auth
from ctrader_open_api.messages import OpenApiMessages_pb2 as messages
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model
from twisted.internet.defer import Deferred

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_TOKENS_PATH = os.path.join(_PROJECT_DIR, ".cache", "ctrader_tokens.json")
_SECRETS_PATH = os.path.join(_PROJECT_DIR, ".streamlit", "secrets.toml")
_BRIDGE_PORT = int(os.environ.get("CTRADER_BRIDGE_PORT", 8700))   # a second bridge instance serves a second account


def _load_secrets():
    import tomllib
    with open(_SECRETS_PATH, "rb") as f:
        return tomllib.load(f)


def _load_tokens():
    with open(_TOKENS_PATH) as f:
        return json.load(f)


def _save_tokens(tokens):
    ctrader_tokens.save_atomic(_TOKENS_PATH, tokens)


class BridgeError(RuntimeError):
    pass


class CTraderBridge:
    """One persistent, authenticated connection to cTrader's demo
    environment. Every public method here blocks (via crochet) until the
    real response arrives or times out — callers never touch a Deferred
    or the Twisted reactor directly."""

    def __init__(self):
        secrets = _load_secrets()
        self._secrets = secrets
        self._client_id = secrets["CTRADER_CLIENT_ID"]
        self._client_secret = secrets["CTRADER_CLIENT_SECRET"]
        self._auth = Auth(self._client_id, self._client_secret, "http://localhost:8600/callback")
        self._tokens = _load_tokens()
        self._client = Client(EndPoints.PROTOBUF_DEMO_HOST, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self._client.setConnectedCallback(lambda c: None)
        self._account_id = None
        self._symbols_by_name = {}   # symbolName -> symbolId
        self._symbols_full = {}      # symbolId -> ProtoOASymbol (cached after first fetch)
        self._lock = threading.Lock()
        self._recorder = None        # tick_recorder.TickRecorder when recording (main bridge only)

    # -------------------------------------------------------- lifecycle --
    def start(self):
        self._client.startService()
        self._wait_connected(timeout=15)
        self._refresh_token_if_needed()
        self._send(messages.ProtoOAApplicationAuthReq(
            clientId=self._client_id, clientSecret=self._client_secret))
        accounts_res = self._send(messages.ProtoOAGetAccountListByAccessTokenReq(
            accessToken=self._tokens["accessToken"]))
        extracted_accounts = Protobuf.extract(accounts_res)
        if type(extracted_accounts).__name__ == "ProtoOAErrorRes":
            newer = ctrader_tokens.reread_if_changed(_TOKENS_PATH, self._tokens)
            if newer is not None:           # another bridge refreshed since we loaded: use its token, once
                self._tokens = newer
                accounts_res = self._send(messages.ProtoOAGetAccountListByAccessTokenReq(
                    accessToken=self._tokens["accessToken"]))
                extracted_accounts = Protobuf.extract(accounts_res)
        # Found live 2026-10-03: this crashed with a bare AttributeError
        # ('ProtoOAErrorRes' object has no attribute 'ctidTraderAccount'),
        # crash-looping launchd every restart with zero indication of the
        # REAL cause — the exact same class of bug symbol_detail() already
        # guards against below. Surfacing cTrader's own real error code/
        # description instead (almost certainly an expired/revoked
        # accessToken if _refresh_token_if_needed's own silent refresh
        # failure above didn't actually get a fresh one).
        if type(extracted_accounts).__name__ == "ProtoOAErrorRes":
            raise BridgeError(
                f"Account list request rejected: {extracted_accounts.errorCode} — "
                f"{extracted_accounts.description}. The saved OAuth token is likely expired or "
                f"revoked and needs re-authorization, not a transient network issue.")
        accounts = extracted_accounts.ctidTraderAccount
        demo_accounts = [a for a in accounts if not a.isLive]
        if not demo_accounts:
            raise BridgeError(
                "No DEMO account found under this cTrader ID's 'trading'-scoped token — "
                "refusing to start rather than guess. Live accounts present: "
                f"{[a.ctidTraderAccountId for a in accounts if a.isLive]}")
        # This cTrader ID can have MULTIPLE demo accounts (confirmed
        # directly: old prop-firm demos plus a fresh one) — "first demo
        # found" is a silent coin flip once that's true. Prefer an
        # explicitly configured account id; only fall back to "first
        # demo" when nothing is configured, and say so loudly rather than
        # quietly picking one.
        configured_id = os.environ.get("CTRADER_DEMO_ACCOUNT_ID") or self._secrets.get("CTRADER_DEMO_ACCOUNT_ID")
        configured_id = int(configured_id) if configured_id is not None else None
        demo_ids = [a.ctidTraderAccountId for a in demo_accounts]
        if configured_id is not None:
            if configured_id not in demo_ids:
                raise BridgeError(
                    f"CTRADER_DEMO_ACCOUNT_ID={configured_id} isn't among this token's demo "
                    f"accounts ({demo_ids}) — refusing to silently fall back to a different one.")
            self._account_id = configured_id
        else:
            print(f"No CTRADER_DEMO_ACCOUNT_ID configured — defaulting to the first demo "
                  f"account found ({demo_ids[0]}) out of {len(demo_ids)}. Set it explicitly "
                  f"in secrets.toml if that's not the intended one.")
            self._account_id = demo_ids[0]
        self._send(messages.ProtoOAAccountAuthReq(
            ctidTraderAccountId=self._account_id, accessToken=self._tokens["accessToken"]))
        self._preload_symbols()
        threading.Thread(target=self._health_check_loop, daemon=True).start()
        self._start_tick_recorder()

    def _health_check_loop(self, interval=30):
        """Confirmed directly: the raw TCP connection can report
        isConnected=True (ClientService's own reconnect handled that
        layer) while the ACCOUNT AUTH session underneath is dead —
        /health looked fine, /trader came back a ProtoOAErrorRes. This
        periodically proves the session still actually works with a
        real lightweight request, tries one in-place re-auth if it
        doesn't, and — if that also fails — exits the process outright
        so launchd's KeepAlive restarts it clean, rather than trying to
        hand-roll every possible partial-failure state. A full restart
        is already confirmed to fix this (that's exactly how this got
        caught and repaired the first time)."""
        while True:
            time.sleep(interval)
            try:
                res = self._send(messages.ProtoOATraderReq(ctidTraderAccountId=self._account_id))
                if type(Protobuf.extract(res)).__name__ == "ProtoOAErrorRes":
                    raise RuntimeError("session rejected by cTrader")
            except Exception as e:
                print(f"[health-check] session check failed ({e}) — attempting re-auth")
                try:
                    self._send(messages.ProtoOAApplicationAuthReq(
                        clientId=self._client_id, clientSecret=self._client_secret))
                    self._send(messages.ProtoOAAccountAuthReq(
                        ctidTraderAccountId=self._account_id, accessToken=self._tokens["accessToken"]))
                    print("[health-check] re-auth succeeded, session restored")
                    self._subscribe_ticks()
                except Exception as e2:
                    print(f"[health-check] re-auth failed too ({e2}) — exiting for launchd to restart")
                    os._exit(1)

    def _wait_connected(self, timeout):
        deadline = time.time() + timeout
        while not self._client.isConnected:
            if time.time() > deadline:
                raise BridgeError("Could not connect to cTrader's demo endpoint within timeout.")
            time.sleep(0.1)

    def _refresh_token_if_needed(self):
        # Refresh only when the token is actually old, under a cross-process lock, adopting a refresh another bridge
        # already did — see ctrader_tokens.py for why refreshing on every start broke the login repeatedly.
        self._tokens = ctrader_tokens.ensure_fresh(_TOKENS_PATH, lambda rt: self._auth.refreshToken(rt))

    @crochet.wait_for(timeout=20)
    def _send_async(self, message):
        # Client.send's OWN default responseTimeoutInSeconds is 5 —
        # confirmed directly (a 1214-row ProtoOASymbolsListRes timed out
        # under it once already) that's tighter than crochet's outer
        # wait_for above, making that 15s/20s number meaningless since
        # the inner timeout always fires first. Passed explicitly here so
        # the two actually agree, with headroom under the outer wait.
        return self._client.send(message, responseTimeoutInSeconds=18)

    def _send(self, message):
        # crochet.wait_for already blocks this (calling) thread until the
        # underlying Deferred fires and returns the resolved value
        # directly — no separate .wait() call, that was assuming the
        # plain Twisted Deferred API instead of crochet's own contract.
        with self._lock:
            result = self._send_async(message)
        return result

    def _preload_symbols(self):
        res = self._send(messages.ProtoOASymbolsListReq(ctidTraderAccountId=self._account_id))
        extracted = Protobuf.extract(res)
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Symbol list request rejected: {extracted.errorCode} — {extracted.description}")
        for s in extracted.symbol:
            self._symbols_by_name[s.symbolName] = s.symbolId

    # -------------------------------------------------------------- api --
    def symbols(self):
        return [{"symbolId": sid, "symbolName": name} for name, sid in self._symbols_by_name.items()]

    # ------------------------------------------------------------ ticks --
    def _start_tick_recorder(self):
        """Record live bid/ask ticks (see tick_recorder.py). Only on the main bridge (port 8700) and only when
        config/tick_symbols.json lists the symbols to record — the second account's bridge would just duplicate
        the same quotes. Never allowed to stop the bridge from starting."""
        try:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "tick_symbols.json")
            if _BRIDGE_PORT != 8700 or not os.path.exists(path):
                return
            with open(path) as f:
                names = json.load(f)
            ids = {self._symbols_by_name[n]: n for n in names if n in self._symbols_by_name}
            if not ids:
                return
            self._recorder = tick_recorder.TickRecorder()
            self._recorder.set_symbols(ids)
            self._client.setMessageReceivedCallback(self._recorder.on_message)
            self._recorder.start()
            self._subscribe_ticks()
            print(f"Recording live ticks for {len(ids)} symbols to {self._recorder.root}")
        except Exception as e:
            print(f"tick recorder not started ({e}) — bridge continues without it")

    def _subscribe_ticks(self):
        if self._recorder is None:
            return
        try:
            res = self._send(messages.ProtoOASubscribeSpotsReq(
                ctidTraderAccountId=self._account_id, symbolId=list(self._recorder._names),
                subscribeToSpotTimestamp=True))
            extracted = Protobuf.extract(res)
            if type(extracted).__name__ == "ProtoOAErrorRes":
                print(f"tick subscription rejected: {extracted.errorCode} — {extracted.description}")
        except Exception as e:
            print(f"tick subscription failed ({e})")

    def tick_status(self):
        return self._recorder.status() if self._recorder is not None else {"recording": False}

    def ticks(self, symbol_id, quote_type, from_ms, to_ms, raw=False):
        """One page of historical ticks. cTrader delta-encodes them (first value absolute, the rest differences);
        raw=True returns them untouched so the encoding can be checked against real data."""
        res = self._send(messages.ProtoOAGetTickDataReq(
            ctidTraderAccountId=self._account_id, symbolId=symbol_id,
            type=model.ProtoOAQuoteType.Value(quote_type), fromTimestamp=from_ms, toTimestamp=to_ms))
        extracted = Protobuf.extract(res)
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Tick data request rejected: {extracted.errorCode} — {extracted.description}")
        pairs = [(t.timestamp, t.tick) for t in extracted.tickData]
        if raw:
            return {"hasMore": extracted.hasMore, "n": len(pairs), "first": pairs[:5], "last": pairs[-5:]}
        ts = px = 0
        out = []
        for dt, dp in pairs:
            ts += dt
            px += dp
            out.append((ts, px / 100000.0))
        return {"hasMore": extracted.hasMore, "ticks": out}

    def symbol_detail(self, symbol_id):
        if symbol_id in self._symbols_full:
            return self._symbols_full[symbol_id]
        res = self._send(messages.ProtoOASymbolByIdReq(
            ctidTraderAccountId=self._account_id, symbolId=[symbol_id]))
        extracted = Protobuf.extract(res)
        # Confirmed directly: SYMBOL_NOT_FOUND is a real, expected error
        # here, not a fluke — this account's own symbol universe (830
        # symbols) differs from another real account's (1214), so a
        # symbolId valid elsewhere can be genuinely absent here. Surfacing
        # that plainly instead of letting a bare AttributeError (no
        # `.symbol` on an error response) look like a bridge bug.
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"symbol {symbol_id}: {extracted.errorCode} — {extracted.description}")
        sym = extracted.symbol[0]
        # Commission/swap: the real, exact fields the strategy's true
        # trading cost is computed from (see brokers/ctrader.py's fee
        # engine) — preciseTradingCommissionRate is the authoritative one
        # per cTrader's own proto docs ("commission" field says to use
        # this instead), scaled by 10^8 for non-percentage commission
        # types, by 10^5 for PERCENTAGE_OF_VALUE.
        #
        # preciseMinCommission/minCommissionType/minCommissionAsset: found
        # live 2026-10-02 — fee_engine.py's percentage-only formula missed
        # these entirely, which is exactly why it understated ^AEX's real
        # commission by ~17x (predicted ~$0.46 round trip, cTrader's own
        # app showed $8 real, confirmed directly by the account owner on
        # 3 separate positions). A PERCENTAGE_OF_VALUE symbol can still
        # carry a real per-side MINIMUM commission charged instead of the
        # percentage whenever the percentage would come out smaller —
        # exactly what a tiny broker-minimum-lot position like this one
        # hits every time. Scaling NOT yet confirmed (unlike
        # preciseTradingCommissionRate's doc-verified 10^5/10^8 split) —
        # surfaced raw here deliberately, to be derived empirically against
        # the real $8 figure rather than guessed. Fix this comment once
        # fee_engine.py's commission_pct_roundtrip() is updated to use it.
        detail = {
            "symbolId": sym.symbolId, "digits": sym.digits, "pipPosition": sym.pipPosition,
            "minVolume": sym.minVolume, "maxVolume": sym.maxVolume, "stepVolume": sym.stepVolume,
            "lotSize": sym.lotSize,
            "commissionType": model.ProtoOACommissionType.Name(sym.commissionType) if sym.commissionType else None,
            "preciseTradingCommissionRate": sym.preciseTradingCommissionRate,
            "swapLong": sym.swapLong, "swapShort": sym.swapShort,
            "minCommission": sym.minCommission,
            "preciseMinCommission": sym.preciseMinCommission,
            "minCommissionType": model.ProtoOAMinCommissionType.Name(sym.minCommissionType) if sym.minCommissionType else None,
            "minCommissionAsset": sym.minCommissionAsset,
        }
        self._symbols_full[symbol_id] = detail
        return detail

    def trendbars(self, symbol_id, period, count):
        # period: one of cTrader's own ProtoOATrendbarPeriod names, e.g.
        # "M1","M5","M15","M30","H1","H4","D1" — passed straight through,
        # the caller (brokers/ctrader.py) owns the timeframe->period mapping.
        # fromTimestamp/toTimestamp are REQUIRED fields on this request
        # (confirmed directly: omitting them raises a protobuf EncodeError
        # — `count` alone isn't a valid request), so this computes a
        # generous window backward from now covering `count` bars at this
        # period, with `count` still passed as the cap on rows returned.
        period_minutes = {
            "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5, "M10": 10, "M15": 15,
            "M30": 30, "H1": 60, "H4": 240, "H12": 720, "D1": 1440, "W1": 10080,
        }[period]
        period_val = model.ProtoOATrendbarPeriod.Value(period)
        now_ms = int(time.time() * 1000)
        # Confirmed directly: cTrader's own trendbar generation can lag
        # real time by 20+ minutes even on a liquid pair, so a tight
        # count-sized window can come back completely empty for a small
        # `count`. Padding by the larger of (2x the requested count) or
        # a flat 2-hour buffer covers both a small `count` on a fast
        # timeframe and normal weekend/session gaps on a slow one.
        padded_bars = max(count * 2, count + 5)
        from_ms = now_ms - max(period_minutes * 60 * 1000 * padded_bars, 2 * 60 * 60 * 1000)
        res = self._send(messages.ProtoOAGetTrendbarsReq(
            ctidTraderAccountId=self._account_id, period=period_val,
            count=count, symbolId=symbol_id, fromTimestamp=from_ms, toTimestamp=now_ms))
        extracted = Protobuf.extract(res)
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Trendbar request rejected: {extracted.errorCode} — {extracted.description}")
        bars = extracted.trendbar
        out = []
        for b in bars:
            low = b.low / 100000.0
            out.append({
                "utcTimestampInMinutes": b.utcTimestampInMinutes,
                "open": low + b.deltaOpen / 100000.0,
                "high": low + b.deltaHigh / 100000.0,
                "low": low,
                "close": low + b.deltaClose / 100000.0,
                "volume": b.volume,
            })
        return out

    def _asset_name(self, asset_id):
        """'USD', 'GBP', ... for an account's deposit asset (looked up once). None if it can't be resolved —
        callers then fall back to their own knowledge rather than assuming."""
        if getattr(self, "_assets", None) is None:
            try:
                res = Protobuf.extract(self._send(messages.ProtoOAAssetListReq(ctidTraderAccountId=self._account_id)))
                self._assets = {a.assetId: a.name for a in res.asset}
            except Exception:
                return None
        return self._assets.get(asset_id)

    def trader(self):
        res = self._send(messages.ProtoOATraderReq(ctidTraderAccountId=self._account_id))
        extracted = Protobuf.extract(res)
        # Found live 2026-10-06, 36 real occurrences in the bridge's own
        # error log: a bare AttributeError here (no `.trader` on an error
        # response) crashed every /trader request mid-handler, after the
        # socket was already committed to a 200 — the real error got
        # masked behind a BrokenPipeError instead of ever reaching the
        # caller as a clean message. Same class of bug symbol_detail()
        # already guards against; /trader just never got the same fix.
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Trader request rejected: {extracted.errorCode} — {extracted.description}")
        t = extracted.trader
        scale = 10 ** t.moneyDigits
        # leverageInCents/maxLeverage: real margin math needs these, not an
        # assumed number — found needing this live, checking whether raising
        # max_position_pct past 100% could ever margin-call this account by
        # stacking several larger positions at once.
        return {"balance": t.balance / scale, "moneyDigits": t.moneyDigits, "depositAssetId": t.depositAssetId,
                "depositAssetName": self._asset_name(t.depositAssetId),
                "leverage": t.leverageInCents / 100.0 if t.leverageInCents else None,
                "maxLeverage": t.maxLeverage / 100.0 if t.maxLeverage else None}

    def list_demo_accounts(self):
        """Every demo account visible under this cTrader ID's token, with
        its real current balance — not just the id/isLive fields the raw
        account list alone gives you. A real, recurring need, not a
        one-off: this same ID has had multiple demo accounts before (old
        prop-firm demos plus a fresh one, per this file's own docstring)
        and will again whenever the user opens a new one — deciding which
        ctidTraderAccountId belongs in CTRADER_DEMO_ACCOUNT_ID has to be
        done from real balances, not guessed from the id alone. Auths
        each candidate account in turn to read its balance, but never
        touches self._account_id — the bridge's own already-selected
        account keeps working exactly as before while this runs."""
        accounts_res = self._send(messages.ProtoOAGetAccountListByAccessTokenReq(
            accessToken=self._tokens["accessToken"]))
        extracted_accounts = Protobuf.extract(accounts_res)
        if type(extracted_accounts).__name__ == "ProtoOAErrorRes":
            raise BridgeError(
                f"Account list request rejected: {extracted_accounts.errorCode} — "
                f"{extracted_accounts.description}.")
        accounts = extracted_accounts.ctidTraderAccount
        demo_accounts = [a for a in accounts if not a.isLive]
        rows = []
        for a in demo_accounts:
            aid = a.ctidTraderAccountId
            balance = None
            try:
                self._send(messages.ProtoOAAccountAuthReq(ctidTraderAccountId=aid, accessToken=self._tokens["accessToken"]))
                trader_res = self._send(messages.ProtoOATraderReq(ctidTraderAccountId=aid))
                t = Protobuf.extract(trader_res).trader
                balance = t.balance / (10 ** t.moneyDigits)
            except Exception:
                pass  # this candidate couldn't be read right now — still list it, just with balance=None
            rows.append({"ctidTraderAccountId": aid, "isCurrent": aid == self._account_id, "balance": balance})
        return rows

    def reconcile(self):
        res = self._send(messages.ProtoOAReconcileReq(ctidTraderAccountId=self._account_id))
        r = Protobuf.extract(res)
        if type(r).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Reconcile request rejected: {r.errorCode} — {r.description}")
        positions = [{
            "positionId": p.positionId, "symbolId": p.tradeData.symbolId,
            "tradeSide": model.ProtoOATradeSide.Name(p.tradeData.tradeSide),
            "volume": p.tradeData.volume / 100.0, "openPrice": p.price,
            "openTimestamp": p.tradeData.openTimestamp,
            "label": p.tradeData.label or None, "comment": p.tradeData.comment or None,
            # cTrader's own real current protection on this position — was
            # never read here before (only amend_position_sltp's own return
            # value echoed it back, and only for the one position just
            # amended). Found missing live 2026-10-06 while tracing a real
            # "2 far-leg positions showing no TP in the cTrader terminal"
            # report: nothing in this project could even ask the broker
            # "does this position actually have a take-profit right now,"
            # so a silently-dropped TP (order accepted, SL/TP attachment
            # rejected — cTrader's own ProtoOAExecutionEvent doesn't
            # distinguish the two) had no way to surface anywhere.
            "stopLoss": p.stopLoss if p.HasField("stopLoss") else None,
            "takeProfit": p.takeProfit if p.HasField("takeProfit") else None,
        } for p in r.position]
        orders = [{
            "orderId": o.orderId, "symbolId": o.tradeData.symbolId,
            "orderStatus": model.ProtoOAOrderStatus.Name(o.orderStatus),
            "tradeSide": model.ProtoOATradeSide.Name(o.tradeData.tradeSide),
            "volume": o.tradeData.volume / 100.0,
        } for o in r.order]
        return {"positions": positions, "orders": orders}

    def position_unrealized_pnl(self):
        """Real floating P&L on every currently open position, in ONE
        account-wide call — ProtoOAGetPositionUnrealizedPnLReq only takes
        ctidTraderAccountId (confirmed directly against the installed
        protobuf schema, 2026-10-08), never a per-position ID; the
        response already carries one ProtoOAPositionUnrealizedPnL per open
        position. This is the real gap get_demo_balance() has been
        hardcoding unrealizedPnL=0.0 for — its own docstring already named
        the reason (a per-position ProtoOAGetPositionUnrealizedPnLReq loop
        measured at 53s for 16 positions, once stalled bot_engine.py's
        whole 60s cycle for over an hour) and named this exact message as
        the fix, just never finished wiring it up. The real fix was never
        "batch the slow thing," it was "this message was never per-
        position to begin with" — one request, every position, no loop.

        netUnrealizedPnL (not gross) — already nets out the position's own
        commission/swap the same way deals()'s own closePositionDetail
        does for a CLOSED position, so an open and a closed position's P&L
        mean the same thing when summed together."""
        res = self._send(messages.ProtoOAGetPositionUnrealizedPnLReq(ctidTraderAccountId=self._account_id))
        extracted = Protobuf.extract(res)
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Unrealized P&L request rejected: {extracted.errorCode} — {extracted.description}")
        scale = 10 ** extracted.moneyDigits if extracted.moneyDigits else 1
        return {p.positionId: p.netUnrealizedPnL / scale for p in extracted.positionUnrealizedPnL}

    def deals(self, from_ts_ms, to_ts_ms, max_rows=200):
        res = self._send(messages.ProtoOADealListReq(
            ctidTraderAccountId=self._account_id, fromTimestamp=from_ts_ms,
            toTimestamp=to_ts_ms, maxRows=max_rows))
        extracted = Protobuf.extract(res)
        if type(extracted).__name__ == "ProtoOAErrorRes":
            raise BridgeError(f"Deal list request rejected: {extracted.errorCode} — {extracted.description}")
        out = []
        for d in extracted.deal:
            row = {
                "dealId": d.dealId, "positionId": d.positionId, "symbolId": d.symbolId,
                "volume": d.volume / 100.0, "executionPrice": d.executionPrice,
                "tradeSide": model.ProtoOATradeSide.Name(d.tradeSide),
                "executionTimestamp": d.executionTimestamp,
                "dealStatus": model.ProtoOADealStatus.Name(d.dealStatus),
                # This deal's own real commission, in the account's deposit
                # currency scaled by ITS OWN moneyDigits (proto docs — same
                # convention trader()'s own balance/scale already uses, not
                # assumed to match the account-level moneyDigits). Every
                # order (open leg AND close leg) is commissioned separately
                # on this account — confirmed live, not assumed — so a real
                # round-trip cost is open_deal["commission"] +
                # close_deal["commission"], never the close leg alone.
                "commission": d.commission / (10 ** d.moneyDigits) if d.moneyDigits else d.commission,
            }
            # closePositionDetail: only present on the CLOSING deal of a
            # position — cTrader's own authoritative real swap (accrued
            # over the position's whole holding period) and real gross
            # P&L for this exact close, straight from the broker, not
            # reverse-engineered from swapLong/swapShort pips-per-day
            # (ProtoOASymbol) — this was the one honest, undecided gap
            # flagged all session ("swap unit unverifiable"); it's not
            # undecided anymore, it's a field cTrader hands back directly.
            if d.HasField("closePositionDetail"):
                cpd = d.closePositionDetail
                cpd_scale = 10 ** cpd.moneyDigits if cpd.moneyDigits else 1
                row["closePositionDetail"] = {
                    "grossProfit": cpd.grossProfit / cpd_scale,
                    "swap": cpd.swap / cpd_scale,
                    "commission": cpd.commission / cpd_scale,
                }
            out.append(row)
        return out

    def place_order(self, symbol_id, is_buy, volume_units, stop_loss=None, take_profit=None,
                    label=None, comment=None):
        """volume_units: real units of the symbol's base asset (already
        converted from this project's own dollar-notional `amount` by the
        caller, using a recent price) — this method does the units->cents
        conversion and step/min/max rounding, since that's cTrader's own
        convention, not this project's.

        label/comment: ProtoOANewOrderReq's own fields, round-tripped back
        via ProtoOATradeData on every position/order/deal this bridge
        already reads — so the strategy/timeframe that placed a trade
        shows up directly in cTrader's own terminal (Label/Comment
        columns), not just this platform's own event log."""
        detail = self.symbol_detail(symbol_id)
        step = detail["stepVolume"] or 100
        min_v = detail["minVolume"] or step
        max_v = detail["maxVolume"] or 10 ** 9
        volume_cents = round(volume_units * 100)
        volume_cents = max(min_v, min(max_v, round(volume_cents / step) * step))

        # Real, confirmed bug (a live order rejection: "Order price =
        # 85529.41908482143 has more digits than symbol allows. Allowed 3
        # digits", BTC-USD): volume already gets rounded to the symbol's
        # own step above — stopLoss/takeProfit never got the same
        # treatment, so a raw ATR-derived float (however many decimal
        # places floating-point arithmetic happened to produce) went
        # straight to cTrader, which enforces its own real per-symbol
        # digit precision and rejects anything finer. Same discipline as
        # volume: ask the symbol what precision it actually wants, conform
        # to it, don't assume the caller already did.
        digits = detail.get("digits")

        def _round_price(p):
            return round(p, digits) if (p is not None and digits is not None) else p

        stop_loss = _round_price(stop_loss)
        take_profit = _round_price(take_profit)

        kwargs = dict(
            ctidTraderAccountId=self._account_id, symbolId=symbol_id,
            orderType=model.MARKET, tradeSide=model.BUY if is_buy else model.SELL,
            volume=volume_cents, timeInForce=model.IMMEDIATE_OR_CANCEL,
        )
        if stop_loss is not None:
            kwargs["stopLoss"] = stop_loss
        if take_profit is not None:
            kwargs["takeProfit"] = take_profit
        if label is not None:
            kwargs["label"] = label
        if comment is not None:
            kwargs["comment"] = comment
        res = self._send(messages.ProtoOANewOrderReq(**kwargs))
        extracted = Protobuf.extract(res)
        payload_name = type(extracted).__name__
        # Confirmed directly: a rejected order comes back as
        # ProtoOAOrderErrorEvent (not ProtoOAErrorRes, which this
        # originally only checked for) — errorCode is a real string like
        # "TRADING_NOT_ALLOWED", description a human-readable reason. A
        # successful order comes back as ProtoOAExecutionEvent with a
        # populated `order`/`position`.
        if payload_name in ("ProtoOAErrorRes", "ProtoOAOrderErrorEvent"):
            return {"error": True, "errorCode": extracted.errorCode, "description": extracted.description}
        if payload_name == "ProtoOAExecutionEvent":
            return {
                "error": False, "orderId": extracted.order.orderId,
                "positionId": extracted.position.positionId if extracted.HasField("position") else None,
                "orderStatus": model.ProtoOAOrderStatus.Name(extracted.order.orderStatus),
            }
        return {"error": True, "errorCode": "UNKNOWN_RESPONSE", "description": f"Unexpected response type: {payload_name}"}

    def close_position(self, position_id, volume):
        """volume: REAL units of the base asset — the same scale
        reconcile()'s own volume field reports (it already divides the
        raw protocol field by 100). Caught directly before this ever ran:
        the raw ProtoOAClosePositionReq field is in cTrader's own
        hundredths-of-a-unit convention, same as place_order's volume —
        passing reconcile()'s real-unit number straight through
        unconverted would have closed at 1% of the intended size.

        position_id cast to int here — every caller's own position_id
        comes from this project's SQLite tracking (positions_seen/
        scale_out_legs' position_id column is TEXT, so every row read
        back is a real Python str), but ProtoOAClosePositionReq.positionId
        is a protobuf int64 field and rejects a str outright with a
        TypeError at construction time, not a clean cTrader-side error.
        Found via the identical, CONFIRMED bug in amend_position_sltp
        below (100% of its own real call attempts failed this exact way,
        every far-leg breakeven-stop-move this account ever earned
        silently never happened) — this site shares the same shape and
        hasn't been proven to fire cleanly yet either, so fixed alongside
        it rather than left to fail the same way whenever _close_stale_
        positions' own time-stop enforcement first actually needs it."""
        res = self._send(messages.ProtoOAClosePositionReq(
            ctidTraderAccountId=self._account_id, positionId=int(position_id), volume=round(volume * 100)))
        extracted = Protobuf.extract(res)
        payload_name = type(extracted).__name__
        if payload_name in ("ProtoOAErrorRes", "ProtoOAOrderErrorEvent"):
            return {"error": True, "errorCode": extracted.errorCode, "description": extracted.description}
        if payload_name == "ProtoOAExecutionEvent":
            return {"error": False, "orderId": extracted.order.orderId,
                    "orderStatus": model.ProtoOAOrderStatus.Name(extracted.order.orderStatus)}
        return {"error": True, "errorCode": "UNKNOWN_RESPONSE", "description": f"Unexpected response type: {payload_name}"}

    def amend_position_sltp(self, position_id, symbol_id=None, stop_loss=None, take_profit=None):
        """Moves an already-OPEN position's real stop-loss (and/or take-
        profit) — confirmed real, not guessed: ProtoOAAmendPositionSLTPReq
        (payload 2110) exists in this exact installed package, same
        family as ProtoOANewOrderReq/ProtoOAClosePositionReq above, and
        has no dedicated response type either — a success still comes
        back as ProtoOAExecutionEvent, a rejection as
        ProtoOAOrderErrorEvent/ProtoOAErrorRes, same branching as both
        methods above.

        symbol_id: optional — when given, applies place_order's own real
        per-symbol digit-precision rounding (symbol_detail(...)["digits"])
        to stop_loss/take_profit before sending, the same real bug class
        place_order's own _round_price already guards against (a raw
        ATR-derived float with more decimal places than cTrader allows
        for this symbol gets rejected outright). Omit only when the
        caller can't resolve a symbol_id — passes the price through
        unrounded rather than refusing to amend at all."""
        if symbol_id is not None:
            digits = self.symbol_detail(symbol_id).get("digits")

            def _round_price(p):
                return round(p, digits) if (p is not None and digits is not None) else p

            stop_loss = _round_price(stop_loss)
            take_profit = _round_price(take_profit)

        # Confirmed live 2026-10-06, 100% of this method's own real call
        # attempts (2 for 2 — every near-leg-target-hit this account ever
        # had): position_id arrives here as a Python str (every caller's
        # own position_id comes from this project's SQLite tracking,
        # whose position_id column is TEXT), but
        # ProtoOAAmendPositionSLTPReq.positionId is a protobuf int64 field
        # — passing a str raises a TypeError at message-construction time,
        # BEFORE this ever reaches cTrader's real server, which do_POST's
        # outer try/except then turns into a generic 500 with no hint this
        # was a type bug and not a real broker rejection. The real,
        # concrete cost: every far leg that should have had its stop
        # locked to breakeven after its near leg hit target instead kept
        # running on its ORIGINAL, riskier stop for the rest of its life —
        # scale-out's own built-in risk-reduction mechanic has never once
        # actually fired for this account.
        kwargs = dict(ctidTraderAccountId=self._account_id, positionId=int(position_id))
        if stop_loss is not None:
            kwargs["stopLoss"] = stop_loss
        if take_profit is not None:
            kwargs["takeProfit"] = take_profit
        res = self._send(messages.ProtoOAAmendPositionSLTPReq(**kwargs))
        extracted = Protobuf.extract(res)
        payload_name = type(extracted).__name__
        if payload_name in ("ProtoOAErrorRes", "ProtoOAOrderErrorEvent"):
            return {"error": True, "errorCode": extracted.errorCode, "description": extracted.description}
        if payload_name == "ProtoOAExecutionEvent":
            return {"error": False, "positionId": extracted.position.positionId,
                    "stopLoss": extracted.position.stopLoss, "takeProfit": extracted.position.takeProfit}
        return {"error": True, "errorCode": "UNKNOWN_RESPONSE", "description": f"Unexpected response type: {payload_name}"}


_bridge = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/health":
                self._json({"connected": _bridge is not None and _bridge._client.isConnected,
                             "accountId": _bridge._account_id if _bridge else None})
            elif parsed.path == "/symbols":
                self._json(_bridge.symbols())
            elif parsed.path == "/symbol":
                self._json(_bridge.symbol_detail(int(qs["symbolId"])))
            elif parsed.path == "/trendbars":
                self._json(_bridge.trendbars(int(qs["symbolId"]), qs["period"], int(qs.get("count", 200))))
            elif parsed.path == "/trader":
                self._json(_bridge.trader())
            elif parsed.path == "/accounts":
                self._json(_bridge.list_demo_accounts())
            elif parsed.path == "/reconcile":
                self._json(_bridge.reconcile())
            elif parsed.path == "/unrealized_pnl":
                self._json(_bridge.position_unrealized_pnl())
            elif parsed.path == "/ticks":
                self._json(_bridge.ticks(int(qs["symbolId"]), qs.get("type", "BID"), int(qs["fromTimestamp"]),
                                         int(qs["toTimestamp"]), raw=qs.get("raw") == "1"))
            elif parsed.path == "/tick_status":
                self._json(_bridge.tick_status())
            elif parsed.path == "/deals":
                self._json(_bridge.deals(int(qs["fromTimestamp"]), int(qs["toTimestamp"]),
                                          int(qs.get("maxRows", 200))))
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as e:
            self._json({"error": str(e)}, status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            # Found in a systematic pass over this file: every OTHER error
            # path here returns a clean {"error": ...} JSON response, but
            # this parse used to happen before the try — a malformed or
            # truncated body raised json.JSONDecodeError uncaught, which
            # HTTPServer's default per-request error handling turns into a
            # traceback on stderr and a reset connection instead of the
            # same clean error shape every other failure mode gets here.
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/order":
                result = _bridge.place_order(
                    body["symbolId"], body["isBuy"], body["volumeUnits"],
                    body.get("stopLoss"), body.get("takeProfit"),
                    body.get("label"), body.get("comment"))
                self._json(result)
            elif parsed.path == "/close":
                result = _bridge.close_position(body["positionId"], body["volume"])
                self._json(result)
            elif parsed.path == "/amend_stop_loss":
                result = _bridge.amend_position_sltp(
                    body["positionId"], body.get("symbolId"), body.get("stopLoss"), body.get("takeProfit"))
                self._json(result)
            else:
                self._json({"error": "not found"}, status=404)
        except Exception as e:
            self._json({"error": str(e)}, status=500)


class _QuietServer(ThreadingHTTPServer):
    """A caller that gives up waiting (its own timeout) closes the socket before the answer is written; the default
    handler then prints a full traceback per request, which had grown the error log to 34 MB of identical noise.
    Real errors are still reported."""

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main():
    global _bridge
    print("Connecting to cTrader (demo)...")
    try:
        _bridge = CTraderBridge()
        _bridge.start()
        print(f"Connected. Demo account: {_bridge._account_id}. "
              f"Serving bridge API on http://localhost:{_BRIDGE_PORT}")
        # Found live: bot_engine.py now sources live detection candles
        # straight from this bridge (one real round trip per unique
        # ticker/timeframe, ~0.5-2s each against cTrader's own servers) —
        # a plain single-threaded HTTPServer means bot_engine's own
        # sequential loop of those calls blocks every OTHER caller (the
        # API server's balance/reconcile checks) from even being ACCEPTED
        # until the whole loop finishes, not just from being answered.
        # CTraderBridge._send()'s own self._lock already serializes access
        # to the real underlying connection correctly regardless of how
        # many OS threads call in, so switching the HTTP layer to threaded
        # only removes the OS-level head-of-line blocking on TOP of that
        # — a caller now waits on the lock (bounded by whichever single
        # call currently holds it), not behind an entire queue of
        # sequential requests from someone else's unrelated loop.
        _QuietServer(("localhost", _BRIDGE_PORT), _Handler).serve_forever()
    except Exception as e:
        # Confirmed as the real cause of a genuine ~4-hour silent hang: a
        # network hiccup exactly like a MacBook lid-close/wake or a wifi
        # handoff makes start()'s own connection/auth calls time out
        # (they're real, bounded — @crochet.wait_for(timeout=20) on
        # _send_async) and raise, correctly. But nothing here used to
        # catch it — the main thread died on the uncaught exception while
        # crochet.setup()'s own reactor thread (started at import time,
        # non-daemon by design so it outlives ordinary thread cleanup)
        # kept running with nothing left to do. The PROCESS never exited,
        # so launchd's KeepAlive (which only restarts on exit) never
        # fired: an empty shell, port dead, alive forever. _health_check_
        # loop already gets this right for the RUNNING case (see its own
        # os._exit(1)) — this wraps the WHOLE startup+serve path in the
        # same discipline, not just start(), since serve_forever() failing
        # (e.g. the port still held by a not-quite-dead previous instance)
        # would hit the identical zombie-process failure mode.
        print(f"main() failed ({e}) — exiting for launchd to restart.")
        os._exit(1)


if __name__ == "__main__":
    main()
