"""Records live bid/ask ticks from the cTrader spot feed to disk, for tick-level research.

Lives inside the bridge process (the only authenticated cTrader connection). Design rules, because the bridge also
carries live orders:
  * the message callback runs on the Twisted reactor thread for EVERY message, including order responses, and the
    library calls it before resolving the request's deferred — so it must be tiny and must never raise;
  * ticks go into an in-memory deque; a separate thread appends them to disk every few seconds;
  * files live OUTSIDE Google Drive (default ~/dev/tick-data) — a day of ticks is ~100 MB of constantly changing
    files, which Drive would try to sync;
  * one CSV per symbol per UTC day: ts_ms,bid,ask  (prices in normal units; the other side is carried forward when a
    spot event only updates one of them).
"""
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

SPOT_EVENT = 2131            # ProtoOAPayloadType.PROTO_OA_SPOT_EVENT
PRICE_SCALE = 100000.0       # spot prices are integers in 1/100000 of a price unit
DEFAULT_ROOT = os.path.expanduser("~/dev/tick-data")
FLUSH_EVERY = 5.0
MAX_BUFFER = 500_000         # hard cap: if the disk thread ever stalls, drop the oldest rather than grow without bound


class TickRecorder:
    def __init__(self, root=DEFAULT_ROOT, flush_every=FLUSH_EVERY):
        self.root = root
        self._names = {}                 # symbolId -> symbolName
        self._last = {}                  # symbolId -> (bid, ask) carried forward
        self._last_at = {}               # symbolId -> wall-clock seconds of its newest tick
        self._buf = deque(maxlen=MAX_BUFFER)
        self._flush_every = flush_every
        self._stop = threading.Event()
        self._thread = None
        self.events = 0                  # spot events seen since start
        self.last_event_at = None        # wall-clock seconds of the newest one
        self.written = 0
        self.errors = 0

    # ----------------------------------------------------------- feed --
    def set_symbols(self, names_by_id):
        self._names = dict(names_by_id)

    def on_message(self, client, message):
        """Message callback. Never raises, never blocks: anything unexpected is counted and ignored."""
        try:
            if message.payloadType != SPOT_EVENT:
                return
            from ctrader_open_api import Protobuf
            self.add_event(Protobuf.extract(message))
        except Exception:
            self.errors += 1

    def add_event(self, ev):
        sid = ev.symbolId
        if sid not in self._names:
            return
        bid, ask = self._last.get(sid, (None, None))
        if ev.HasField("bid"):
            bid = ev.bid / PRICE_SCALE
        if ev.HasField("ask"):
            ask = ev.ask / PRICE_SCALE
        self._last[sid] = (bid, ask)
        if bid is None or ask is None:
            return                       # need both sides before a tick is meaningful
        now = time.time()
        ts_ms = ev.timestamp if ev.HasField("timestamp") else int(now * 1000)
        self._buf.append((sid, ts_ms, bid, ask))
        self.events += 1
        self.last_event_at = now
        self._last_at[sid] = now

    def quote(self, symbol_id):
        """Latest (bid, ask) for a symbol and how old it is, or None when nothing has ticked yet."""
        bid_ask = self._last.get(symbol_id)
        at = self._last_at.get(symbol_id)
        if bid_ask is None or at is None or bid_ask[0] is None or bid_ask[1] is None:
            return None
        return {"bid": bid_ask[0], "ask": bid_ask[1], "age_s": round(time.time() - at, 1)}

    # ----------------------------------------------------------- disk --
    @staticmethod
    def _path(root, name, ts_ms):
        day = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        return os.path.join(root, day, f"{name}.csv")

    def flush(self):
        """Append everything buffered to disk. Returns the number of rows written."""
        rows = []
        while self._buf:
            try:
                rows.append(self._buf.popleft())
            except IndexError:
                break
        if not rows:
            return 0
        by_file = {}
        for sid, ts_ms, bid, ask in rows:
            by_file.setdefault(self._path(self.root, self._names.get(sid, str(sid)), ts_ms), []).append(
                f"{ts_ms},{bid:.5f},{ask:.5f}\n")
        for path, lines in by_file.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a") as fh:
                if new:
                    fh.write("ts_ms,bid,ask\n")
                fh.writelines(lines)
        self.written += len(rows)
        return len(rows)

    def _loop(self):
        while not self._stop.wait(self._flush_every):
            try:
                self.flush()
            except Exception:
                self.errors += 1

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="tick-recorder")
            self._thread.start()

    def stop(self):
        self._stop.set()
        self.flush()

    def status(self):
        return {"symbols": len(self._names), "events": self.events, "written": self.written, "errors": self.errors,
                "buffered": len(self._buf),
                "seconds_since_last_event": None if self.last_event_at is None else round(time.time() - self.last_event_at, 1)}
