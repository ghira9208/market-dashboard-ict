"""Pull historical bid/ask ticks from the cTrader bridge (ProtoOAGetTickDataReq, 10,000 ticks a page, newest first)
into ~/dev/tick-data/history/<SYMBOL>/<UTC day>_<BID|ASK>.csv. Throttled and resumable: a finished day leaves a
file behind and is skipped next time, and every request is followed by a pause so live orders sharing the bridge's
single connection never queue behind a long run. Uses the second account's bridge (8701) by default.

    python3 tick_backfill.py [days=30] [port=8701]
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/dev/tick-data/history")
PAUSE = 1.5
DAY_MS = 86_400_000


def fetch_day(get_page, symbol_id, side, day_start_ms, day_end_ms, sleep=time.sleep, pause=PAUSE):
    """All ticks of one UTC day, oldest first. get_page(symbol_id, side, from_ms, to_ms) -> {"ticks": [[ts, px]...] newest
    first, "hasMore": bool}. Pages walk backwards from the end of the day until the broker says there is no more."""
    out, to_ms = [], day_end_ms
    for _ in range(200):
        page = get_page(symbol_id, side, day_start_ms, to_ms)
        ticks = page.get("ticks") or []
        out.extend(ticks)
        if not page.get("hasMore") or not ticks:
            break
        oldest = min(t[0] for t in ticks)
        if oldest - 1 >= to_ms:        # no progress: stop instead of looping forever
            break
        to_ms = oldest - 1
        sleep(pause)
    uniq = {t[0:2][0]: t[1] for t in out}          # a boundary tick can repeat across pages
    return sorted(uniq.items())


def main(days=30, port=8701):
    base = f"http://localhost:{port}"
    names = json.load(open(os.path.join(HERE, ".cache", "tick_symbols.json")))
    ids = {s["symbolName"]: s["symbolId"] for s in requests.get(f"{base}/symbols", timeout=30).json()}

    def get_page(sid, side, f, t):
        r = requests.get(f"{base}/ticks", params={"symbolId": sid, "type": side, "fromTimestamp": f, "toTimestamp": t},
                         timeout=120)
        r.raise_for_status()
        j = r.json()
        if "error" in j:
            raise RuntimeError(j["error"])
        return j

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(1, days + 1):
        day = today - timedelta(days=back)
        if day.weekday() >= 5:                      # Saturday/Sunday: FX/index CFDs closed (crypto weekends skipped too, for now)
            continue
        s_ms = int(day.timestamp() * 1000)
        for name in names:
            for side in ("BID", "ASK"):
                path = os.path.join(ROOT, name, f"{day:%Y-%m-%d}_{side}.csv")
                if os.path.exists(path) or name not in ids:
                    continue
                try:
                    rows = fetch_day(get_page, ids[name], side, s_ms, s_ms + DAY_MS - 1)
                except Exception as e:
                    print(f"{day:%Y-%m-%d} {name} {side}: {e}", flush=True)
                    time.sleep(10)
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".tmp", "w") as fh:
                    fh.write("ts_ms,price\n")
                    fh.writelines(f"{ts},{px:.5f}\n" for ts, px in rows)
                os.replace(path + ".tmp", path)
                print(f"{day:%Y-%m-%d} {name} {side}: {len(rows)} ticks", flush=True)
                time.sleep(PAUSE)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, int(sys.argv[2]) if len(sys.argv) > 2 else 8701)
