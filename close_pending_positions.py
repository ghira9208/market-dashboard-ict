"""One-shot: closes the two untracked GBPUSD positions (292685390,
292688679) found via reconciliation on 2026-09-27, the moment the FX
market actually accepts it. Retries every 5 minutes since MARKET_CLOSED
is a real, honest broker response on a Sunday, not an error to guess a
fixed reopen time around — gives up after 12 hours if something else is
actually wrong, rather than retrying forever silently.

One-off script for this specific cleanup — not meant to be scheduled or reused."""
import sys
import time

sys.path.insert(0, "/Users/apple/dev/etoro-dashboard")
from notify import notify

import requests

POSITIONS = [
    {"positionId": 292685390, "volume": 20000.0},
    {"positionId": 292688679, "volume": 1000.0},
]
_BRIDGE = "http://localhost:8700"
_MAX_WAIT_SECONDS = 12 * 3600
_RETRY_EVERY = 300


def try_close_all():
    results = []
    for p in POSITIONS:
        try:
            r = requests.post(f"{_BRIDGE}/close", json=p, timeout=10).json()
        except Exception as e:
            r = {"error": True, "errorCode": "BRIDGE_UNREACHABLE", "description": str(e)}
        results.append((p["positionId"], r))
    return results


def main():
    t0 = time.time()
    while time.time() - t0 < _MAX_WAIT_SECONDS:
        results = try_close_all()
        still_pending = [pid for pid, r in results if r.get("error")]
        if not still_pending:
            print("Both positions closed successfully.")
            notify("Cleanup done", "Both untracked GBPUSD positions closed once the market reopened.",
                   severity="good")
            return
        last_reason = next(r["description"] for pid, r in results if pid == still_pending[-1])
        print(f"Still pending ({last_reason}) — retrying in {_RETRY_EVERY}s.")
        time.sleep(_RETRY_EVERY)

    notify("Cleanup failed", "Could not close the two pending GBPUSD positions within 12 hours — "
                             "worth checking manually, this may not just be a closed-market issue.",
           severity="critical")
    print("Gave up after 12 hours — see above for the last real error.")


if __name__ == "__main__":
    main()
