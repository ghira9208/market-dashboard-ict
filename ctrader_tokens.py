"""One place that owns the cTrader login token file.

Why this exists: cTrader hands out a new access token AND a new refresh token on every refresh, and the previous
access token stops working. The bridge used to refresh on EVERY start. So any burst of restarts (the watchdog kicking
a bridge, a wake from sleep, a second bridge for a second account starting beside the first) meant several processes
refreshing at once, each invalidating the others' tokens, and the saved file ended up pointing at a dead token. The
bridge then crash-looped on CH_ACCESS_TOKEN_INVALID until someone re-authorised by hand. It "always happened" — on
2026-10-09 with one bridge, and again 2026-10-20 with two.

The rules now:
  * refresh only when the token is actually getting old (REFRESH_AFTER_SECONDS), never merely because a process started;
  * only one process at a time may refresh (exclusive file lock), and it re-reads the file inside the lock — if another
    process already refreshed, it adopts that result instead of spending the refresh token a second time;
  * the file is replaced atomically (temp file + rename) so a crash can never leave half a token, and a response
    without an access token is never written over a good one;
  * after a failed refresh, don't try again for COOLDOWN_SECONDS, so a crash loop can't hammer cTrader's token endpoint.
"""
import fcntl
import json
import os
import time

REFRESH_AFTER_SECONDS = 20 * 24 * 3600     # access tokens last ~30.4 days (expiresIn 2,628,000 s)
COOLDOWN_SECONDS = 600


def _lock_path(path):
    return path + ".lock"


def load(path):
    with open(path) as f:
        return json.load(f)


def age_seconds(tokens, path, now=None):
    """How old the access token is: its recorded obtainedAt, else the file's own modification time."""
    now = now if now is not None else time.time()
    stamp = tokens.get("obtainedAt")
    if stamp is None:
        try:
            stamp = os.path.getmtime(path)
        except OSError:
            return float("inf")
    return now - float(stamp)


def save_atomic(path, tokens, now=None):
    """Stamps obtainedAt and replaces the file atomically. Refuses to write anything without an access token."""
    if not tokens or not tokens.get("accessToken"):
        raise ValueError("refusing to save a token response that has no accessToken")
    out = dict(tokens)
    out["obtainedAt"] = now if now is not None else time.time()
    out.pop("lastRefreshFailedAt", None)
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return out


def ensure_fresh(path, refresh_fn, now=None, force=False):
    """Returns the tokens to use. `refresh_fn(refresh_token)` must return the cTrader token response dict (or raise).
    Only refreshes when the token is old (or force=True), under an exclusive lock, adopting a refresh another process
    already did. Never raises for a failed refresh: returns the tokens it already has."""
    now = now if now is not None else time.time()
    tokens = load(path)
    if not force and age_seconds(tokens, path, now) < REFRESH_AFTER_SECONDS:
        return tokens
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(_lock_path(path), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            current = load(path)                                   # re-read INSIDE the lock
            if current.get("accessToken") != tokens.get("accessToken"):
                return current                                     # another process already refreshed: adopt it
            if not force and age_seconds(current, path, now) < REFRESH_AFTER_SECONDS:
                return current
            failed = current.get("lastRefreshFailedAt")
            if failed and now - failed < COOLDOWN_SECONDS:
                return current                                     # a refresh just failed: don't hammer the endpoint
            try:
                response = refresh_fn(current["refreshToken"])
                if not response or not response.get("accessToken"):
                    raise ValueError("refresh returned no accessToken")
                return save_atomic(path, response, now)
            except Exception:
                current["lastRefreshFailedAt"] = now
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(current, f, indent=2)
                os.replace(tmp, path)
                return current
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reread_if_changed(path, held):
    """Used when a start is rejected with an invalid token: another process may have refreshed since we loaded.
    Returns the on-disk tokens if they differ from what we hold, else None."""
    try:
        disk = load(path)
    except (OSError, ValueError):
        return None
    return disk if disk.get("accessToken") and disk.get("accessToken") != held.get("accessToken") else None
