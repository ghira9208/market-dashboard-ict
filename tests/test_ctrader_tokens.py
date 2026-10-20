"""ctrader_tokens.py — the login token must survive restart storms and two bridges sharing one file."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ctrader_tokens as ct


def _write(path, access="A1", refresh="R1", obtained=None, **extra):
    d = {"accessToken": access, "refreshToken": refresh, "expiresIn": 2628000, **extra}
    if obtained is not None:
        d["obtainedAt"] = obtained
    Path(path).write_text(json.dumps(d))


class Refresher:
    """A fake cTrader token endpoint that, like the real one, rotates BOTH tokens and counts how often it's used."""
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
        self.lock = threading.Lock()

    def __call__(self, refresh_token):
        with self.lock:
            self.calls += 1
            n = self.calls
        time.sleep(0.05)
        if self.fail:
            raise RuntimeError("cTrader said no")
        return {"accessToken": f"A{n + 1}", "refreshToken": f"R{n + 1}", "expiresIn": 2628000}


def test_a_young_token_is_never_refreshed_just_because_a_process_started(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, obtained=time.time() - 3600)
    r = Refresher()
    for _ in range(5):                                           # a restart storm
        assert ct.ensure_fresh(p, r)["accessToken"] == "A1"
    assert r.calls == 0


def test_an_old_token_is_refreshed_once_and_saved_with_a_timestamp(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, obtained=time.time() - 25 * 86400)
    r = Refresher()
    out = ct.ensure_fresh(p, r)
    assert r.calls == 1 and out["accessToken"] == "A2" and json.loads(Path(p).read_text())["obtainedAt"] > time.time() - 5
    ct.ensure_fresh(p, r)
    assert r.calls == 1                                          # now young again


def test_two_bridges_starting_together_spend_the_refresh_token_exactly_once(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, obtained=time.time() - 25 * 86400)
    r = Refresher()
    results = []
    threads = [threading.Thread(target=lambda: results.append(ct.ensure_fresh(p, r)["accessToken"])) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert r.calls == 1 and set(results) == {"A2"}                # everyone ends up on the SAME, valid token
    assert json.loads(Path(p).read_text())["accessToken"] == "A2"


def test_a_failed_refresh_keeps_the_old_token_and_is_not_retried_in_a_loop(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, obtained=time.time() - 25 * 86400)
    r = Refresher(fail=True)
    assert ct.ensure_fresh(p, r)["accessToken"] == "A1"
    assert ct.ensure_fresh(p, r)["accessToken"] == "A1"
    assert r.calls == 1                                          # cooldown: a crash loop can't hammer the token endpoint
    assert json.loads(Path(p).read_text())["refreshToken"] == "R1"     # and the good refresh token was not lost


def test_a_response_without_an_access_token_never_overwrites_a_good_file(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, obtained=time.time() - 25 * 86400)
    out = ct.ensure_fresh(p, lambda rt: {"errorCode": "BAD", "description": "nope"})
    assert out["accessToken"] == "A1" and json.loads(Path(p).read_text())["accessToken"] == "A1"
    with pytest.raises(ValueError):
        ct.save_atomic(p, {"errorCode": "BAD"})


def test_without_a_timestamp_the_files_own_age_is_used(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p)                                                   # legacy file: no obtainedAt
    os.utime(p, (time.time() - 25 * 86400,) * 2)
    r = Refresher()
    ct.ensure_fresh(p, r)
    assert r.calls == 1


def test_a_rejected_token_is_replaced_by_a_newer_one_another_process_saved(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, access="NEW", refresh="RNEW", obtained=time.time())
    assert ct.reread_if_changed(p, {"accessToken": "OLD"})["accessToken"] == "NEW"
    assert ct.reread_if_changed(p, {"accessToken": "NEW"}) is None


def test_forcing_adopts_a_refresh_someone_else_just_did_instead_of_refreshing_again(tmp_path):
    p = str(tmp_path / "t.json")
    _write(p, access="A1", obtained=time.time() - 3600)
    held = ct.load(p)
    _write(p, access="A2", refresh="R2", obtained=time.time())   # another bridge refreshed meanwhile
    seen = []
    r = Refresher()
    out = ct.ensure_fresh(p, lambda rt: seen.append(rt) or r(rt), force=True)
    assert seen == ["R2"]                                        # spent the NEWEST refresh token, never the stale one
