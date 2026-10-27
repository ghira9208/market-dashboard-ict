"""tick_recorder.py runs inside the live bridge: it must never raise from the message callback, never write a
half-known tick, carry the unchanged side forward, and put each tick in its own UTC-day file."""
import tick_recorder as tr


class _Ev:
    def __init__(self, sid, bid=None, ask=None, ts=None):
        self.symbolId, self.bid, self.ask, self.timestamp = sid, bid, ask, ts

    def HasField(self, name):
        return getattr(self, name) is not None


def _rec(tmp_path):
    r = tr.TickRecorder(root=str(tmp_path))
    r.set_symbols({1: "EURUSD", 2: "GBPUSD"})
    return r


def test_needs_both_sides_then_carries_the_unchanged_side(tmp_path):
    r = _rec(tmp_path)
    r.add_event(_Ev(1, bid=108000))                       # ask unknown yet: nothing recorded
    assert r.flush() == 0
    r.add_event(_Ev(1, ask=108012, ts=1_700_000_000_000))
    r.add_event(_Ev(1, bid=108002, ts=1_700_000_001_000))  # only bid changed: ask carried forward
    assert r.flush() == 2
    lines = (tmp_path / "2023-11-14" / "EURUSD.csv").read_text().splitlines()
    assert lines == ["ts_ms,bid,ask", "1700000000000,1.08000,1.08012", "1700000001000,1.08002,1.08012"]


def test_unknown_symbols_are_ignored_and_files_split_by_utc_day(tmp_path):
    r = _rec(tmp_path)
    r.add_event(_Ev(99, bid=1, ask=2, ts=1_700_000_000_000))
    r.add_event(_Ev(2, bid=125000, ask=125010, ts=1_700_000_000_000))
    r.add_event(_Ev(2, bid=125001, ask=125011, ts=1_700_000_000_000 + 86_400_000))
    assert r.flush() == 2
    assert (tmp_path / "2023-11-14" / "GBPUSD.csv").exists() and (tmp_path / "2023-11-15" / "GBPUSD.csv").exists()


def test_the_message_callback_never_raises(tmp_path):
    r = _rec(tmp_path)
    r.on_message(None, object())                           # no payloadType at all
    class _Msg: payloadType = tr.SPOT_EVENT                # a spot event that cannot be decoded
    r.on_message(None, _Msg())
    assert r.errors >= 1


def test_buffer_is_bounded(tmp_path):
    r = _rec(tmp_path)
    for i in range(tr.MAX_BUFFER + 10):
        r._buf.append((1, i, 1.0, 1.0))
    assert len(r._buf) == tr.MAX_BUFFER
