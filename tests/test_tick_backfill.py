"""tick_backfill.fetch_day pages backwards through a day, de-duplicates boundary ticks and can never loop forever."""
import tick_backfill as tb


def _pager(pages):
    calls = []

    def get_page(sid, side, f, t):
        calls.append(t)
        return pages[len(calls) - 1]
    return get_page, calls


def test_walks_back_until_no_more_and_returns_oldest_first():
    get, calls = _pager([{"ticks": [[300, 3.0], [200, 2.0]], "hasMore": True},
                         {"ticks": [[199, 1.5], [100, 1.0]], "hasMore": False}])
    rows = tb.fetch_day(get, 1, "BID", 0, 1000, sleep=lambda s: None)
    assert rows == [(100, 1.0), (199, 1.5), (200, 2.0), (300, 3.0)]
    assert calls == [1000, 199]            # second page starts just before the oldest tick of the first


def test_a_repeated_boundary_tick_is_kept_once():
    get, _ = _pager([{"ticks": [[300, 3.0], [200, 2.0]], "hasMore": True}, {"ticks": [[200, 2.0], [100, 1.0]], "hasMore": False}])
    assert len(tb.fetch_day(get, 1, "BID", 0, 1000, sleep=lambda s: None)) == 3


def test_no_progress_stops_instead_of_looping():
    get = lambda *a: {"ticks": [[5000, 1.0]], "hasMore": True}     # claims more, but nothing older than the window end
    assert tb.fetch_day(get, 1, "BID", 0, 1000, sleep=lambda s: None) == [(5000, 1.0)]


def test_empty_day():
    assert tb.fetch_day(lambda *a: {"ticks": [], "hasMore": False}, 1, "BID", 0, 1000) == []
