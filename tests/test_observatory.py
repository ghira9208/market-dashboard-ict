"""observatory.py — on-demand Strategy Observatory analytics (2026-10-08).
Network/disk data fetching and the real touch-finder builders are
monkeypatched out here (both already have their own test coverage
elsewhere); these tests are about observatory.py's OWN aggregation logic
— regime_matrix/rolling_stability/parameter_grid — and the "never writes
to the trial log" contract."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments as ex
import observatory as obs


def _fake_events(rows):
    return pd.DataFrame(rows, columns=["entry_time", "r_real", "r_flip", "direction",
                                        "mfe_r", "mae_r", "vol_regime", "trend_regime"])


@pytest.fixture(autouse=True)
def _clear_caches():
    obs._detail_cache.clear()
    obs._grid_cache.clear()
    yield
    obs._detail_cache.clear()
    obs._grid_cache.clear()


@pytest.fixture
def no_network(monkeypatch):
    # trade_detail/regime_matrix/rolling_stability only ever need a df to
    # exist and be non-empty before calling the (also monkeypatched)
    # builder — contents never matter once the builder itself is faked.
    monkeypatch.setattr(obs, "_fetch_df", lambda ticker, tf_label: pd.DataFrame({"Close": [1.0, 2.0]}))


def test_trade_detail_never_calls_append_trial(monkeypatch, no_network):
    # The core safety contract: viewing a chart must never write a trial.
    called = []
    monkeypatch.setattr(ex, "_append_trial", lambda trial: called.append(trial))
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr",
                         lambda df, **kw: _fake_events([
                             {"entry_time": pd.Timestamp("2026-01-01"), "r_real": 1.0, "r_flip": -1.0,
                              "direction": "bullish", "mfe_r": 1.5, "mae_r": 0.3,
                              "vol_regime": "normal", "trend_regime": "trend"},
                         ]))
    rows = obs.trade_detail("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0})
    assert len(rows) == 1
    assert rows[0]["r_real"] == 1.0
    assert called == []


def test_trade_detail_strips_underscore_settings_before_calling_builder(monkeypatch, no_network):
    seen = {}
    def fake_builder(df, **kw):
        seen.update(kw)
        return _fake_events([])
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", fake_builder)
    obs.trade_detail("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 2.0, "_strategy_id": "abc123"})
    assert "_strategy_id" not in seen
    assert seen["rr_multiple"] == 2.0


def test_trade_detail_is_cached_within_ttl(monkeypatch, no_network):
    calls = []
    def fake_builder(df, **kw):
        calls.append(1)
        return _fake_events([])
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", fake_builder)
    obs.trade_detail("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0})
    obs.trade_detail("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0})
    assert len(calls) == 1  # second call served from cache, not a second simulation


def test_regime_matrix_buckets_by_vol_and_trend(monkeypatch, no_network):
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", lambda df, **kw: _fake_events([
        {"entry_time": pd.Timestamp("2026-01-01"), "r_real": 2.0, "r_flip": 0, "direction": "bullish",
         "mfe_r": 2.0, "mae_r": 0.1, "vol_regime": "high", "trend_regime": "trend"},
        {"entry_time": pd.Timestamp("2026-01-02"), "r_real": -1.0, "r_flip": 0, "direction": "bearish",
         "mfe_r": 0.2, "mae_r": 1.0, "vol_regime": "high", "trend_regime": "trend"},
        {"entry_time": pd.Timestamp("2026-01-03"), "r_real": 0.5, "r_flip": 0, "direction": "bullish",
         "mfe_r": 0.5, "mae_r": 0.2, "vol_regime": "low", "trend_regime": "range"},
    ]))
    result = obs.regime_matrix("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0})
    assert result["n_total"] == 3
    high_trend = next(c for c in result["cells"] if c["vol_regime"] == "high" and c["trend_regime"] == "trend")
    assert high_trend["n"] == 2
    assert high_trend["mean_r"] == pytest.approx(0.5)  # (2.0 + -1.0) / 2
    low_range = next(c for c in result["cells"] if c["vol_regime"] == "low" and c["trend_regime"] == "range")
    assert low_range["n"] == 1
    empty_cell = next(c for c in result["cells"] if c["vol_regime"] == "normal")
    assert empty_cell["n"] == 0
    assert empty_cell["mean_r"] is None  # never a fabricated 0 for an empty cell


def test_regime_matrix_counts_unclassified_without_dropping_them(monkeypatch, no_network):
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", lambda df, **kw: _fake_events([
        {"entry_time": pd.Timestamp("2026-01-01"), "r_real": 1.0, "r_flip": 0, "direction": "bullish",
         "mfe_r": 1.0, "mae_r": 0.1, "vol_regime": None, "trend_regime": None},
    ]))
    result = obs.regime_matrix("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0})
    assert result["n_total"] == 1
    assert result["n_unclassified"] == 1
    assert sum(c["n"] for c in result["cells"]) == 0  # the unclassified trade isn't force-bucketed into any cell


def test_rolling_stability_is_chronological_and_carries_window_size(monkeypatch, no_network):
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", lambda df, **kw: _fake_events([
        {"entry_time": pd.Timestamp("2026-01-03"), "r_real": -1.0, "r_flip": 0, "direction": "bullish",
         "mfe_r": 0, "mae_r": 1.0, "vol_regime": "normal", "trend_regime": "range"},
        {"entry_time": pd.Timestamp("2026-01-01"), "r_real": 1.0, "r_flip": 0, "direction": "bullish",
         "mfe_r": 1.0, "mae_r": 0, "vol_regime": "normal", "trend_regime": "range"},
        {"entry_time": pd.Timestamp("2026-01-02"), "r_real": 2.0, "r_flip": 0, "direction": "bullish",
         "mfe_r": 2.0, "mae_r": 0, "vol_regime": "normal", "trend_regime": "range"},
    ]))
    out = obs.rolling_stability("EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0}, window_trades=2)
    assert [o["entry_time"] for o in out] == ["2026-01-01T00:00:00", "2026-01-02T00:00:00", "2026-01-03T00:00:00"]
    assert out[0]["n_in_window"] == 1  # first point: nothing to roll yet
    assert out[0]["rolling_mean_r"] == pytest.approx(1.0)
    assert out[2]["n_in_window"] == 2  # window_trades=2, third point only sees the last 2
    assert out[2]["rolling_mean_r"] == pytest.approx((2.0 + -1.0) / 2)


def test_parameter_grid_covers_every_cell_and_marks_production_point(monkeypatch, no_network):
    def fake_builder(df, **kw):
        # mean_r deterministic from the cell's own params, so the test can
        # assert exact values without depending on real touch-detection.
        return _fake_events([
            {"entry_time": pd.Timestamp("2026-01-01"), "r_real": kw["rr_multiple"] - kw["atr_mult_stop"],
             "r_flip": 0, "direction": "bullish", "mfe_r": 0, "mae_r": 0, "vol_regime": None, "trend_regime": None},
        ])
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", fake_builder)
    result = obs.parameter_grid(
        "EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 2.0, "atr_mult_stop": 1.0},
        "rr_multiple", [1.0, 2.0], "atr_mult_stop", [0.5, 1.0],
    )
    assert len(result["cells"]) == 4
    assert result["production_x"] == 2.0 and result["production_y"] == 1.0
    cell = next(c for c in result["cells"] if c["rr_multiple"] == 2.0 and c["atr_mult_stop"] == 0.5)
    assert cell["mean_r"] == pytest.approx(1.5)


def test_parameter_grid_never_calls_append_trial(monkeypatch, no_network):
    called = []
    monkeypatch.setattr(ex, "_append_trial", lambda trial: called.append(trial))
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", lambda df, **kw: _fake_events([]))
    obs.parameter_grid(
        "EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 2.0, "atr_mult_stop": 1.0},
        "rr_multiple", [1.0, 2.0, 3.0], "atr_mult_stop", [0.5, 1.0, 1.5],
    )
    assert called == []


def test_parameter_grid_a_failing_cell_reads_zero_n_not_a_crash(monkeypatch, no_network):
    def flaky_builder(df, **kw):
        if kw["rr_multiple"] == 99.0:
            raise ValueError("unreachable rr_multiple")
        return _fake_events([])
    monkeypatch.setitem(obs.BUILDER, "clean_expansion_fixed_rr", flaky_builder)
    result = obs.parameter_grid(
        "EURUSD=X", "1h", "clean_expansion_fixed_rr", {"rr_multiple": 1.0, "atr_mult_stop": 1.0},
        "rr_multiple", [1.0, 99.0], "atr_mult_stop", [1.0],
    )
    bad_cell = next(c for c in result["cells"] if c["rr_multiple"] == 99.0)
    assert bad_cell["n"] == 0 and bad_cell["mean_r"] is None


# ---- effective_settings: simulate the target the bot really trades ----

def test_effective_settings_multiplies_the_target_by_the_validated_widen_factor():
    out = obs.effective_settings({"rr_multiple": 1.5, "atr_mult_stop": 1.0}, 4.0)
    assert out["rr_multiple"] == pytest.approx(6.0) and out["atr_mult_stop"] == 1.0


def test_effective_settings_leaves_unwidened_and_unknown_factors_alone_and_never_mutates():
    base = {"rr_multiple": 2.0}
    assert obs.effective_settings(base, 1.0) == {"rr_multiple": 2.0}
    assert obs.effective_settings(base, None) == {"rr_multiple": 2.0}
    assert obs.effective_settings(base, float("nan")) == {"rr_multiple": 2.0}
    obs.effective_settings(base, 3.0)
    assert base == {"rr_multiple": 2.0}
