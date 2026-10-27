"""The researched universe and the data-fetch recipe the discovery sweeps and the momentum forward test share."""

TF_FETCH = {"1D": dict(period="10y", interval="1d"), "1h": dict(period="730d", interval="1h")}


def universe():
    """Every ticker the main research ledger has ever scored — the real
    researched universe, read BEFORE a sweep redirects its own ledger."""
    import experiments as ex
    t = ex.load_experiment_trials()
    return sorted(t[t["verdict"] == "SCORED"]["ticker"].unique())
