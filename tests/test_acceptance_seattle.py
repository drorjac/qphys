"""Acceptance test 11: the real-data comparison, where the HQMM loses.

This is the headline negative result, so it is pinned twice: the committed
table in `reports/` must keep showing the loss, and the comparison is re-run
(at reduced restarts) to check the loss is a property of the data rather than
of one saved CSV.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qphys.collapse import data as D
from qphys.collapse import experiments as E

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def test_the_data_is_what_it_claims_to_be():
    series = D.seattle_wet_dry()
    assert len(series) == 1461
    assert series.values.mean() == pytest.approx(0.426, abs=0.001)
    if series.is_real:
        assert series.provenance["sha256"]
        assert series.provenance["bytes"] > 0


def test_the_split_is_temporal_and_not_pathological():
    """Honesty rule 4: fixed before the first fit, and not a lucky cut."""
    train, test = D.temporal_split(D.seattle_wet_dry().values)
    assert len(train) == 1022
    assert len(test) == 439
    assert test.mean() == pytest.approx(train.mean(), abs=0.02)


def test_reported_table_still_shows_the_hqmm_losing():
    """The claims table says claim 1 is NOT SHOWN. If a future run overturns
    that, this fails and the README must be rewritten -- not the other way
    round."""
    path = REPORTS / "seattle_table.csv"
    if not path.exists():
        pytest.skip("run qphys.collapse.experiments.seattle_comparison first")
    import pandas as pd

    df = pd.read_csv(path)
    for _, row in df[df.model.str.startswith("hqmm")].iterrows():
        rival = E.best_at_or_below(df, row.params)
        assert rival.nll <= row.nll, (
            f"{row.model} ({row.params} params, {row.nll:.4f} bits) now beats "
            f"every classical model at or below its capacity (best was "
            f"{rival.model} at {rival.nll:.4f}). The README claims table says "
            "this has NOT been shown -- update the README, not this test."
        )


def test_the_best_model_overall_is_classical():
    path = REPORTS / "seattle_table.csv"
    if not path.exists():
        pytest.skip("comparison not run")
    import pandas as pd

    df = pd.read_csv(path)
    assert not df.loc[df.nll.idxmin()].model.startswith("hqmm")


def test_memory_saving_is_large_and_is_not_a_prediction_claim():
    """The REPRESENTATION result: ~72% on the estimated chain.

    Holding at the same time as the prediction result failing is the point.
    """
    m = E.seattle_memory(save=False)
    assert m["C_q_bits"] <= m["C_mu_bits"]
    assert m["C_mu_bits"] == pytest.approx(0.9844, abs=0.002)
    assert m["C_q_bits"] == pytest.approx(0.2763, abs=0.002)
    assert m["saving_pct"] == pytest.approx(71.9, abs=1.0)


def test_memory_saving_survives_a_change_of_symbolisation():
    """A result surviving only one symbolisation is not a result."""
    sweep = E.symbolisation_sweep()
    assert (sweep.C_q <= sweep.C_mu + 1e-12).all()
    assert (100 * (1 - sweep.C_q / sweep.C_mu) > 50).all()


@pytest.mark.slow
def test_hqmm_loses_on_a_fresh_fit_too():
    """Not just in the saved table. Reduced restarts, so this stays inside a
    test budget -- fewer restarts can only make the CLASSICAL models weaker,
    so an HQMM win here would still be meaningful."""
    df = E.seattle_comparison(hmm_restarts=6, hqmm_restarts=3, save=False)
    hqmm2 = df[df.model == "hqmm-d2"].iloc[0]
    rival = E.best_at_or_below(df, hqmm2.params)
    assert rival.nll <= hqmm2.nll, (
        f"hqmm-d2 {hqmm2.nll:.4f} now beats {rival.model} {rival.nll:.4f}"
    )
