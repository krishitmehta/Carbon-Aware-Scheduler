"""Unit tests for the scheduler and the backtest."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import run_backtest, summarise                       # noqa: E402
from scheduler import (SLOT, Job, best_start, emissions_kg,         # noqa: E402
                       schedule_jobs, window_costs)


def _series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="30min", tz="UTC")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def _job(x, name="job", hours=2, kw=5.0):
    return Job.from_hours(name, hours, kw, x.index[0], x.index[-1] + SLOT)


def test_emissions_formula():
    x = _series([100.0] * 10)
    job = _job(x, hours=2, kw=5.0)
    # 4 slots x 100 g/kWh x 5 kW x 0.5 h / 1000
    assert emissions_kg(x, x.index[0], job) == pytest.approx(1.0)


def test_best_start_finds_cleanest_window():
    x = _series([200.0] * 40 + [50.0] * 8 + [200.0] * 48)
    start, kg = best_start(x, _job(x))
    assert start == x.index[40]
    assert kg == pytest.approx(0.5)


def test_window_costs_match_brute_force():
    rng = np.random.default_rng(3)
    x = _series(rng.uniform(50, 300, 60))
    job = _job(x, hours=1.5, kw=3.0)
    fast = window_costs(x, job)
    slow = pd.Series({s: emissions_kg(x, s, job) for s in fast.index})
    np.testing.assert_allclose(fast.to_numpy(), slow.to_numpy())


def test_missing_values_in_window_raise():
    x = _series([100.0] * 20)
    x.iloc[5] = np.nan
    with pytest.raises(ValueError):
        best_start(x, _job(x))


def test_capacity_constraint_is_respected():
    x = _series([200.0] * 40 + [50.0] * 8 + [200.0] * 48)
    jobs = [_job(x, "a", 2, 5.0), _job(x, "b", 2, 5.0)]
    free = schedule_jobs(jobs, x)
    assert free["a"] == free["b"]                                  # both pick the same cleanest slot
    capped = schedule_jobs(jobs, x, capacity_kw=6.0)
    assert capped["a"] != capped["b"]
    a_end = capped["a"] + 4 * SLOT
    b_end = capped["b"] + 4 * SLOT
    assert a_end <= capped["b"] or b_end <= capped["a"]            # no overlap


def test_infeasible_capacity_raises():
    x = _series([100.0] * 8)
    jobs = [Job.from_hours("a", 4, 10.0, x.index[0], x.index[-1] + SLOT),
            Job.from_hours("b", 4, 10.0, x.index[0], x.index[-1] + SLOT),
            Job.from_hours("c", 4, 10.0, x.index[0], x.index[-1] + SLOT)]
    with pytest.raises(RuntimeError):
        schedule_jobs(jobs, x, capacity_kw=15.0)


def test_backtest_oracle_is_never_beaten():
    idx = pd.date_range("2024-06-01", periods=48 * 12, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    tod = idx.hour * 2 + idx.minute // 30
    actual = 150 + 60 * np.sin(2 * np.pi * (tod - 30) / 48) + rng.normal(0, 10, len(idx))
    pred = pd.DataFrame({"actual": actual,
                         "pred_lightgbm": actual + rng.normal(0, 15, len(idx)),
                         "official_forecast": actual + rng.normal(0, 25, len(idx)),
                         "naive_7d": pd.Series(actual, index=idx).shift(336)}, index=idx).dropna()
    summary = summarise(run_backtest(pred))
    for (_, _), g in summary.groupby(["scenario", "window_h"]):
        by = g.set_index("strategy")["mean_kg"].astype(float)
        assert by["oracle"] <= by.drop("oracle").min() + 1e-9
        assert by["lightgbm"] <= by["immediate"] + 1e-6           # informative forecast should not lose here
