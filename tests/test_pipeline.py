"""Unit tests for the cleaning and feature-engineering logic (no network access needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clean_eda import clean, fill_short_gaps          # noqa: E402
from features import (leakage_check, make_features,   # noqa: E402
                      time_split_dates)


def _grid(n_days: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_days * 48, freq="30min", tz="UTC", name="ts")


def _clean_frame(n_days: int = 40) -> pd.DataFrame:
    idx = _grid(n_days)
    rng = np.random.default_rng(0)
    t = np.arange(len(idx))
    wind = 8 + 3 * np.sin(2 * np.pi * t / 240) + rng.normal(0, 1, len(idx))
    actual = 200 - 8 * wind + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 5, len(idx))
    return pd.DataFrame({
        "actual": actual,
        "forecast": actual + rng.normal(0, 10, len(idx)),
        "gen_wind": np.clip(wind * 3, 0, None),
        "gen_gas": 50.0,
        "wind_mean": wind,
    }, index=idx)


def test_fill_short_gaps_only_fills_short_runs():
    s = pd.Series(np.arange(30, dtype=float), index=_grid(1)[:30])
    s.iloc[5:7] = np.nan           # run of 2 -> filled
    s.iloc[12:20] = np.nan         # run of 8 -> untouched
    out = fill_short_gaps(s, max_gap=4)
    assert out.iloc[5:7].notna().all()
    assert out.iloc[12:20].isna().all()


def test_clean_builds_regular_grid_and_removes_duplicates():
    idx = _grid(2)
    main = pd.DataFrame({"ts": idx, "forecast": 100.0, "actual": 100.0, "idx": "low"})
    main = main.drop(index=[10, 11])                          # two missing intervals
    main = pd.concat([main, main.iloc[[3]]])                  # one duplicate timestamp
    wx = pd.DataFrame({"ts": idx[::2], "wind_mean": 5.0})
    df, report = clean(main, wx)
    assert report["duplicate_timestamps"] == 1
    assert report["missing_intervals"] == 2
    assert df.index.is_unique and df.index.is_monotonic_increasing
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("30min")).all()


def test_out_of_bounds_values_are_removed():
    idx = _grid(2)
    main = pd.DataFrame({"ts": idx, "forecast": 100.0, "actual": 100.0, "idx": "low"})
    main.loc[20, "actual"] = 5000.0
    wx = pd.DataFrame({"ts": idx, "wind_mean": 5.0})
    _, report = clean(main, wx)
    assert report["actual_out_of_bounds"] == 1


@pytest.mark.parametrize("min_lag", [12, 96])
def test_features_do_not_use_the_future(min_lag):
    df = _clean_frame()
    feat, cols = make_features(df, min_lag)
    leakage_check(feat, df["actual"], min_lag)

    # Changing the target from row k onwards must not alter any feature before row k + min_lag.
    k = 1000
    changed = df.copy()
    changed.iloc[k:, changed.columns.get_loc("actual")] += 100
    changed.iloc[k:, changed.columns.get_loc("gen_wind")] += 10
    feat2, _ = make_features(changed, min_lag)
    pd.testing.assert_frame_equal(feat[cols].iloc[:k + min_lag], feat2[cols].iloc[:k + min_lag])


def test_time_split_is_chronological_and_non_overlapping():
    idx = _grid(400)
    s = time_split_dates(idx)
    train_end = pd.Timestamp(s["train"][1])
    val_end = pd.Timestamp(s["validation"][1])
    assert pd.Timestamp(s["train"][0]) < train_end <= pd.Timestamp(s["validation"][0])
    assert train_end < val_end <= pd.Timestamp(s["test"][0])


# ---------------------------------------------------------------- forecasting helpers
def test_forecast_metrics_and_split_labels():
    from forecast import metrics, split_labels
    y = pd.Series([100.0, 120.0, 140.0])
    m = metrics(y, y + 10)
    assert m["MAE"] == 10 and m["RMSE"] == 10 and m["bias"] == 10 and m["n"] == 3

    idx = _grid(400)
    splits = time_split_dates(idx)
    labels = split_labels(idx, splits)
    assert labels.iloc[0] == "train" and labels.iloc[-1] == "test"
    order = labels.map({"train": 0, "validation": 1, "test": 2})
    assert order.is_monotonic_increasing                    # chronological, no interleaving
