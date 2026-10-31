"""
Turns the cleaned data into a table a model can learn from.

The one rule that matters: a forecast is made at some moment (the "origin") for a time up to H
hours later, so any feature built from past intensity must be at least H hours old. Otherwise the
model gets to peek at the future and looks far better than it really is. MIN_LAG (in half-hours)
enforces this. 96 means a 48-hour horizon and 12 means 6 hours.

Calendar and weather features are fine to use for future times, because calendars and weather
forecasts are known in advance. This project uses weather that was actually observed in place of
archived weather forecasts, which flatters the results a little. The README explains this.

Writes data/processed/features.parquet, feature_list.json and split_dates.json.

    python features.py
    python features.py --min-lag 12
"""
import argparse
import json

import numpy as np
import pandas as pd
import holidays

from common import PROCESSED, ensure_dirs

BASE_LAGS = [48, 96, 144, 336, 672]     # 1d, 2d, 3d, 7d, 14d in half-hours
ROLL_WINDOWS = [48, 336]                # 1 day, 7 days
WEATHER_COLS = ["wind_mean", "wind_scotland", "wind_north_sea", "solar_south",
                "solar_mean", "temp_mean", "cloud_mean"]


def uk_holidays(years: range) -> holidays.HolidayBase:
    try:
        return holidays.UnitedKingdom(subdiv="ENG", years=years)
    except NotImplementedError:                      # older holidays versions
        return holidays.UnitedKingdom(subdiv="England", years=years)


def make_features(df: pd.DataFrame, min_lag: int) -> tuple[pd.DataFrame, list[str]]:
    y = df["actual"]
    feats: dict[str, pd.Series] = {}

    # --- 1. autoregressive features (all >= min_lag old) ---------------------------------
    lags = sorted({l for l in BASE_LAGS if l >= min_lag} | {min_lag})
    for l in lags:
        feats[f"lag_{l}"] = y.shift(l)
    base = y.shift(min_lag)                           # newest value we are allowed to see
    for w in ROLL_WINDOWS:
        feats[f"roll_mean_{w}"] = base.rolling(w, min_periods=w // 2).mean()
        feats[f"roll_std_{w}"] = base.rolling(w, min_periods=w // 2).std()
    feats["roll_min_48"] = base.rolling(48, min_periods=24).min()
    feats["roll_max_48"] = base.rolling(48, min_periods=24).max()

    # --- 2. calendar features (London local time drives human demand) -------------------
    local = df.index.tz_convert("Europe/London")
    tod = np.asarray(local.hour * 2 + local.minute // 30)          # 0..47
    dow = np.asarray(local.dayofweek)
    doy = np.asarray(local.dayofyear)
    hol = uk_holidays(range(local.year.min() - 1, local.year.max() + 2))
    dates = pd.Series(local.date, index=df.index)
    is_hol = dates.map(lambda d: d in hol).astype(int)
    feats["tod"] = pd.Series(tod, index=df.index)
    feats["dow"] = pd.Series(dow, index=df.index)
    feats["month"] = pd.Series(np.asarray(local.month), index=df.index)
    feats["is_weekend"] = pd.Series((dow >= 5).astype(int), index=df.index)
    feats["is_bank_holiday"] = is_hol
    feats["is_day_before_hol"] = is_hol.shift(-48, fill_value=0)
    feats["is_day_after_hol"] = is_hol.shift(48, fill_value=0)
    feats["tod_sin"] = pd.Series(np.sin(2 * np.pi * tod / 48), index=df.index)
    feats["tod_cos"] = pd.Series(np.cos(2 * np.pi * tod / 48), index=df.index)
    feats["dow_sin"] = pd.Series(np.sin(2 * np.pi * dow / 7), index=df.index)
    feats["dow_cos"] = pd.Series(np.cos(2 * np.pi * dow / 7), index=df.index)
    feats["doy_sin"] = pd.Series(np.sin(2 * np.pi * doy / 365.25), index=df.index)
    feats["doy_cos"] = pd.Series(np.cos(2 * np.pi * doy / 365.25), index=df.index)

    # --- 3. weather (known in advance via forecasts) -------------------------------------
    for c in WEATHER_COLS:
        if c in df:
            feats[c] = df[c]
    if "wind_mean" in df:
        feats["wind_cube"] = df["wind_mean"] ** 3                       # wind power ~ v^3
        feats["wind_roll_6h"] = df["wind_mean"].rolling(12, min_periods=6).mean()
        feats["wind_delta_3h"] = df["wind_mean"].diff(6)

    # --- 4. generation mix, ONLY as lagged values ---------------------------------------
    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    for c in gen_cols:
        feats[f"{c}_lag_{min_lag}"] = df[c].shift(min_lag)
    renew = [c for c in ("gen_wind", "gen_solar", "gen_hydro") if c in df]
    if renew:
        feats[f"renewables_lag_{min_lag}"] = df[renew].sum(axis=1, min_count=1).shift(min_lag)

    X = pd.DataFrame(feats, index=df.index)
    feature_cols = list(X.columns)

    # --- 5. target and benchmark columns (not model inputs) ------------------------------
    out = X.copy()
    out["actual"] = y
    out["official_forecast"] = df["forecast"]                           # benchmark only
    out["naive_24h"] = y.shift(48)                                      # baseline 1
    out["naive_7d"] = y.shift(336)                                      # baseline 2 (seasonal)
    return out, feature_cols


def leakage_check(feat: pd.DataFrame, y: pd.Series, min_lag: int, n: int = 300) -> None:
    """Check that lag features equal the target from at least min_lag ago."""
    rng = np.random.default_rng(0)
    idx = feat.dropna(subset=[f"lag_{min_lag}"]).index
    for ts in rng.choice(idx, size=min(n, len(idx)), replace=False):
        ts = pd.Timestamp(ts)
        past = y.loc[ts - pd.Timedelta(minutes=30 * min_lag)]
        got = feat.loc[ts, f"lag_{min_lag}"]
        assert (np.isnan(past) and np.isnan(got)) or np.isclose(past, got), f"leak at {ts}"
    # rolling features must not include anything newer than the origin
    first = feat.index[min_lag + 60]
    manual = y.loc[:first - pd.Timedelta(minutes=30 * min_lag)].tail(48).mean()
    assert np.isclose(manual, feat.loc[first, "roll_mean_48"], equal_nan=True), "rolling leak"
    print("Leakage checks passed.")


def time_split_dates(index: pd.DatetimeIndex, test_months: int = 6, val_months: int = 3) -> dict:
    end = index.max()
    test_start = (end - pd.DateOffset(months=test_months)).floor("D")
    val_start = (test_start - pd.DateOffset(months=val_months)).floor("D")
    return {"train": [str(index.min()), str(val_start)],
            "validation": [str(val_start), str(test_start)],
            "test": [str(test_start), str(end)]}


def main() -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-lag", type=int, default=96, help="half-hours; 96 = 48h horizon")
    args = ap.parse_args()

    df = pd.read_parquet(PROCESSED / "clean.parquet")
    feat, feature_cols = make_features(df, args.min_lag)
    leakage_check(feat, df["actual"], args.min_lag)

    # keep rows with a target and complete features (drops the first 14 days of warm-up
    # and the most recent days where weather archive / actuals are not yet available)
    dataset = feat.dropna(subset=feature_cols + ["actual"])
    splits = time_split_dates(dataset.index)

    dataset.to_parquet(PROCESSED / "features.parquet")
    (PROCESSED / "feature_list.json").write_text(json.dumps(
        {"min_lag_half_hours": args.min_lag, "features": feature_cols,
         "target": "actual", "benchmarks": ["official_forecast", "naive_24h", "naive_7d"]},
        indent=2))
    (PROCESSED / "split_dates.json").write_text(json.dumps(splits, indent=2))

    print(f"\n{len(feature_cols)} features, {len(dataset):,} usable rows "
          f"({dataset.index.min():%Y-%m-%d} -> {dataset.index.max():%Y-%m-%d})")
    for name, (a, b) in splits.items():
        n = ((dataset.index >= a) & (dataset.index < b)).sum()
        print(f"  {name:<11} {a[:10]} -> {b[:10]}  ({n:,} rows)")


if __name__ == "__main__":
    main()
