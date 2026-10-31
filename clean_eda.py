"""
Cleans the raw data and takes a first look at it.

What happens here
  1. The raw tables are joined with SQL (intensity + generation mix + weather).
  2. The data is put on a complete half-hourly UTC grid, so every missing half-hour shows up as a
     visible gap. Duplicates, gaps and impossible values are counted.
  3. Gaps of up to two hours are filled by interpolation. Longer gaps are left empty on purpose.
  4. Odd-looking spikes are flagged but not deleted.
  5. The cleaned table, a data-quality report and six charts are saved.

Everything stays in UTC. London time is only used for the charts and calendar patterns.

    python clean_eda.py
"""
import json
from contextlib import closing

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, PROCESSED, REPORTS, connect, ensure_dirs  # noqa: E402

MAX_GAP = 4              # half-hours that may be interpolated (4 = 2 hours)
MAX_PLAUSIBLE = 600      # gCO2/kWh; GB grid has never been near this

INTENSITY_SQL = """
SELECT i.ts, i.forecast, i.actual, i.idx, g.*
FROM intensity AS i
LEFT JOIN generation AS g ON g.ts = i.ts
ORDER BY i.ts
"""

# Collapse the 4 weather points into model-friendly columns (hourly)
WEATHER_SQL = """
SELECT ts,
       AVG(wind_speed_100m)                                            AS wind_mean,
       MAX(CASE WHEN point = 'scotland'      THEN wind_speed_100m END) AS wind_scotland,
       MAX(CASE WHEN point = 'north_sea'     THEN wind_speed_100m END) AS wind_north_sea,
       MAX(CASE WHEN point = 'south_england' THEN shortwave_radiation END) AS solar_south,
       AVG(shortwave_radiation)                                        AS solar_mean,
       AVG(temperature_2m)                                             AS temp_mean,
       AVG(cloud_cover)                                                AS cloud_mean
FROM weather_hourly
GROUP BY ts
ORDER BY ts
"""


# ---------------------------------------------------------------- load
def load_from_sql() -> tuple[pd.DataFrame, pd.DataFrame]:
    with closing(connect()) as con:
        main = pd.read_sql(INTENSITY_SQL, con)
        wx = pd.read_sql(WEATHER_SQL, con)
    main = main.loc[:, ~main.columns.duplicated()]          # g.* repeats 'ts'
    for d in (main, wx):
        d["ts"] = pd.to_datetime(d["ts"], utc=True)
    return main, wx


# ---------------------------------------------------------------- clean
def fill_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Time-interpolate NaN runs of length <= max_gap; longer runs are left untouched.
    (DataFrame.interpolate(limit=...) would fill the first rows of a long gap, so it is not used.)"""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")          # length of each NaN run, 0 otherwise
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~(isna & (run_len <= max_gap)), filled)


def clean(main: pd.DataFrame, wx: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rep: dict = {}
    main = main.set_index("ts").sort_index()
    rep["rows_raw"] = int(len(main))
    rep["duplicate_timestamps"] = int(main.index.duplicated().sum())
    main = main[~main.index.duplicated(keep="last")]

    # complete 30-min UTC grid -> missing intervals become explicit NaN rows
    full = pd.date_range(main.index.min(), main.index.max(), freq="30min", tz="UTC")
    rep["missing_intervals"] = int(len(full) - len(main))
    main = main.reindex(full)
    main.index.name = "ts"

    # physically implausible values -> NaN
    for c in ("forecast", "actual"):
        bad = (main[c] < 0) | (main[c] > MAX_PLAUSIBLE)
        rep[f"{c}_out_of_bounds"] = int(bad.sum())
        main.loc[bad, c] = np.nan

    # spike flag: deviation from rolling median > 6 x rolling MAD (min threshold 6 g/kWh)
    med = main["actual"].rolling(48, center=True, min_periods=24).median()
    mad = (main["actual"] - med).abs().rolling(48, center=True, min_periods=24).median()
    main["spike_flag"] = ((main["actual"] - med).abs() > 6 * mad.clip(lower=1)).astype(int)
    rep["spikes_flagged"] = int(main["spike_flag"].sum())

    # interpolate short gaps only
    gen_cols = [c for c in main.columns if c.startswith("gen_")]
    was_na = main["actual"].isna()
    for c in ["forecast", "actual", *gen_cols]:
        main[c] = fill_short_gaps(main[c], MAX_GAP)
    main["actual_imputed"] = (was_na & main["actual"].notna()).astype(int)
    rep["actual_imputed"] = int(main["actual_imputed"].sum())
    rep["actual_still_missing"] = int(main["actual"].isna().sum())

    # weather: hourly -> half-hourly on the same grid
    wx = wx.set_index("ts").sort_index()
    wx = wx[~wx.index.duplicated(keep="last")].reindex(full)
    wx = wx.apply(lambda col: fill_short_gaps(col, 2))
    wx.index.name = "ts"
    df = main.join(wx)
    rep["weather_missing_rows"] = int(df["wind_mean"].isna().sum()) if "wind_mean" in df else None

    df["official_error"] = df["forecast"] - df["actual"]
    rep["date_range"] = [str(df.index.min()), str(df.index.max())]
    rep["rows_clean"] = int(len(df))
    return df, rep


# ---------------------------------------------------------------- EDA
def _season(month: pd.Index) -> pd.Series:
    mapping = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
               6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    return pd.Series(month, index=month).map(mapping)


def eda(df: pd.DataFrame) -> dict:
    stats: dict = {}
    local = df.index.tz_convert("Europe/London")
    d = df.copy()
    d["hour"] = local.hour + local.minute / 60
    d["dow"] = local.dayofweek
    d["season"] = _season(pd.Index(local.month)).values

    # 1. long-term trend (monthly mean)
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly = d["actual"].resample("MS").mean()
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set(title="GB grid carbon intensity - monthly mean", ylabel="gCO2/kWh")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "01_monthly_trend.png", dpi=150); plt.close(fig)

    # 2. daily profile by season
    fig, ax = plt.subplots(figsize=(10, 4))
    for s, g in d.groupby("season"):
        prof = g.groupby("hour")["actual"].mean()
        ax.plot(prof.index, prof.values, label=s)
    ax.set(title="Average daily profile by season (London time)", xlabel="Hour of day",
           ylabel="gCO2/kWh")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "02_daily_profile_season.png", dpi=150); plt.close(fig)

    # 3. weekday x hour heatmap
    heat = d.assign(h=np.floor(d["hour"]).astype(int)).pivot_table(
        index="dow", columns="h", values="actual", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_yticks(range(7)); ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set(title="Mean intensity: weekday x hour (London time)", xlabel="Hour")
    fig.colorbar(im, label="gCO2/kWh")
    fig.tight_layout(); fig.savefig(FIGS / "03_weekday_hour_heatmap.png", dpi=150); plt.close(fig)

    # 4. wind vs intensity
    if "wind_mean" in d and d["wind_mean"].notna().any():
        s = d[["wind_mean", "actual"]].dropna()
        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(s["wind_mean"], s["actual"], gridsize=40, cmap="viridis", mincnt=1)
        ax.set(title="Wind speed vs carbon intensity", xlabel="Mean 100m wind (m/s)",
               ylabel="gCO2/kWh")
        fig.colorbar(hb, label="count")
        fig.tight_layout(); fig.savefig(FIGS / "04_wind_vs_intensity.png", dpi=150); plt.close(fig)
        stats["corr_wind_intensity"] = round(float(s["wind_mean"].corr(s["actual"])), 3)

    # 5. correlation with generation mix
    gen_cols = [c for c in d.columns if c.startswith("gen_")]
    if gen_cols:
        corr = d[gen_cols + ["actual"]].corr()["actual"].drop("actual").sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(corr.index, corr.values)
        ax.set(title="Correlation of generation share with intensity")
        ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIGS / "05_generation_corr.png", dpi=150); plt.close(fig)
        stats["generation_corr"] = corr.round(3).to_dict()

    # 6. official forecast error
    err = d["official_error"].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err.clip(-100, 100), bins=80)
    ax.set(title="Official forecast error (forecast - actual)", xlabel="gCO2/kWh")
    fig.tight_layout(); fig.savefig(FIGS / "06_official_forecast_error.png", dpi=150); plt.close(fig)
    stats["official_forecast"] = {
        "MAE": round(float(err.abs().mean()), 2),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 2),
        "bias": round(float(err.mean()), 2),
    }

    stats["mean_intensity"] = round(float(d["actual"].mean()), 1)
    stats["by_season"] = d.groupby("season")["actual"].mean().round(1).to_dict()
    return stats


# ---------------------------------------------------------------- main
def main() -> None:
    ensure_dirs()
    main_df, wx = load_from_sql()
    df, report = clean(main_df, wx)
    report["eda"] = eda(df)

    df.to_parquet(PROCESSED / "clean.parquet")
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=2, default=str))

    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {PROCESSED / 'clean.parquet'} and figures in {FIGS}")


if __name__ == "__main__":
    main()
