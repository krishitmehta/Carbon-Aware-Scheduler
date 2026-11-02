"""
Replays the scheduler on past data to see what it would have saved.

At 00:00, 06:00, 12:00 and 18:00 UTC on every day of the test period, a job is scheduled using only
forecasts that existed at that moment. Its CO2 is then worked out from the intensity that really
happened. This is repeated for three example jobs and four waiting windows.

Ways of choosing the start time
  immediate          start right away
  window_mean        the average result of picking a random start inside the window
  naive_7d           schedule using last week's pattern as the forecast
  official_forecast  schedule using the grid operator's forecast
  lightgbm           schedule using this project's LightGBM forecast
  oracle             schedule using the real intensity (perfect hindsight, so a ceiling)

Forecasts for time t only use data at least 48 hours old (see forecast.py), so a window of up to
48 hours from a decision point never relies on anything unknown at that point. Decision points whose
window has missing data are skipped.

Writes reports/backtest_summary.csv, reports/results.md and a chart in reports/figures.

    python backtest.py
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402

SLOT_HOURS = 0.5
# name: (duration in hours, power in kW)
SCENARIOS = {
    "Nightly ETL job (2 h, 5 kW)": (2, 5.0),
    "EV charge (4 h, 7 kW)": (4, 7.0),
    "ML training run (6 h, 20 kW)": (6, 20.0),
}
WINDOWS_H = [6, 12, 24, 48]
ORIGIN_HOURS = (0, 6, 12, 18)
FORECASTS = {"naive_7d": "naive_7d", "official_forecast": "official_forecast", "lightgbm": "pred_lightgbm"}
STRATEGY_ORDER = ["immediate", "window_mean", "naive_7d", "official_forecast", "lightgbm", "oracle"]


def run_backtest(pred: pd.DataFrame, scenarios=SCENARIOS, windows_h=WINDOWS_H,
                 origin_hours=ORIGIN_HOURS, forecasts=FORECASTS) -> pd.DataFrame:
    """Origin-level emissions (kg) for every strategy. `pred` needs 'actual' plus forecast columns."""
    grid = pd.date_range(pred.index.min(), pred.index.max(), freq="30min", tz="UTC")
    p = pred.reindex(grid)
    actual = p["actual"].to_numpy(float)
    fc = {name: p[col].to_numpy(float) for name, col in forecasts.items()}
    bad = np.isnan(actual)
    for v in fc.values():
        bad |= np.isnan(v)
    bad_cum = np.concatenate([[0], np.cumsum(bad)])

    def cumsum(a):
        return np.concatenate([[0.0], np.cumsum(np.nan_to_num(a))])

    cs_actual = cumsum(actual)
    cs_fc = {k: cumsum(v) for k, v in fc.items()}
    origins = [i for i, t in enumerate(grid) if t.hour in origin_hours and t.minute == 0]

    rows = []
    for scenario, (dur_h, power) in scenarios.items():
        d = int(dur_h / SLOT_HOURS)
        for win_h in windows_h:
            if win_h <= dur_h:
                continue                                  # no flexibility to exploit
            W = int(win_h / SLOT_HOURS)
            n_starts = W - d + 1
            for i in origins:
                if i + W > len(grid) or bad_cum[i + W] - bad_cum[i] > 0:
                    continue                              # incomplete window -> skipped
                scale = power * SLOT_HOURS / 1000.0
                a_cost = (cs_actual[i + d: i + W + 1] - cs_actual[i: i + n_starts]) * scale
                result = {"immediate": a_cost[0], "window_mean": a_cost.mean(), "oracle": a_cost.min()}
                for name, cs in cs_fc.items():
                    f_cost = cs[i + d: i + W + 1] - cs[i: i + n_starts]
                    result[name] = a_cost[int(np.argmin(f_cost))]
                for strategy, kg in result.items():
                    rows.append((scenario, win_h, grid[i], strategy, kg))
    return pd.DataFrame(rows, columns=["scenario", "window_h", "origin", "strategy", "kg"])


def summarise(origin_level: pd.DataFrame) -> pd.DataFrame:
    mean = (origin_level.groupby(["scenario", "window_h", "strategy"])["kg"].agg(["mean", "count"])
            .rename(columns={"mean": "mean_kg", "count": "n_origins"}).reset_index())
    ref = mean.pivot(index=["scenario", "window_h"], columns="strategy", values="mean_kg")
    out = mean.join(ref["immediate"].rename("imm"), on=["scenario", "window_h"])
    out = out.join(ref["window_mean"].rename("wmean"), on=["scenario", "window_h"])
    out = out.join(ref["oracle"].rename("orc"), on=["scenario", "window_h"])
    out["saving_vs_immediate_pct"] = 100 * (out["imm"] - out["mean_kg"]) / out["imm"]
    out["saving_vs_window_mean_pct"] = 100 * (out["wmean"] - out["mean_kg"]) / out["wmean"]
    gap = (out["imm"] - out["orc"]).replace(0, np.nan)
    out["share_of_oracle_saving_pct"] = 100 * (out["imm"] - out["mean_kg"]) / gap
    out["strategy"] = pd.Categorical(out["strategy"], STRATEGY_ORDER, ordered=True)
    return (out.drop(columns=["imm", "wmean", "orc"])
            .sort_values(["scenario", "window_h", "strategy"]).reset_index(drop=True))


# ---------------------------------------------------------------- reporting
def md_table(df: pd.DataFrame, decimals: int = 2) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n"
    sep = "|" + "|".join("---" for _ in df.columns) + "|\n"
    body = ""
    for _, r in df.iterrows():
        cells = [f"{v:.{decimals}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
        body += "| " + " | ".join(cells) + " |\n"
    return head + sep + body


def write_results(summary: pd.DataFrame, origin_level: pd.DataFrame, test_range) -> None:
    lines = ["# Results", "",
             "Written by `backtest.py` and `forecast.py`. Re-run the pipeline to refresh it.", "",
             f"Test period: {test_range[0]:%Y-%m-%d} to {test_range[1]:%Y-%m-%d}. "
             f"Origins: {', '.join(f'{h:02d}:00' for h in ORIGIN_HOURS)} UTC daily. "
             "Emissions are computed from actual intensity.", ""]
    metrics_file = REPORTS / "forecast_metrics.csv"
    if metrics_file.exists():
        m = pd.read_csv(metrics_file)[["horizon", "model", "MAE", "RMSE", "bias", "n"]]
        lines += ["## Forecast accuracy on the test period (gCO2/kWh)", "", md_table(m), ""]
    lines += ["## Scheduling backtest", "",
              "`mean_kg` is the average emissions of one job run. Savings are relative to starting immediately "
              "or to a randomly chosen start inside the window. `share_of_oracle_saving_pct` is the part of the "
              "perfect-hindsight saving that was captured.", ""]
    for scenario, g in summary.groupby("scenario"):
        n = g.groupby("window_h")["n_origins"].first().to_dict()
        lines += [f"### {scenario}", ""]
        t = g[["window_h", "strategy", "mean_kg", "saving_vs_immediate_pct",
               "saving_vs_window_mean_pct", "share_of_oracle_saving_pct"]].copy()
        t["strategy"] = t["strategy"].astype(str)
        lines += [md_table(t, 3), f"Origins evaluated per window (h): {n}", ""]
    (REPORTS / "results.md").write_text("\n".join(lines))


FIG_LABELS = {
    "naive_7d": "Copy last week's pattern",
    "official_forecast": "Grid operator's forecast (NESO)",
    "lightgbm": "My LightGBM forecast",
    "oracle": "Perfect hindsight (upper limit)",
}
FIG_COLOURS = {"naive_7d": "#9aa0a6", "official_forecast": "#3b73b9", "lightgbm": "#2e9e5b", "oracle": "#222222"}
FIG_TITLES = {
    "EV charge (4 h, 7 kW)": "Electric car charge (4 h, 7 kW)",
    "Nightly ETL job (2 h, 5 kW)": "Nightly data job (2 h, 5 kW)",
    "ML training run (6 h, 20 kW)": "Model training (6 h, 20 kW)",
}


def plot_savings(summary: pd.DataFrame) -> None:
    wanted = list(FIG_TITLES)
    scenarios = [s for s in wanted if s in set(summary["scenario"])]
    scenarios += [s for s in summary["scenario"].unique() if s not in scenarios]
    with plt.rc_context({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13}):
        fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 5.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, sc in zip(axes, scenarios):
            for strat in FIG_LABELS:
                g = summary[(summary["scenario"] == sc) & (summary["strategy"].astype(str) == strat)]
                g = g.sort_values("window_h")
                ax.plot(g["window_h"], g["saving_vs_immediate_pct"], marker="o", label=FIG_LABELS[strat],
                        color=FIG_COLOURS[strat], lw=3.6 if strat == "lightgbm" else 2.2,
                        ls="--" if strat == "oracle" else "-", zorder=3 if strat == "lightgbm" else 2)
            ax.set_title(FIG_TITLES.get(sc, sc), fontweight="bold")
            ax.set_xlabel("How long the job can wait (hours)")
            ax.set_xticks(sorted(summary.loc[summary["scenario"] == sc, "window_h"].unique()))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Less CO\u2082 than starting straight away (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=12)
        fig.suptitle("The longer a job can wait, the more CO\u2082 it saves", fontsize=18, fontweight="bold")
        fig.tight_layout(rect=(0, 0.08, 1, 0.94))
        fig.savefig(FIGS / "10_savings_by_window.png", dpi=200)
        plt.close(fig)

def main() -> None:
    ensure_dirs()
    pred = pd.read_parquet(PROCESSED / "predictions_48h.parquet")
    test = pred[pred["split"] == "test"]
    origin_level = run_backtest(test)
    summary = summarise(origin_level)
    summary.to_csv(REPORTS / "backtest_summary.csv", index=False)
    write_results(summary, origin_level, (test.index.min(), test.index.max()))
    plot_savings(summary)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
