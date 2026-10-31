"""
Trains the forecasting models and measures how good they are.

For each horizon (6, 12, 24 and 48 hours ahead) a separate feature table is built, then two models
are trained on the oldest part of the data: a ridge regression (simple, linear) and LightGBM
(gradient-boosted trees). A validation period is used to stop LightGBM training at the right moment,
and the most recent six months are kept back as a test set the models never see.

Both models are compared with three benchmarks: the official forecast from the grid operator,
"same time yesterday" and "same time last week".

The 48-hour LightGBM model is saved. Its features for any time t only use data from at least 48
hours earlier, so a prediction for any time within 48 hours of a decision point could really have
been made at that decision point. The backtest relies on this.

Writes reports/forecast_metrics.csv, charts in reports/figures, models/lgbm_48h.joblib and
data/processed/predictions_48h.parquet.

    python forecast.py
"""
import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, MODELS, PROCESSED, REPORTS, ensure_dirs  # noqa: E402
from features import make_features, time_split_dates  # noqa: E402

HORIZONS = {"6h": 12, "12h": 24, "24h": 48, "48h": 96}     # label -> half-hours
PRODUCTION_HORIZON = "48h"

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
)


# ---------------------------------------------------------------- helpers
def metrics(y: pd.Series, pred: pd.Series) -> dict:
    err = pred - y
    return {"MAE": float(err.abs().mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "n": int(len(y))}


def split_labels(index: pd.DatetimeIndex, splits: dict) -> pd.Series:
    val_start = pd.Timestamp(splits["validation"][0])
    test_start = pd.Timestamp(splits["test"][0])
    lab = np.where(index >= test_start, "test", np.where(index >= val_start, "validation", "train"))
    return pd.Series(lab, index=index, name="split")


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    return model.fit(X_train, y_train)


def fit_lgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l1",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


# ---------------------------------------------------------------- main
def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PROCESSED / "clean.parquet")

    # split boundaries come from the longest-horizon table so every horizon uses the same periods
    ref, ref_cols = make_features(df, max(HORIZONS.values()))
    splits = time_split_dates(ref.dropna(subset=ref_cols + ["actual"]).index)
    print("Split periods:", json.dumps(splits, indent=2))

    rows = []
    for label, h in HORIZONS.items():
        feat, cols = make_features(df, h)
        data = feat.dropna(subset=cols + ["actual"])
        lab = split_labels(data.index, splits)
        tr, va, te = (data[lab == s] for s in ("train", "validation", "test"))
        print(f"\n[{label}] train={len(tr):,} val={len(va):,} test={len(te):,}")

        ridge = fit_ridge(tr[cols], tr["actual"])
        gbm = fit_lgbm(tr[cols], tr["actual"], va[cols], va["actual"])
        print(f"[{label}] LightGBM best iteration: {gbm.best_iteration_}")

        te = te.assign(ridge=ridge.predict(te[cols]), lightgbm=gbm.predict(te[cols]))
        candidates = {"lightgbm": "lightgbm", "ridge": "ridge", "official_forecast": "official_forecast",
                      "naive_7d": "naive_7d"}
        if h <= 48:                                   # 'same time yesterday' needs a 24 h-old value
            candidates["naive_24h"] = "naive_24h"
        common_rows = te.dropna(subset=list(candidates.values()) + ["actual"])
        for name, col in candidates.items():
            rows.append({"horizon": label, "model": name, **metrics(common_rows["actual"], common_rows[col])})

        if label == PRODUCTION_HORIZON:
            data = data.assign(split=lab,
                               pred_lightgbm=gbm.predict(data[cols]), pred_ridge=ridge.predict(data[cols]))
            keep = ["actual", "official_forecast", "naive_7d", "pred_lightgbm", "pred_ridge", "split"]
            data[keep].to_parquet(PROCESSED / "predictions_48h.parquet")
            joblib.dump({"model": gbm, "features": cols, "min_lag": h}, MODELS / "lgbm_48h.joblib")
            _plot_importance(gbm, cols)
            _plot_test_week(data[data["split"] == "test"])

    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "forecast_metrics.csv", index=False)
    (REPORTS / "split_periods.json").write_text(json.dumps(splits, indent=2))
    _plot_mae(result)
    print("\n" + result.round(2).to_string(index=False))


# ---------------------------------------------------------------- figures
def _plot_mae(result: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, g in result.groupby("model"):
        g = g.set_index("horizon").reindex(list(HORIZONS))
        ax.plot(g.index, g["MAE"], marker="o", label=model)
    ax.set(title="Test-period forecast error by horizon", xlabel="Forecast horizon", ylabel="MAE (gCO2/kWh)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "07_forecast_mae_by_horizon.png", dpi=150); plt.close(fig)


def _plot_test_week(test: pd.DataFrame) -> None:
    week = test.dropna(subset=["actual", "pred_lightgbm"]).iloc[: 48 * 7]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week.index, week["actual"], label="actual")
    ax.plot(week.index, week["pred_lightgbm"], label="LightGBM (48 h model)")
    ax.plot(week.index, week["official_forecast"], label="official forecast", alpha=.7)
    ax.set(title="First week of the test period", ylabel="gCO2/kWh")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIGS / "08_test_week_forecast.png", dpi=150); plt.close(fig)


def _plot_importance(model: lgb.LGBMRegressor, cols: list) -> None:
    gain = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(gain.index, gain.values)
    ax.set(title="LightGBM feature importance (gain), 48 h model")
    fig.tight_layout(); fig.savefig(FIGS / "09_feature_importance.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
