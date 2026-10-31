"""
Dashboard for the carbon-aware scheduler.

Run the pipeline first (or use the files already in the repository), then:

    streamlit run app.py

Everything in here is a replay of past data. Nothing is live.
"""
import datetime as dt
import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import PROCESSED, REPORTS
from scheduler import Job, best_start, emissions_kg, window_costs

LONDON = "Europe/London"

st.set_page_config(page_title="Carbon-aware scheduler", layout="wide")

# ---------------------------------------------------------------- plain-English names
FORECAST_SOURCES = {
    "This project's LightGBM forecast": "pred_lightgbm",
    "The grid operator's own forecast (NESO)": "official_forecast",
    "Simply copy last week's pattern": "naive_7d",
}

JOB_PRESETS = {
    "Charging an electric car": (4.0, 7.0,
                                 "A home charger delivers about 7 kW. Four hours of charging adds roughly 28 kWh."),
    "A nightly data job on a small server": (2.0, 5.0,
                                              "The kind of batch job a data team runs every night, such as "
                                              "refreshing reports."),
    "Training a machine-learning model": (6.0, 20.0,
                                           "A GPU machine working for six hours. Uses far more power than a "
                                           "car charger."),
    "My own job": (2.0, 5.0, "Set the length and power yourself."),
}

SCENARIO_NAMES = {
    "Nightly ETL job (2 h, 5 kW)": "Nightly data job (2 hours, 5 kW)",
    "EV charge (4 h, 7 kW)": "Electric car charge (4 hours, 7 kW)",
    "ML training run (6 h, 20 kW)": "Machine-learning training (6 hours, 20 kW)",
}

STRATEGY_NAMES = {
    "immediate": "Start straight away",
    "window_mean": "Start at a random time in the window",
    "naive_7d": "Copy last week's pattern",
    "official_forecast": "Grid operator's forecast (NESO)",
    "lightgbm": "This project's LightGBM forecast",
    "oracle": "Perfect hindsight (the best possible)",
}

MODEL_NAMES = {
    "lightgbm": "LightGBM (this project)",
    "ridge": "Ridge regression (simple model)",
    "official_forecast": "Grid operator's forecast (NESO)",
    "naive_24h": "Same time yesterday",
    "naive_7d": "Same time last week",
}

HORIZON_NAMES = {"6h": "6 hours ahead", "12h": "12 hours ahead", "24h": "24 hours ahead", "48h": "48 hours ahead"}

GREEN, RED, GREY = "#2e9e5b", "#d6453d", "#8a8f98"


# ---------------------------------------------------------------- loading
@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    path = PROCESSED / "predictions_48h.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df[df["split"] == "test"]


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def uk_time(ts: pd.Timestamp) -> pd.Timestamp:
    """Timezone-aware timestamp -> naive UK clock time, so charts show the time on a UK wall clock."""
    return ts.tz_convert(LONDON).tz_localize(None)


def uk_series(s: pd.Series) -> pd.Series:
    return s.set_axis(s.index.tz_convert(LONDON).tz_localize(None))


def nice_time(ts: pd.Timestamp) -> str:
    return uk_time(ts).strftime("%A %d %B, %H:%M")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("New here?")
    st.markdown(
        "1. Read the short intro at the top of the page.\n"
        "2. Open the **Try it yourself** tab and press **Show me a random day**.\n"
        "3. Look at the green and red boxes on the chart. That is the whole idea."
    )
    with st.expander("Plain-English glossary", expanded=False):
        st.markdown(
            "**Carbon intensity** is how much CO₂ is released to make one unit of electricity. It is "
            "measured in grams of CO₂ per kWh. A low number means cleaner power.\n\n"
            "**kWh** is a unit of energy. A 7 kW car charger running for 4 hours uses 7 x 4 = 28 kWh.\n\n"
            "**kg CO₂** is the weight of carbon dioxide a job causes. It is the job's energy (kWh) times "
            "the carbon intensity at that time.\n\n"
            "**Forecast** is a prediction of carbon intensity for the coming hours.\n\n"
            "**Perfect hindsight** means choosing the start time after already knowing what happened. "
            "Nobody can do this in real life, but it shows the best possible result.\n\n"
            "**Window** is how long a job is allowed to wait before it must be finished."
        )
    st.caption(
        "Contains data from the NESO Carbon Intensity API (CC BY 4.0). "
        "[Weather data by Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0)."
    )

# ---------------------------------------------------------------- data check
pred = load_predictions()
if pred is None:
    st.error(
        "The data this dashboard needs is missing. It is created by the project's scripts. "
        "Run them in this order, then reload the page: ingest.py, clean_eda.py, features.py, "
        "forecast.py, backtest.py."
    )
    st.stop()

first, last = pred.index.min(), pred.index.max()

# ---------------------------------------------------------------- header and intro
st.title("Carbon-aware scheduler")
st.subheader("Run energy-hungry jobs when Britain's electricity is at its cleanest")

st.markdown(
    "Electricity in Britain is not equally clean all day. When it is windy or sunny, a lot of power comes "
    "from renewables and each unit of electricity causes very little CO₂. On a calm evening, more comes "
    "from gas and the same unit causes much more.\n\n"
    "Plenty of jobs do not have to start right now: charging a car overnight, running a nightly data job, "
    "training a model. If they can wait a few hours, they can wait for a cleaner moment. "
    "**This dashboard shows how much CO₂ that saves, using real past data.**"
)

c1, c2, c3 = st.columns(3)
c1.markdown("**1. Pick a job**  \nHow long it runs and how much power it uses.")
c2.markdown("**2. Pick a day and a deadline**  \nWhen you would press start, and how long you can wait.")
c3.markdown("**3. Compare**  \nStarting straight away versus starting at the time the scheduler picks.")

st.info(
    f"Everything here is a **replay of the past**, from {uk_time(first):%d %B %Y} to {uk_time(last):%d %B %Y}. "
    "The forecasting model never saw this period while it was learning, so the results are a fair test. "
    "Nothing on this page is live, and no real devices are being controlled."
)

tab_try, tab_save, tab_acc, tab_about = st.tabs(
    ["Try it yourself", "How much does it save?", "How accurate is the forecast?", "About this project"]
)

# ================================================================ TAB 1: try it
with tab_try:
    st.markdown("### Step 1. What kind of job is it?")
    preset = st.radio("Choose a job", list(JOB_PRESETS), horizontal=True, label_visibility="collapsed")
    preset_hours, preset_kw, preset_note = JOB_PRESETS[preset]
    st.caption(preset_note)

    s1, s2 = st.columns(2)
    duration_h = s1.slider("How long does it run? (hours)", 0.5, 12.0, preset_hours, 0.5, key=f"dur_{preset}")
    power_kw = s2.number_input("How much power does it use? (kW)", 0.1, 500.0, preset_kw, 0.5, key=f"kw_{preset}")
    st.caption(f"That is {duration_h * power_kw:,.1f} kWh of electricity in total ({duration_h:g} hours x {power_kw:g} kW).")

    st.markdown("### Step 2. When would you start, and how long can it wait?")
    window_min = duration_h + 0.5
    window_h = st.slider("The job must be finished within (hours from the start time)", float(window_min), 48.0,
                         max(24.0, float(window_min)), 0.5, key=f"win_{preset}",
                         help="A longer window gives the scheduler more chances to find a clean spell.")

    # the latest day that still leaves a full window inside the data
    latest_start = (last - pd.Timedelta(hours=window_h)).tz_convert(LONDON).date()
    earliest_start = first.tz_convert(LONDON).date()
    if "day" not in st.session_state or not (earliest_start <= st.session_state["day"] <= latest_start):
        st.session_state["day"] = min(latest_start, earliest_start + dt.timedelta(days=14))

    def pick_random_day() -> None:
        span = (latest_start - earliest_start).days
        st.session_state["day"] = earliest_start + dt.timedelta(days=random.randint(0, max(span, 0)))
        st.session_state["hour"] = random.choice(["06:00", "12:00", "18:00", "22:00"])

    st.session_state.setdefault("hour", "18:00")
    d1, d2, d3 = st.columns([2, 2, 2])
    day = d1.date_input("Day (UK time)", key="day", min_value=earliest_start, max_value=latest_start)
    hour = d2.selectbox("Time you would press start (UK time)", [f"{h:02d}:00" for h in range(24)],
                        key="hour")
    d3.markdown("&nbsp;")
    d3.button("Show me a random day", on_click=pick_random_day)

    st.markdown("### Step 3. Which forecast should the scheduler trust?")
    source_label = st.selectbox("Forecast", list(FORECAST_SOURCES), label_visibility="collapsed",
                                help="The scheduler uses the forecast to guess which hours will be cleanest.")
    source = FORECAST_SOURCES[source_label]

    st.markdown("---")
    st.markdown("### What happens")

    origin = (pd.Timestamp(f"{day} {hour}").tz_localize(LONDON, ambiguous=True, nonexistent="shift_forward")
              .tz_convert("UTC"))
    job = Job.from_hours("job", duration_h, power_kw, origin, origin + pd.Timedelta(hours=window_h))
    forecast_series, actual_series = pred[source], pred["actual"]

    try:
        start, _ = best_start(forecast_series, job)
        oracle_start, _ = best_start(actual_series, job)
        kg_now = emissions_kg(actual_series, origin, job)
        kg_chosen = emissions_kg(actual_series, start, job)
        kg_oracle = emissions_kg(actual_series, oracle_start, job)
        kg_random = float(window_costs(actual_series, job).mean())
    except (ValueError, KeyError):
        st.warning("There is missing data around that time, so it cannot be replayed. "
                   "Try another day, or press **Show me a random day**.")
    else:
        saved = kg_now - kg_chosen
        saved_pct = 100 * saved / kg_now if kg_now > 0 else 0.0
        possible = kg_now - kg_oracle

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("If you start straight away", f"{kg_now:.2f} kg CO₂")
        m2.metric("If the scheduler picks the time", f"{kg_chosen:.2f} kg CO₂",
                  f"{-saved_pct:.1f}%", delta_color="inverse")
        m3.metric("CO₂ avoided", f"{saved:.2f} kg")
        m4.metric("Best possible (hindsight)", f"{kg_oracle:.2f} kg CO₂")

        if start == origin:
            st.markdown(
                f"The scheduler looked at every possible start time between **{nice_time(origin)}** and the "
                f"deadline and decided that **starting straight away** looked best, so nothing changes in this case."
            )
        else:
            st.markdown(
                f"If you press start at **{nice_time(origin)}**, the job releases **{kg_now:.2f} kg of CO₂**. "
                f"Instead, the scheduler waits and starts at **{nice_time(start)}**. By the time it finishes, "
                f"it has released **{kg_chosen:.2f} kg**."
            )
            if saved > 0:
                msg = f"That is **{saved:.2f} kg less ({saved_pct:.1f}%)**."
                if possible > 0:
                    msg += (f" Knowing the future, the best start would have been **{nice_time(oracle_start)}** "
                            f"({kg_oracle:.2f} kg), so the scheduler got **{100 * saved / possible:.0f}%** of the "
                            "best possible saving.")
                st.markdown(msg)
            else:
                st.markdown(
                    "This time the forecast pointed to a slightly dirtier moment, so waiting did not help "
                    f"({-saved:.2f} kg more). That happens on some days. What matters is the average over many "
                    "days, which you can see in the next tab."
                )

        # ---- chart
        lo, hi = origin - pd.Timedelta(hours=3), origin + pd.Timedelta(hours=window_h + 3)
        a_view, f_view = uk_series(actual_series.loc[lo:hi]), uk_series(forecast_series.loc[lo:hi])
        w0, w1 = uk_time(origin), uk_time(origin + pd.Timedelta(hours=window_h))
        i0, i1 = w0, uk_time(origin + pd.Timedelta(hours=duration_h))
        r0, r1 = uk_time(start), uk_time(start + pd.Timedelta(hours=duration_h))

        fig = go.Figure()
        fig.add_vrect(x0=w0, x1=w1, fillcolor=GREY, opacity=0.12, line_width=0)
        fig.add_vrect(x0=i0, x1=i1, fillcolor=RED, opacity=0.25, line_width=0)
        fig.add_vrect(x0=r0, x1=r1, fillcolor=GREEN, opacity=0.35, line_width=0)
        fig.add_trace(go.Scatter(x=a_view.index, y=a_view.values, name="What the grid was really like",
                                 line=dict(width=2.5)))
        fig.add_trace(go.Scatter(x=f_view.index, y=f_view.values, name=f"Forecast used: {source_label}",
                                 line=dict(dash="dot", width=2)))
        for label, colour in (("Time you are allowed to use", GREY), ("Job if started straight away", RED),
                              ("Job at the scheduled time", GREEN)):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                     marker=dict(size=12, color=colour, symbol="square")))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=20, b=10),
                          xaxis_title="Time (UK clock)",
                          yaxis_title="Carbon intensity (grams of CO₂ per kWh)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "How to read it: the higher the line, the dirtier the electricity. The grey area is the time the job "
            "may use. The red box is where it would run if started straight away. The green box is where the "
            "scheduler puts it, ideally in a dip. "
            f"Picking a start time at random inside the window would release {kg_random:.2f} kg on average."
        )

# ================================================================ TAB 2: savings
with tab_save:
    summary = load_csv("backtest_summary.csv")
    if summary is None:
        st.info("The results file is missing. Run backtest.py to create it.")
    else:
        st.markdown(
            "One lucky day proves nothing, so the decision was replayed **hundreds of times**: four times a "
            "day (at 00:00, 06:00, 12:00 and 18:00 UTC) for the whole test period, for three example jobs and "
            "four different waiting times. Each time, the scheduler only used forecasts that existed at that "
            "moment. The CO₂ was then worked out from what really happened."
        )
        summary = summary.assign(job=summary["scenario"].map(lambda s: SCENARIO_NAMES.get(s, s)))
        job_choice = st.selectbox("Job", sorted(summary["job"].unique()))
        sub = summary[summary["job"] == job_choice]
        windows = sorted(sub["window_h"].unique())
        win_choice = st.select_slider("The job can wait up to (hours)", options=windows,
                                      value=24 if 24 in windows else windows[-1])
        row = sub[sub["window_h"] == win_choice].set_index("strategy")
        n_runs = int(row["n_origins"].iloc[0])

        h1, h2, h3 = st.columns(3)
        h1.metric("CO₂ saved: this project's forecast",
                  f"{row.loc['lightgbm', 'saving_vs_immediate_pct']:.1f}%",
                  help="Compared with starting the job straight away.")
        h2.metric("CO₂ saved: grid operator's forecast",
                  f"{row.loc['official_forecast', 'saving_vs_immediate_pct']:.1f}%")
        h3.metric("Upper limit (perfect hindsight)", f"{row.loc['oracle', 'saving_vs_immediate_pct']:.1f}%")
        st.markdown(
            f"On average, one run of this job released **{row.loc['immediate', 'mean_kg']:.2f} kg of CO₂** when "
            f"started straight away and **{row.loc['lightgbm', 'mean_kg']:.2f} kg** when scheduled with this "
            f"project's forecast. That is "
            f"**{row.loc['lightgbm', 'share_of_oracle_saving_pct']:.0f}%** of the saving that perfect hindsight "
            f"would have given. Based on {n_runs} replayed decisions."
        )

        bars = row.reset_index()
        bars["strategy"] = pd.Categorical(bars["strategy"], list(STRATEGY_NAMES), ordered=True)
        bars = bars.sort_values("strategy")
        fig = go.Figure(go.Bar(
            y=[STRATEGY_NAMES[s] for s in bars["strategy"]], x=bars["saving_vs_immediate_pct"], orientation="h",
            text=[f"{v:.1f}%" for v in bars["saving_vs_immediate_pct"]], textposition="outside",
            marker_color=[GREEN if s == "lightgbm" else GREY for s in bars["strategy"]]))
        fig.update_layout(height=340, margin=dict(l=10, r=40, t=20, b=10),
                          xaxis_title="Less CO₂ than starting straight away (%)",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### What happens as the job is allowed to wait longer?")
        fig2 = go.Figure()
        for strat in ["naive_7d", "official_forecast", "lightgbm", "oracle"]:
            g = sub[sub["strategy"] == strat].sort_values("window_h")
            fig2.add_trace(go.Scatter(x=g["window_h"], y=g["saving_vs_immediate_pct"], mode="lines+markers",
                                      name=STRATEGY_NAMES[strat]))
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                           xaxis_title="How long the job can wait (hours)",
                           yaxis_title="Less CO₂ than starting straight away (%)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig2, width="stretch")
        st.caption("More flexibility means more chances to catch a windy or sunny spell, so the saving grows.")

        with st.expander("What do the six ways of choosing a start time mean?"):
            st.markdown(
                "- **Start straight away:** the baseline. No scheduling at all.\n"
                "- **Start at a random time in the window:** what you would get, on average, by picking any "
                "moment without looking at a forecast. It is a fair 'no cleverness' comparison.\n"
                "- **Copy last week's pattern:** assumes this week will look like last week.\n"
                "- **Grid operator's forecast (NESO):** the official forecast published with the data.\n"
                "- **This project's LightGBM forecast:** the machine-learning model built here.\n"
                "- **Perfect hindsight:** picks the best time knowing what really happened. Nobody can do "
                "this live. It is the ceiling."
            )
        with st.expander("Show the full numbers"):
            table = sub[["window_h", "strategy", "mean_kg", "saving_vs_immediate_pct",
                         "saving_vs_window_mean_pct", "share_of_oracle_saving_pct"]].copy()
            table["strategy"] = table["strategy"].map(STRATEGY_NAMES)
            table.columns = ["Can wait (hours)", "Way of choosing the start", "Average kg CO₂ per run",
                             "Saved vs starting straight away (%)", "Saved vs a random start (%)",
                             "Share of the best possible saving (%)"]
            st.dataframe(table.round(2), width="stretch", hide_index=True)

# ================================================================ TAB 3: accuracy
with tab_acc:
    metrics = load_csv("forecast_metrics.csv")
    if metrics is None:
        st.info("The accuracy file is missing. Run forecast.py to create it.")
    else:
        st.markdown(
            "The scheduler is only as good as its forecast. To check the forecast, it was asked to predict "
            "carbon intensity for the test period and compared with what really happened. "
            "The score used is the **average error**: on average, by how many grams of CO₂ per kWh the "
            "prediction was off. **Lower is better.**"
        )
        horizon = st.select_slider("How far ahead is the forecast looking?", options=list(HORIZON_NAMES),
                                   value="24h", format_func=lambda h: HORIZON_NAMES[h])
        sub = metrics[metrics["horizon"] == horizon].copy()
        sub["name"] = sub["model"].map(MODEL_NAMES)
        sub = sub.sort_values("MAE")
        fig = go.Figure(go.Bar(
            y=sub["name"], x=sub["MAE"], orientation="h", text=[f"{v:.1f}" for v in sub["MAE"]],
            textposition="outside",
            marker_color=[GREEN if m == "lightgbm" else GREY for m in sub["model"]]))
        fig.update_layout(height=320, margin=dict(l=10, r=40, t=20, b=10),
                          xaxis_title="Average error (grams of CO₂ per kWh). Lower is better.",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

        by = sub.set_index("model")["MAE"]
        if {"lightgbm", "official_forecast"} <= set(by.index):
            st.markdown(
                f"At **{HORIZON_NAMES[horizon]}**, this project's model is off by about **{by['lightgbm']:.1f}** "
                f"on average. The grid operator's own forecast is off by **{by['official_forecast']:.1f}**. "
                "So the official forecast is more accurate. It is built by NESO with its own modelling and "
                "system data, which this project does not have. The gap matters less than it looks, because "
                "the scheduler only needs to rank the hours (is the dip at 3am or 5am?), not predict each "
                "value exactly. Check the previous tab to see how close the saved CO₂ ends up."
            )

        st.markdown("#### Accuracy at every horizon")
        fig2 = go.Figure()
        for model, g in metrics.groupby("model"):
            g = g.set_index("horizon").reindex(list(HORIZON_NAMES)).reset_index()
            fig2.add_trace(go.Scatter(x=[HORIZON_NAMES[h] for h in g["horizon"]], y=g["MAE"],
                                      mode="lines+markers", name=MODEL_NAMES.get(model, model)))
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10),
                           yaxis_title="Average error (g CO₂ per kWh)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(fig2, width="stretch")
        st.caption(
            "The grid operator's forecast and the two 'same time' guesses do not depend on how far ahead you "
            "look, so they are flat lines. 'Same time yesterday' is only available up to 24 hours ahead."
        )
        st.warning(
            "One thing flatters this model: it was given the weather that actually happened, not a weather "
            "forecast. That is why its error barely grows with distance. In real use, the longer horizons "
            "would be less accurate."
        )
        with st.expander("Show the full numbers"):
            table = metrics.copy()
            table["model"] = table["model"].map(MODEL_NAMES)
            table["horizon"] = table["horizon"].map(HORIZON_NAMES)
            table.columns = ["Looking ahead", "Model", "Average error (MAE)", "Larger errors weigh more (RMSE)",
                             "Leans too high (+) or too low (-)", "Half-hours tested"]
            st.dataframe(table.round(2), width="stretch", hide_index=True)

# ================================================================ TAB 4: about
with tab_about:
    st.markdown(
        """
### What this project is
A data science project that forecasts the carbon intensity of electricity in Great Britain and uses the forecast to
choose the cleanest time to run flexible jobs. It was built as a portfolio project to show the full path from raw
data to a working tool: downloading data, cleaning it, building features, training and testing models, scheduling,
testing the whole idea on past data, and presenting it.

### Where the data comes from
- **Carbon intensity and generation mix:** the [Carbon Intensity API](https://carbonintensity.org.uk) from the
  National Energy System Operator (NESO), licensed CC BY 4.0. Half-hourly values for Great Britain.
- **Weather:** [Open-Meteo](https://open-meteo.com/), CC BY 4.0. Hourly wind, sunshine, temperature and cloud cover at
  four points around Britain.

### How it works, in five steps
1. **Collect** half-hourly grid data and weather for several years.
2. **Clean** it: fill short gaps, flag odd values, put everything on a regular clock.
3. **Forecast** carbon intensity up to 48 hours ahead with a LightGBM model, tested against simple guesses and
   against the grid operator's own forecast.
4. **Schedule**: for each job, work out the CO₂ of every possible start time and pick the lowest.
5. **Replay** hundreds of past decisions to measure how much CO₂ the approach would really have saved.

### What to keep in mind
- This is a **replay of past data**, not a live service. Nothing here controls a real device.
- The three example jobs (car charge, data job, model training) are **illustrative**. They are not measured from
  real equipment.
- The model was given the weather that **actually happened**, not archived weather forecasts. Real-world accuracy
  would be somewhat lower, especially for longer horizons.
- The numbers are about **average** grid intensity, not the exact CO₂ caused by one extra job.
- The test period covers roughly April to October, so winter is not part of the results.
- The grid operator's forecast is **more accurate** than this project's model. The scheduling results are close
  because the scheduler only needs to rank the hours.

### Want to use this with real data?
The next step is testing on real private workloads, such as building or EV-charging data, or compute-cluster logs.
If you have data like that and would like to collaborate, get in touch through the repository.
"""
    )
