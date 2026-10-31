"""
Settings and small helpers shared by every script in the project.

Folders are only created when a script that writes files calls ensure_dirs(), so importing this
module (for example from the dashboard) never touches the disk.
"""
import sqlite3
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- folders
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = RAW / "chunks"          # one JSON file per API request window, so downloads can resume
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
DB_PATH = DATA / "carbon.db"


def ensure_dirs() -> None:
    for folder in (RAW, CACHE, PROCESSED, MODELS, FIGS):
        folder.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- settings
START = "2023-01-01"            # first day to download
WINDOW_DAYS = 13                # the API allows about 14 days per request, so stay just under
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"   # timestamps are stored in SQLite as UTC text

# Four fixed weather points: windy places for wind generation, southern England for sunshine.
WEATHER_POINTS = {
    "scotland": (56.5, -4.0),
    "north_sea": (53.8, 1.0),
    "south_england": (51.0, -1.0),
    "wales": (52.3, -3.8),
}
WEATHER_VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"]


# ---------------------------------------------------------------- SQLite
def connect() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _ts_to_str(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.strftime(TS_FMT)
    return df


def save_to_sqlite(intensity: pd.DataFrame, generation: pd.DataFrame, weather: pd.DataFrame) -> None:
    """Write the three raw tables to SQLite. Timestamps go in as UTC text, which sorts correctly
    and avoids any daylight-saving confusion."""
    con = connect()
    try:
        _ts_to_str(intensity).to_sql("intensity", con, if_exists="replace", index=False)
        _ts_to_str(generation).to_sql("generation", con, if_exists="replace", index=False)
        _ts_to_str(weather).to_sql("weather_hourly", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS ix_intensity_ts ON intensity(ts)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_generation_ts ON generation(ts)")
        con.execute("CREATE INDEX IF NOT EXISTS ix_weather_ts ON weather_hourly(ts, point)")
        con.commit()
    finally:
        con.close()
