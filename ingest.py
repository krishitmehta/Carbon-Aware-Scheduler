"""
Downloads the raw data for the project.

What it pulls
  - half-hourly carbon intensity for Great Britain (the grid operator's forecast and the actual value)
  - the generation mix: what share of the power came from wind, gas, nuclear and so on
  - hourly weather at four points around Britain, from Open-Meteo

The Carbon Intensity API returns about 14 days per request, so the history is fetched in 13-day
chunks. Every chunk is saved as JSON in data/raw/chunks. If the script stops half way, run it
again and it carries on where it left off. The last three days are always fetched again, because
the "actual" values are filled in a few hours late.

Writes data/raw/*.parquet and data/carbon.db (SQLite).

    python ingest.py
    python ingest.py --start 2024-01-01
    python ingest.py --skip-weather
"""
import argparse
import json
import time

import pandas as pd
import requests

from common import (CACHE, RAW, START, WEATHER_POINTS, WEATHER_VARS, WINDOW_DAYS,
                    save_to_sqlite, ensure_dirs)

API = "https://api.carbonintensity.org.uk"
API_FMT = "%Y-%m-%dT%H:%MZ"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_LAG_DAYS = 6            # Open-Meteo archive trails real time by several days

session = requests.Session()
session.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------- helpers
def get_json(url: str, params: dict | None = None, retries: int = 4, timeout: int = 60) -> dict:
    """GET with exponential back-off."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            wait = 2 ** attempt
            print(f"   ! {exc} -> retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def windows(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


# ---------------------------------------------------------------- Carbon Intensity API
def download_api(endpoint: str, name: str, start: pd.Timestamp, end: pd.Timestamp,
                 refresh_days: int = 3, pause: float = 0.4) -> list:
    """Download an endpoint window by window, caching each window to disk."""
    records: list = []
    recent_cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=refresh_days)
    wins = list(windows(start, end, WINDOW_DAYS))
    for i, (a, b) in enumerate(wins, 1):
        cache_file = CACHE / f"{name}_{a:%Y%m%d}_{b:%Y%m%d}.json"
        if cache_file.exists() and b < recent_cutoff:
            data = json.loads(cache_file.read_text())
        else:
            url = f"{API}{endpoint}/{a.strftime(API_FMT)}/{b.strftime(API_FMT)}"
            data = get_json(url)["data"]
            cache_file.write_text(json.dumps(data))
            time.sleep(pause)                     # be polite to a free API
        records.extend(data)
        if i % 10 == 0 or i == len(wins):
            print(f"   [{name}] {i}/{len(wins)} windows done")
    return records


def intensity_frame(records: list) -> pd.DataFrame:
    df = pd.json_normalize(records).rename(columns={
        "from": "ts",
        "intensity.forecast": "forecast",
        "intensity.actual": "actual",
        "intensity.index": "idx",
    })
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[["ts", "forecast", "actual", "idx"]]
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


def generation_frame(records: list) -> pd.DataFrame:
    rows = [{"ts": r["from"], **{f"gen_{m['fuel']}": m["perc"] for m in r["generationmix"]}}
            for r in records]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------- Open-Meteo
def download_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One request per point per calendar year (Open-Meteo handles long ranges)."""
    weather_end = end - pd.Timedelta(days=ARCHIVE_LAG_DAYS)
    frames = []
    for point, (lat, lon) in WEATHER_POINTS.items():
        for year in range(start.year, weather_end.year + 1):
            a = max(start, pd.Timestamp(year=year, month=1, day=1, tz="UTC"))
            b = min(weather_end, pd.Timestamp(year=year, month=12, day=31, tz="UTC"))
            if a > b:
                continue
            cache_file = CACHE / f"weather_{point}_{year}.json"
            if cache_file.exists() and year < weather_end.year:
                data = json.loads(cache_file.read_text())
            else:
                params = dict(latitude=lat, longitude=lon,
                              start_date=a.strftime("%Y-%m-%d"), end_date=b.strftime("%Y-%m-%d"),
                              hourly=",".join(WEATHER_VARS), timezone="UTC", wind_speed_unit="ms")
                data = get_json(OPEN_METEO, params)
                cache_file.write_text(json.dumps(data))
                time.sleep(0.5)
            hourly = pd.DataFrame(data["hourly"])
            hourly["ts"] = pd.to_datetime(hourly.pop("time"), utc=True)
            hourly["point"] = point
            frames.append(hourly)
            print(f"   [weather] {point} {year} done")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- main
def main() -> None:
    ensure_dirs()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None, help="default: today 00:00 UTC")
    ap.add_argument("--skip-weather", action="store_true")
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") if args.end else pd.Timestamp.now(tz="UTC").floor("D")
    print(f"Downloading {start:%Y-%m-%d} -> {end:%Y-%m-%d}")

    print("1/3 National carbon intensity")
    intensity = intensity_frame(download_api("/intensity", "intensity", start, end))

    print("2/3 Generation mix")
    generation = generation_frame(download_api("/generation", "generation", start, end))

    print("3/3 Weather")
    if args.skip_weather:
        weather = pd.DataFrame(columns=["ts", "point", *WEATHER_VARS])
    else:
        weather = download_weather(start, end)

    intensity.to_parquet(RAW / "intensity.parquet", index=False)
    generation.to_parquet(RAW / "generation.parquet", index=False)
    weather.to_parquet(RAW / "weather_hourly.parquet", index=False)
    save_to_sqlite(intensity, generation, weather)

    print(f"\nDone. intensity={len(intensity):,} rows, generation={len(generation):,}, "
          f"weather={len(weather):,}")
    print(f"Range: {intensity['ts'].min()} -> {intensity['ts'].max()}")


if __name__ == "__main__":
    main()
