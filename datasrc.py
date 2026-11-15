"""Data access: AZMet station observations and Open-Meteo forecasts.
Set SPRAY_SYNTH=1 to use generated data (offline testing only)."""
import os, time, math, datetime as dt
import numpy as np
import pandas as pd
import requests
from config import *

SYNTH = os.environ.get("SPRAY_SYNTH") == "1"
_S = requests.Session()
_S.headers["User-Agent"] = "SprayWindowYuma/1.0 (Hack Arizona AgX student project)"


def now_local():
    """Arizona time (UTC-7, no daylight saving), naive."""
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=7)).replace(tzinfo=None, microsecond=0)


def _get(url, params=None, timeout=120, tries=3):
    last = None
    for i in range(tries):
        try:
            r = _S.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError("HTTP %s from %s: %s" % (r.status_code, url, r.text[:200]))
            if r.status_code in (400, 404):
                break
        except Exception as e:  # network problem
            last = e
        time.sleep(2 * (i + 1))
    raise last


def _num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# ---------------------------------------------------------------- AZMet
def azmet_hourly(sid, start=None, interval=None):
    """Hourly observations. `time` is the END of each hour, Arizona time."""
    if SYNTH:
        end = now_local().replace(minute=0, second=0)
        beg = start or (end - dt.timedelta(hours=24))
        return _synth_obs(sid, pd.date_range(beg, end, freq="h")[1:])
    s = start.strftime("%Y-%m-%dT%H:%M") if start else "*"
    j = _get("%s/hourly/%s/%s/%s" % (AZMET, sid, s, interval or "*"))
    df = pd.DataFrame(j.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["time", "ws", "wd", "temp", "rh", "gust_obs"])
    yr = _num(df, "date_year").astype(int)
    doy = _num(df, "date_doy").astype(int)
    hr = (_num(df, "date_hour") // 100).astype(int)
    t = pd.to_datetime(yr.astype(str) + "-01-01") + pd.to_timedelta(doy - 1, unit="D") + pd.to_timedelta(hr, unit="h")
    out = pd.DataFrame({
        "time": t, "ws": _num(df, "wind_spd_mph"), "wd": _num(df, "wind_vector_dir"),
        "temp": _num(df, "temp_airF"), "rh": _num(df, "relative_humidity"),
        "gust_obs": _num(df, "wind_spd_max_mph"),
    })
    out = out[(out.ws >= 0) & (out.ws < 90) & (out.temp > -20) & (out.temp < 135)]
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------- Open-Meteo
def _om(url, lat, lon, hourly, **extra):
    p = {"latitude": round(float(lat), 4), "longitude": round(float(lon), 4), "hourly": ",".join(hourly),
         "wind_speed_unit": "mph", "temperature_unit": "fahrenheit", "timezone": "America/Phoenix"}
    p.update(extra)
    j = _get(url, p, timeout=240)
    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df, j.get("elevation")


def om_forecast(lat, lon):
    """Yesterday + next 4 days, hourly. Returns (df, elevation_m)."""
    if SYNTH:
        t0 = now_local().replace(hour=0, minute=0, second=0) - dt.timedelta(days=1)
        return _synth_fc(lat, lon, pd.date_range(t0, periods=120, freq="h")), 40.0
    df, el = _om(OM_FC, lat, lon, list(OMV), past_days=1, forecast_days=4)
    return df.rename(columns=OMV), el


def om_hist(lat, lon, start, end):
    """Archived forecasts (short lead) for training."""
    if SYNTH:
        return _synth_fc(lat, lon, pd.date_range(start, end + dt.timedelta(hours=23), freq="h"))
    df, _ = _om(OM_HIST, lat, lon, list(OMV), start_date=start.strftime("%Y-%m-%d"), end_date=end.strftime("%Y-%m-%d"))
    return df.rename(columns=OMV)


def om_prev(lat, lon, past_days=PREV_DAYS):
    """Forecasts as issued 1, 2 and 3 days earlier. Returns {lead_day: df}."""
    if SYNTH:
        t0 = now_local().replace(hour=0, minute=0, second=0) - dt.timedelta(days=past_days)
        times = pd.date_range(t0, periods=(past_days + 1) * 24, freq="h")
        return {k: _synth_fc(lat, lon, times, lead=k) for k in (1, 2, 3)}
    names = list(OMV)
    for attempt in (names, [n for n in names if n != "boundary_layer_height"]):
        try:
            hv = ["%s_previous_day%d" % (n, k) for k in (1, 2, 3) for n in attempt]
            df, _ = _om(OM_PREV, lat, lon, hv, past_days=past_days, forecast_days=1)
            out = {}
            for k in (1, 2, 3):
                d = pd.DataFrame({"time": df["time"]})
                for n in names:
                    c = "%s_previous_day%d" % (n, k)
                    d[OMV[n]] = df[c] if c in df.columns else np.nan
                out[k] = d
            return out
        except Exception as e:
            err = e
    raise err


# ---------------------------------------------------------------- synthetic (testing only)
def _synth_fc(lat, lon, times, lead=0):
    t = np.asarray((pd.DatetimeIndex(times) - pd.Timestamp("2025-01-01")) / pd.Timedelta(hours=1), dtype=float)
    h = pd.DatetimeIndex(times).hour.values
    ph = (lon + 114.7) * 2.0 + (lat - 32.7) * 3.0
    reg = 2.2 * np.sin(t / 53.0 + ph) + 1.6 * np.sin(t / 131.0) + 0.9 * np.sin(t / 17.0 + 1.0)
    err = lead * 0.9 * np.sin(t / 29.0 + lead * 2.1)
    ws10 = np.clip(6.0 + 2.5 * np.sin((h - 15) / 24 * 2 * np.pi) + reg + err, 0.3, None)
    t2 = 72 + 18 * np.sin(t / 8760 * 2 * np.pi - 1.9) + 12 * np.sin((h - 9) / 24 * 2 * np.pi) + 2 * np.sin(t / 97.0) + lead * 0.8 * np.sin(t / 41.0)
    rad = np.clip(800 * np.sin((h - 6) / 12 * np.pi), 0, None) * (h >= 6) * (h <= 18)
    wd10 = (250 + 60 * np.sin(t / 47.0 + ph) + 40 * np.sin((h - 3) / 24 * 2 * np.pi)) % 360
    night = rad < 20
    t80 = t2 + np.where(night, 2.5 - 0.35 * ws10, -3.5)
    return pd.DataFrame({
        "time": pd.DatetimeIndex(times), "t2": t2, "rh": 35 + 20 * np.sin((h - 3) / 24 * 2 * np.pi), "td": t2 - 25,
        "ws10": ws10, "wd10": wd10, "gust": ws10 * 1.6, "cloud": np.clip(30 + 40 * np.sin(t / 71.0), 0, 100),
        "rad": rad, "pres": 1009 + 3 * np.sin(t / 113.0), "t80": t80, "ws80": ws10 * np.where(night, 1.9, 1.15),
        "wd80": (wd10 + np.where(night, 35, 5)) % 360, "blh": np.where(night, 120, 1400.0),
    })


def _synth_obs(sid, times):
    st = STATIONS[sid]
    f = _synth_fc(st["lat"], st["lon"], times)
    rng = np.random.default_rng(int(sid[2:]) * 7919 + len(times))
    h = f.time.dt.hour.values
    off = (int(sid[2:]) % 5) * 0.12
    ws = np.clip(f.ws10 * (0.62 + off + 0.12 * np.sin((h - 14) / 24 * 2 * np.pi)) + rng.normal(0, 1.0, len(f)), 0, None)
    return pd.DataFrame({"time": f.time, "ws": ws, "wd": (f.wd10 + rng.normal(0, 25, len(f))) % 360,
                         "temp": f.t2 - 1.5 + 4 * off + rng.normal(0, 1.2, len(f)), "rh": f.rh, "gust_obs": ws * 1.5})
