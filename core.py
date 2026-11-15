"""Features, the correction model, and probability maths shared by training and live forecasts."""
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from config import STATIONS

FEATS = ["t2", "rh", "td", "ws10", "gust", "cloud", "rad", "pres", "t80", "ws80", "blh",
         "u10", "v10", "u80", "v80", "dT", "shear", "ws10_prev", "ws10_next", "ws10_m3", "dt2", "spread",
         "hs", "hc", "ds", "dc", "nb_ratio", "nb_tb", "ws_adj", "t_adj"]
Z80 = 1.2816  # z-score of the 90th percentile


def uv(ws, wd):
    r = np.deg2rad(wd)
    return -ws * np.sin(r), -ws * np.cos(r)


def direction(u, v):
    return (np.rad2deg(np.arctan2(-u, -v)) + 360.0) % 360.0


def dir_diff(a, b):
    d = np.abs((np.asarray(a, float) - np.asarray(b, float) + 180.0) % 360.0 - 180.0)
    return d


def km(lat1, lon1, lat2, lon2):
    x = (lon2 - lon1) * 93.6  # km per degree longitude at 32.7 N
    y = (lat2 - lat1) * 110.9
    return math.hypot(x, y)


def add_features(fc):
    """fc: hourly forecast frame (time + short Open-Meteo names), sorted by time."""
    d = fc.sort_values("time").reset_index(drop=True).copy()
    d["u10"], d["v10"] = uv(d.ws10, d.wd10)
    d["u80"], d["v80"] = uv(d.ws80, d.wd80)
    d["dT"] = d.t80 - d.t2
    d["shear"] = d.ws80 - d.ws10
    d["ws10_prev"] = d.ws10.shift(1)
    d["ws10_next"] = d.ws10.shift(-1)
    d["ws10_m3"] = d.ws10.rolling(3, center=True, min_periods=1).mean()
    d["dt2"] = d.t2.diff()
    d["spread"] = d.t2 - d.td
    hr = d.time.dt.hour + 0.0
    doy = d.time.dt.dayofyear + 0.0
    d["hour"] = d.time.dt.hour
    d["hs"], d["hc"] = np.sin(hr / 24 * 2 * np.pi), np.cos(hr / 24 * 2 * np.pi)
    d["ds"], d["dc"] = np.sin(doy / 365.25 * 2 * np.pi), np.cos(doy / 365.25 * 2 * np.pi)
    return d


def bias_tables(tabs):
    """Per-station, per-hour-of-day habit: how observed wind/temperature differ from the forecast."""
    out = {}
    for sid, d in tabs.items():
        g = d.groupby("hour")
        ratio = (g.ws.mean() / g.ws10.mean().clip(lower=0.5)).reindex(range(24))
        tb = (g.temp.mean() - g.t2.mean()).reindex(range(24))
        out[sid] = {"ratio": ratio.fillna(ratio.mean()).fillna(0.75).tolist(), "tb": tb.fillna(tb.mean()).fillna(0.0).tolist()}
    return out


def idw_weights(lat, lon, sids, power=2.0):
    w = {}
    for s in sids:
        st = STATIONS[s]
        w[s] = 1.0 / max(km(lat, lon, st["lat"], st["lon"]), 1.0) ** power
    tot = sum(w.values()) or 1.0
    return {s: v / tot for s, v in w.items()}


def attach(d, lat, lon, tables, exclude=None):
    """Add neighbour-station knowledge for the point (lat, lon), distance-weighted."""
    sids = [s for s in tables if s != exclude]
    w = idw_weights(lat, lon, sids)
    ratio = np.zeros(24)
    tb = np.zeros(24)
    for s in sids:
        ratio += w[s] * np.asarray(tables[s]["ratio"])
        tb += w[s] * np.asarray(tables[s]["tb"])
    d = d.copy()
    h = d["hour"].values.astype(int)
    d["nb_ratio"] = ratio[h]
    d["nb_tb"] = tb[h]
    d["ws_adj"] = d.ws10 * d.nb_ratio
    d["t_adj"] = d.t2 + d.nb_tb
    return d


def _hgb(loss="squared_error", q=None, iters=160):
    kw = dict(max_iter=iters, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=0.5, random_state=7)
    if q is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=q, **kw)
    return HistGradientBoostingRegressor(loss=loss, **kw)


def train(df, only=None):
    X = df[FEATS]
    spec = {"ws_q10": ("ws", 0.1), "ws_q50": ("ws", 0.5), "ws_q90": ("ws", 0.9), "temp": ("temp", None), "u": ("u", None), "v": ("v", None)}
    models = {}
    for name, (target, q) in spec.items():
        if only and name not in only:
            continue
        m = _hgb(q=q)
        ok = df[target].notna()
        m.fit(X[ok], df.loc[ok, target])
        models[name] = m
    return models


def predict(models, d):
    X = d[FEATS]
    out = pd.DataFrame(index=d.index)
    if "ws_q10" in models:
        q = np.sort(np.clip(np.column_stack([models[k].predict(X) for k in ("ws_q10", "ws_q50", "ws_q90")]), 0, None), axis=1)
        out["q10"], out["q50"], out["q90"] = q[:, 0], q[:, 1], q[:, 2]
    elif "ws_q50" in models:
        out["q50"] = np.clip(models["ws_q50"].predict(X), 0, None)
    if "temp" in models:
        out["temp_p"] = models["temp"].predict(X)
    if "u" in models:
        out["dir_p"] = direction(models["u"].predict(X), models["v"].predict(X))
    return out


def _ncdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z, float) / math.sqrt(2.0)))


def cdf(x, q10, q50, q90, c=1.0):
    """P(wind < x) from the three predicted quantiles (two-piece normal), widened by c."""
    lo = np.maximum((q50 - q10) * c / Z80, 0.35)
    hi = np.maximum((q90 - q50) * c / Z80, 0.35)
    s = np.where(x < q50, lo, hi)
    return _ncdf((x - q50) / s)


def p_window(q10, q50, q90, wmin, wmax, c=1.0):
    return np.clip(cdf(wmax, q10, q50, q90, c) - cdf(wmin, q10, q50, q90, c), 0, 1)


def fit_c(obs, q10, q50, q90, target=0.8):
    """Smallest widening factor that makes the 10-90 band contain `target` of observations."""
    for c in np.arange(0.5, 4.01, 0.05):
        cov = np.mean((obs >= q50 - c * (q50 - q10)) & (obs <= q50 + c * (q90 - q50)))
        if cov >= target:
            return float(round(c, 2)), float(cov)
    return 4.0, float(cov)


def inversion_prob(rad, dT, cloud, ws50):
    """Estimated chance of a surface inversion. No two-height sensor exists at the field, so this is a
    rule-based estimate from forecast temperature at 2 m and 80 m, daylight, cloud and wind."""
    rad, dT, cloud, ws50 = (np.asarray(a, float) for a in (rad, dT, cloud, ws50))
    night = rad < 20
    p = np.where(night, 0.35, np.where(rad < 150, 0.15, 0.03))
    p = p + np.where(dT > 0, 0.30, 0.0) + np.where(dT > 2, 0.10, 0.0)
    p = p + np.where(night & (cloud < 30), 0.10, 0.0) + np.where(night & (ws50 < 4), 0.10, 0.0)
    p = p - np.where(ws50 > 8, 0.25, 0.0)
    return np.clip(p, 0.02, 0.95)


CARD = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def cardinal(deg):
    return CARD[int((float(deg) % 360) / 22.5 + 0.5) % 16]
