"""Download data, test the model honestly, train the final model.
Usage: python pipeline.py [--refresh]"""
import os, sys, json, time, pickle, datetime as dt
import numpy as np
import pandas as pd
from config import *
import datasrc as ds
from core import *

REFRESH = "--refresh" in sys.argv
WMIN, WMAX = 3.0, 10.0


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


def cached(name, fn, max_age_h=None):
    path = os.path.join(DATA, name + ".csv")
    fresh = os.path.exists(path) and (max_age_h is None or (time.time() - os.path.getmtime(path)) < max_age_h * 3600)
    if fresh and not REFRESH:
        return pd.read_csv(path, parse_dates=["time"])
    try:
        df = fn()
        df.to_csv(path, index=False)
        return df
    except Exception as e:
        if os.path.exists(path):
            log("  download failed (%s); using saved copy of %s" % (e, name))
            return pd.read_csv(path, parse_dates=["time"])
        raise


def download():
    end = ds.now_local().replace(hour=0, minute=0, second=0)
    start = end - dt.timedelta(days=HISTORY_DAYS)
    obs, hist, prev = {}, {}, {}
    for sid, st in STATIONS.items():
        log("download", sid, st["name"])
        obs[sid] = cached("obs_" + sid, lambda: ds.azmet_hourly(sid, start, "P%dD" % (HISTORY_DAYS + 1)))
        hist[sid] = cached("hist_" + sid, lambda: ds.om_hist(st["lat"], st["lon"], start, end - dt.timedelta(days=1)))
        try:
            pv = {}
            if REFRESH or not all(os.path.exists(os.path.join(DATA, "prev_%s_d%d.csv" % (sid, k))) for k in (1, 2, 3)):
                got = ds.om_prev(st["lat"], st["lon"])
                for k in (1, 2, 3):
                    got[k].to_csv(os.path.join(DATA, "prev_%s_d%d.csv" % (sid, k)), index=False)
            for k in (1, 2, 3):
                pv[k] = pd.read_csv(os.path.join(DATA, "prev_%s_d%d.csv" % (sid, k)), parse_dates=["time"])
            prev[sid] = pv
        except Exception as e:
            log("  lead-time data unavailable for", sid, "-", e)
        log("  obs rows %d, forecast rows %d" % (len(obs[sid]), len(hist[sid])))
    return obs, hist, prev


def table(sid, obs, fc):
    f = add_features(fc)
    d = obs.merge(f, on="time", how="inner", suffixes=("_obs", ""))
    d["sid"] = sid
    d["u"], d["v"] = uv(d.ws, d.wd)
    return d.dropna(subset=["ws", "temp", "ws10", "t2"]).reset_index(drop=True)


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[ok] - b[ok]))) if ok.any() else None


def window_scores(truth_ws, pred_ok):
    t = (truth_ws >= WMIN) & (truth_ws <= WMAX)
    p = np.asarray(pred_ok, bool)
    acc = float(np.mean(t == p))
    false_safe = float(np.mean(~t[p])) if p.any() else None   # said OK, was not OK
    return {"accuracy": acc, "false_safe": false_safe, "said_ok": float(np.mean(p))}


def nearest(sid, pool):
    st = STATIONS[sid]
    return min((s for s in pool if s != sid), key=lambda s: km(st["lat"], st["lon"], STATIONS[s]["lat"], STATIONS[s]["lon"]))


def main():
    t_start = time.time()
    obs, hist, prev = download()
    tabs = {s: table(s, obs[s], hist[s]) for s in STATIONS}
    tabs = {s: d for s, d in tabs.items() if len(d) > 500}
    sids = list(tabs)
    n_rows = int(sum(len(d) for d in tabs.values()))
    log("training rows:", n_rows, "stations:", sids)
    M = {"stations": {s: dict(STATIONS[s], rows=len(tabs[s])) for s in sids}, "n_rows": n_rows,
         "period": [str(min(d.time.min() for d in tabs.values())), str(max(d.time.max() for d in tabs.values()))],
         "synthetic": ds.SYNTH}

    # ---------- Test A: leave one station out (can it predict a place with no station?)
    log("Test A: leave-one-station-out")
    pooled = []
    for s in sids:
        others = [o for o in sids if o != s]
        tb = bias_tables({o: tabs[o] for o in others})
        tr = pd.concat([attach(tabs[o], STATIONS[o]["lat"], STATIONS[o]["lon"], tb, exclude=o) for o in others], ignore_index=True)
        te = attach(tabs[s], STATIONS[s]["lat"], STATIONS[s]["lon"], tb)
        p = predict(train(tr), te)
        r = pd.concat([te[["time", "sid", "ws", "wd", "temp", "ws10", "wd10", "t2"]].reset_index(drop=True), p.reset_index(drop=True)], axis=1)
        nb = nearest(s, sids)
        r = r.merge(tabs[nb][["time", "ws", "temp", "wd"]].rename(columns={"ws": "nb_ws", "temp": "nb_temp", "wd": "nb_wd"}), on="time", how="left")
        r["nb"] = nb
        pooled.append(r)
        log("  %s: wind MAE model %.2f | forecast %.2f | nearest station %.2f" % (s, mae(r.ws, r.q50), mae(r.ws, r.ws10 * HEIGHT_FACTOR), mae(r.ws, r.nb_ws)))
    P = pd.concat(pooled, ignore_index=True)
    c0, cov0 = fit_c(P.ws.values, P.q10.values, P.q50.values, P.q90.values)
    pw = p_window(P.q10.values, P.q50.values, P.q90.values, WMIN, WMAX, c0)
    strong = P.ws > 5

    def block(d, pwd):
        st = d.ws > 5
        return {
            "wind_mae": {"model": mae(d.ws, d.q50), "forecast_raw": mae(d.ws, d.ws10), "forecast_height_adj": mae(d.ws, d.ws10 * HEIGHT_FACTOR), "nearest_station": mae(d.ws, d.nb_ws)},
            "temp_mae": {"model": mae(d.temp, d.temp_p), "forecast_raw": mae(d.temp, d.t2), "nearest_station": mae(d.temp, d.nb_temp)},
            "dir_mae": {"model": float(np.mean(dir_diff(d.wd[st], d.dir_p[st]))) if st.any() else None,
                        "forecast_raw": float(np.mean(dir_diff(d.wd[st], d.wd10[st]))) if st.any() else None,
                        "nearest_station": float(np.nanmean(dir_diff(d.wd[st], d.nb_wd[st]))) if st.any() else None},
            "window": {"model": window_scores(d.ws.values, pwd >= 0.5),
                       "forecast_height_adj": window_scores(d.ws.values, ((d.ws10 * HEIGHT_FACTOR) >= WMIN) & ((d.ws10 * HEIGHT_FACTOR) <= WMAX)),
                       "nearest_station": window_scores(d.ws.values, (d.nb_ws >= WMIN) & (d.nb_ws <= WMAX))},
            "n": int(len(d)),
        }
    A = block(P, pw)
    A["coverage_raw"] = float(np.mean((P.ws >= P.q10) & (P.ws <= P.q90)))
    A["coverage_calibrated"] = cov0
    A["per_station"] = {s: {"name": STATIONS[s]["name"], "wind_mae_model": mae(g.ws, g.q50), "wind_mae_forecast": mae(g.ws, g.ws10 * HEIGHT_FACTOR),
                            "wind_mae_nearest": mae(g.ws, g.nb_ws), "temp_mae_model": mae(g.temp, g.temp_p), "nearest": STATIONS[g.nb.iloc[0]]["name"]}
                        for s, g in P.groupby("sid")}
    # reliability: when we say X% chance the wind is in range, how often is it?
    truth = ((P.ws >= WMIN) & (P.ws <= WMAX)).values
    rel = []
    for lo in np.arange(0, 1.0, 0.1):
        m = (pw >= lo) & (pw < lo + 0.1 + (1e-9 if lo < 0.9 else 1))
        if m.sum() >= 30:
            rel.append({"said": float(np.mean(pw[m])), "happened": float(np.mean(truth[m])), "n": int(m.sum())})
    A["reliability"] = rel
    hi = pw >= 0.8
    A["high_conf"] = {"share": float(np.mean(hi)), "happened": float(np.mean(truth[hi])) if hi.any() else None}
    M["loso"] = A

    # ---------- Test B: train on the past, test on the last 60 days at 0/1/2/3 days ahead
    log("Test B: forward test by lead time")
    tmax = max(d.time.max() for d in tabs.values())
    cut = tmax - pd.Timedelta(days=TEST_DAYS)
    past = {s: d[d.time < cut] for s, d in tabs.items()}
    tbB = bias_tables(past)
    trB = pd.concat([attach(past[s], STATIONS[s]["lat"], STATIONS[s]["lon"], tbB, exclude=s) for s in sids], ignore_index=True)
    mB = train(trB)
    leads, calib, resid0 = [], [c0], {}
    for k in (0, 1, 2, 3):
        rows = []
        for s in sids:
            if k == 0:
                te = tabs[s][tabs[s].time >= cut]
            else:
                if s not in prev:
                    continue
                te = table(s, obs[s], prev[s][k])
                te = te[te.time >= cut]
            if len(te) < 50:
                continue
            te = attach(te, STATIONS[s]["lat"], STATIONS[s]["lon"], tbB, exclude=s)
            p = predict(mB, te)
            r = pd.concat([te[["time", "sid", "ws", "temp", "ws10", "t2"]].reset_index(drop=True), p.reset_index(drop=True)], axis=1)
            rows.append(r)
        if not rows:
            continue
        R = pd.concat(rows, ignore_index=True)
        ck, covk = fit_c(R.ws.values, R.q10.values, R.q50.values, R.q90.values)
        pk = p_window(R.q10.values, R.q50.values, R.q90.values, WMIN, WMAX, ck)
        leads.append({"lead_days": k, "wind_mae_model": mae(R.ws, R.q50), "wind_mae_forecast": mae(R.ws, R.ws10 * HEIGHT_FACTOR),
                      "temp_mae_model": mae(R.temp, R.temp_p), "temp_mae_forecast": mae(R.temp, R.t2),
                      "window_accuracy": window_scores(R.ws.values, pk >= 0.5)["accuracy"], "band_factor": ck, "n": int(len(R))})
        if k == 0:
            resid0 = R.assign(res=R.ws - R.q50)
            calib = [ck]
        else:
            calib.append(max(ck, calib[-1]))
        log("  lead %d d: wind MAE model %.2f | forecast %.2f | band x%.2f" % (k, leads[-1]["wind_mae_model"], leads[-1]["wind_mae_forecast"], ck))
    while len(calib) < 4:
        calib.append(round(calib[-1] * 1.15, 2))
    M["lead"] = leads
    M["calib"] = calib

    # station nudge: does the latest error at nearby stations help the next few hours?
    nudge = []
    if len(resid0):
        piv = resid0.pivot_table(index="time", columns="sid", values="res")
        for L in (1, 3, 6):
            base, fixed = [], []
            for s in piv.columns:
                w = idw_weights(STATIONS[s]["lat"], STATIONS[s]["lon"], [o for o in piv.columns if o != s])
                nb = sum(piv[o].shift(L) * w[o] for o in w)
                g = resid0[resid0.sid == s].set_index("time")
                adj = np.clip(g.q50 + np.exp(-L / 3.0) * nb.reindex(g.index).fillna(0), 0, None)
                base.append(np.abs(g.ws - g.q50)); fixed.append(np.abs(g.ws - adj))
            nudge.append({"hours_ahead": L, "mae_without": float(pd.concat(base).mean()), "mae_with": float(pd.concat(fixed).mean())})
        log("  station nudge:", nudge)
    M["nudge"] = nudge

    # learning curve: more history -> lower error?
    curve = []
    teL = pd.concat([attach(tabs[s][tabs[s].time >= cut], STATIONS[s]["lat"], STATIONS[s]["lon"], tbB, exclude=s) for s in sids], ignore_index=True)
    for months in (1, 2, 4, 7, 10):
        sub = trB[trB.time >= cut - pd.Timedelta(days=30 * months)]
        if len(sub) < 800 or (curve and len(sub) == curve[-1]["rows"]):
            continue
        p = predict(train(sub, only=("ws_q50", "temp")), teL)
        curve.append({"months": months, "rows": int(len(sub)), "wind_mae": mae(teL.ws, p.q50), "temp_mae": mae(teL.temp, p.temp_p)})
    M["learning_curve"] = curve
    log("  learning curve:", [(c["months"], round(c["wind_mae"], 2)) for c in curve])

    # ---------- Final model on everything
    log("training final model")
    t0 = time.time()
    tb = bias_tables(tabs)
    full = pd.concat([attach(tabs[s], STATIONS[s]["lat"], STATIONS[s]["lon"], tb, exclude=s) for s in sids], ignore_index=True)
    models = train(full)
    fit_s = time.time() - t0
    t0 = time.time()
    predict(models, full.head(72))
    pred_ms = (time.time() - t0) * 1000
    path = os.path.join(DATA, "model.pkl")
    with open(path, "wb") as f:
        pickle.dump({"models": models, "tables": tb, "calib": calib, "feats": FEATS, "sids": sids}, f)
    M["spec"] = {"final_fit_seconds": round(fit_s, 1), "pipeline_seconds": round(time.time() - t_start, 1), "predict_ms_72h": round(pred_ms, 1),
                 "model_mb": round(os.path.getsize(path) / 1e6, 2), "features": len(FEATS), "algorithm": "Gradient-boosted trees (scikit-learn), 6 small models",
                 "data_cost_usd": 0.0}
    M["trained_at"] = ds.now_local().strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(DATA, "metrics.json"), "w") as f:
        json.dump(M, f, indent=1)
    hp = os.path.join(DATA, "runs.json")
    runs = json.load(open(hp)) if os.path.exists(hp) else []
    runs.append({"trained_at": M["trained_at"], "rows": n_rows, "wind_mae": A["wind_mae"]["model"], "temp_mae": A["temp_mae"]["model"],
                 "window_accuracy": A["window"]["model"]["accuracy"]})
    json.dump(runs, open(hp, "w"), indent=1)
    log("DONE. wind MAE (unseen station): model %.2f mph vs forecast %.2f vs nearest station %.2f" % (
        A["wind_mae"]["model"], A["wind_mae"]["forecast_height_adj"], A["wind_mae"]["nearest_station"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("PIPELINE FAILED", flush=True)
        sys.exit(1)
