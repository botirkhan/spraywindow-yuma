"""Live forecast for any point: corrected wind/temperature, confidence, spray windows, log."""
import os, csv, json, time, pickle, threading, datetime as dt
import numpy as np
import pandas as pd
from config import *
import datasrc as ds
from core import *

_lock = threading.Lock()
_cache = {}
_model = {"mtime": 0, "obj": None}
LOG_PATH = os.path.join(LOGS, "decision_log_v2.csv")
USAGE_PATH = os.path.join(DATA, "usage.json")
LOG_COLS = ["logged_at", "lat", "lon", "valid_hour", "status", "confidence_pct", "ground_station_view", "wind_mph", "wind_low", "wind_high",
            "wind_from", "temp_f", "temp_80m_minus_2m_f", "inversion_chance_pct", "wind_80m_mph", "triggers", "next_window", "crop", "pesticide", "sensitive_areas", "data_source"]


def model():
    path = os.path.join(DATA, "model.pkl")
    if not os.path.exists(path):
        return None
    mt = os.path.getmtime(path)
    if _model["obj"] is None or mt != _model["mtime"]:
        with open(path, "rb") as f:
            _model["obj"] = pickle.load(f)
        _model["mtime"] = mt
    return _model["obj"]


def metrics():
    p = os.path.join(DATA, "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _cached(key, ttl, fn):
    """Return (value, source). Falls back to the last saved value when the network fails."""
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1], "cached"
    try:
        val = fn()
        with _lock:
            _cache[key] = (now, val)
        return val, "live"
    except Exception as e:
        if hit:
            return hit[1], "saved copy (network error)"
        raise


def _fc(lat, lon):
    key = "fc_%.3f_%.3f" % (lat, lon)
    disk = os.path.join(DATA, key + ".csv")

    def fn():
        df, el = ds.om_forecast(lat, lon)
        df["elev"] = el
        df.to_csv(disk, index=False)
        return df
    try:
        return _cached(key, 1200, fn)
    except Exception:
        if os.path.exists(disk):
            return pd.read_csv(disk, parse_dates=["time"]), "saved copy (offline)"
        raise


def station_state():
    """Latest reading at each station, plus how far our model is off there right now."""
    M = model()

    def fn():
        out = []
        for sid in (M["sids"] if M else STATIONS):
            st = STATIONS[sid]
            row = {"id": sid, "name": st["name"], "lat": st["lat"], "lon": st["lon"]}
            try:
                o = ds.azmet_hourly(sid, ds.now_local() - dt.timedelta(hours=8), "P1D")
                if len(o):
                    last = o.iloc[-1]
                    row.update(time=str(last.time), ws=round(float(last.ws), 1), wd=None if pd.isna(last.wd) else int(last.wd),
                               temp=round(float(last.temp), 1), dir=None if pd.isna(last.wd) else cardinal(last.wd))
                    if M:
                        fc, _ = _fc(st["lat"], st["lon"])
                        f = attach(add_features(fc), st["lat"], st["lon"], M["tables"], exclude=sid)
                        f = f[f.time == last.time]
                        if len(f):
                            p = predict(M["models"], f).iloc[0]
                            row.update(model_ws=round(float(p.q50), 1), res_ws=float(last.ws - p.q50), res_t=float(last.temp - p.temp_p),
                                       fc_ws=round(float(f.ws10.iloc[0]), 1))
            except Exception as e:
                row["error"] = str(e)[:120]
            out.append(row)
        return out
    try:
        return _cached("stations", 900, fn)
    except Exception:
        return [], "unavailable"


def _status_word(s):
    return {"GO": "Go", "CAUTION": "Caution", "STOP": "Stop"}[s]


def bearing(lat1, lon1, lat2, lon2):
    """Compass direction (degrees) from point 1 to point 2."""
    x = (lon2 - lon1) * 93.6
    y = (lat2 - lat1) * 110.9
    return (np.rad2deg(np.arctan2(x, y)) + 360.0) % 360.0


def forecast_point(lat, lon, wmin=3.0, wmax=10.0, conf=0.7, sens=None, profile=None, write_log=True):
    profile = profile or {}
    ptype = (profile.get("ptype") or "").lower()
    vol = (profile.get("vol") or "").lower()
    areas = []
    for i, a in enumerate(sens or []):
        try:
            la, lo_ = float(a["lat"]), float(a["lon"])
        except Exception:
            continue
        b = float(bearing(lat, lon, la, lo_))
        areas.append({"name": str(a.get("name") or "Area %d" % (i + 1))[:40], "lat": la, "lon": lo_, "bearing": round(b), "card": cardinal(b),
                      "miles": round(km(lat, lon, la, lo_) * 0.621, 2)})
    M = model()
    if M is None:
        raise RuntimeError("Model not trained yet. Run 2_train.bat first.")
    met = metrics() or {}
    fc, src = _fc(lat, lon)
    elev = float(fc["elev"].iloc[0]) if "elev" in fc.columns and pd.notna(fc["elev"].iloc[0]) else None
    f = attach(add_features(fc), lat, lon, M["tables"])
    p = predict(M["models"], f)
    d = pd.concat([f.reset_index(drop=True), p.reset_index(drop=True)], axis=1)
    now = ds.now_local()
    hour0 = now.replace(minute=0, second=0)
    d = d[d.time >= hour0].head(72).reset_index(drop=True)
    lead_h = np.asarray((d.time - pd.Timestamp(hour0)) / pd.Timedelta(hours=1), dtype=float)
    c = np.interp(lead_h / 24.0, [0, 1, 2, 3], M["calib"])

    # nudge the first hours toward what nearby stations are reading right now (only if testing showed it helps)
    stations, st_src = station_state()
    nudge_on = bool(met.get("nudge")) and met["nudge"][0]["mae_with"] < met["nudge"][0]["mae_without"]
    nudge_note = None
    if nudge_on:
        good = [s for s in stations if "res_ws" in s]
        if good:
            w = idw_weights(lat, lon, [s["id"] for s in good])
            near = min(km(lat, lon, s["lat"], s["lon"]) for s in good)
            damp = np.exp(-near / 40.0)
            rw = sum(w[s["id"]] * s["res_ws"] for s in good) * damp
            rt = sum(w[s["id"]] * s["res_t"] for s in good) * damp
            t_obs = pd.Timestamp(max(s["time"] for s in good))
            L = np.clip(np.asarray((d.time - t_obs) / pd.Timedelta(hours=1), dtype=float), 0, None)
            k = np.exp(-L / 3.0)
            for col in ("q10", "q50", "q90"):
                d[col] = np.clip(d[col] + k * rw, 0, None)
            d["temp_p"] = d["temp_p"] + k * rt
            nudge_note = "Nearby stations are reading %+.1f mph vs the model; first hours adjusted." % rw

    pwin = p_window(d.q10.values, d.q50.values, d.q90.values, wmin, wmax, c)
    pinv = inversion_prob(d.rad.values, d.dT.values, d.cloud.values, d.q50.values)
    lo = np.clip(d.q50 - c * (d.q50 - d.q10), 0, None)
    hi = d.q50 + c * (d.q90 - d.q50)
    ddiff = dir_diff(d.wd10.values, d.wd80.values)
    hours = []
    for i in range(len(d)):
        r = d.iloc[i]
        psafe = float(pwin[i] * (1 - pinv[i]))
        reasons, caution = [], False
        night = r.rad < 20
        if r.dT > 0:
            reasons.append({"code": "INVERSION", "text": "Air at 80 m is %.1f°F warmer than near the ground: a temperature inversion that traps spray in a shallow layer." % r.dT})
        elif pinv[i] >= 0.5:
            reasons.append({"code": "INVERSION_RISK", "text": "Calm, clear night air: an inversion is likely to form (%d%% chance)." % round(pinv[i] * 100)})
        if r.q50 < wmin:
            reasons.append({"code": "WIND_LOW", "text": "Wind near %.1f mph is under the %g mph minimum: spray can hang in the air and move unpredictably." % (r.q50, wmin)})
        elif r.q50 > wmax:
            reasons.append({"code": "WIND_HIGH", "text": "Wind near %.1f mph is over the %g mph limit." % (r.q50, wmax)})
        elif pwin[i] < 0.6:
            reasons.append({"code": "WIND_UNCERTAIN", "text": "Wind could be anywhere from %.0f to %.0f mph, so staying inside %g-%g mph is not assured." % (lo[i], hi[i], wmin, wmax)})
        if (ddiff[i] > 45 and r.ws80 > 6) or (r.ws80 - r.ws10) > 12:
            caution = True
            reasons.append({"code": "SHEAR", "text": "Wind above the canopy differs from ground level (%.0f mph from %s at 80 m vs %.0f mph from %s at 10 m)." % (
                r.ws80, cardinal(r.wd80), r.ws10, cardinal(r.wd10))})
        toward = False
        blowing_to = (float(r.dir_p) + 180) % 360
        hit = [a for a in areas if dir_diff(blowing_to, a["bearing"]) <= 45]
        if hit:
            caution = True
            toward = True
            reasons.append({"code": "TOWARD_SENSITIVE", "text": "Wind is blowing toward %s." % ", ".join("%s (%.1f mi to the %s)" % (a["name"], a["miles"], a["card"]) for a in hit)})
        if vol == "high" and r.temp_p >= 85:
            caution = True
            reasons.append({"code": "VOLATILITY_HEAT", "text": "At %.0f°F a high-volatility product turns to vapor more readily and can move after spraying." % r.temp_p})
        if vol == "high" and 0.3 <= pinv[i] < 0.5 and r.dT <= 0:
            caution = True
            reasons.append({"code": "VOLATILITY_INVERSION", "text": "A %d%% inversion chance is too high for a high-volatility product." % round(pinv[i] * 100)})
        if psafe < 0.4 or (r.dT > 0 and pinv[i] >= 0.6) or (toward and ptype == "herbicide"):
            status = "STOP"
        elif psafe >= conf and not caution:
            status = "GO"
        else:
            status = "CAUTION"
        if status == "GO" and not reasons:
            reasons.append({"code": "OK", "text": "Wind %.0f-%.0f mph from the %s, inside %g-%g mph, and no inversion expected." % (lo[i], hi[i], cardinal(r.dir_p), wmin, wmax)})
        ground = "GO" if wmin <= r.q50 <= wmax else "STOP"
        hours.append({
            "time": r.time.strftime("%Y-%m-%dT%H:%M"), "lead_h": int(lead_h[i]), "status": status, "conf": round(psafe, 3),
            "p_wind": round(float(pwin[i]), 3), "p_inv": round(float(pinv[i]), 3), "ws": round(float(r.q50), 1), "lo": round(float(lo[i]), 1),
            "hi": round(float(hi[i]), 1), "dir": int(r.dir_p), "card": cardinal(r.dir_p), "temp": round(float(r.temp_p), 1),
            "dT": round(float(r.dT), 1), "ws80": round(float(r.ws80), 1), "wd80": int(r.wd80), "fc_ws10": round(float(r.ws10), 1),
            "gust": round(float(r.gust * r.q50 / max(r.ws10, 0.5)), 1) if pd.notna(r.gust) else None, "cloud": int(r.cloud) if pd.notna(r.cloud) else None,
            "night": bool(night), "ground": ground, "reasons": reasons,
        })

    # spray windows: runs of consecutive Go hours
    slots, i = [], 0
    while i < len(hours):
        if hours[i]["status"] == "GO":
            j = i
            while j + 1 < len(hours) and hours[j + 1]["status"] == "GO":
                j += 1
            seg = hours[i:j + 1]
            end = pd.Timestamp(seg[-1]["time"]) + pd.Timedelta(hours=1)
            slots.append({"start": seg[0]["time"], "end": end.strftime("%Y-%m-%dT%H:%M"), "hours": len(seg),
                          "conf": round(float(np.mean([h["conf"] for h in seg])), 3), "conf_min": round(min(h["conf"] for h in seg), 3),
                          "ws_lo": min(h["lo"] for h in seg), "ws_hi": max(h["hi"] for h in seg), "card": seg[len(seg) // 2]["card"],
                          "temp_lo": min(h["temp"] for h in seg), "temp_hi": max(h["temp"] for h in seg)})
            after = hours[j + 1:j + 4]
            if vol in ("medium", "high") and after and max(h["p_inv"] for h in after) >= 0.5:
                slots[-1]["warn"] = "an inversion is likely within 3 hours after this window; vapor from a %s-volatility product can still drift, so finish early and keep watching" % vol
            i = j + 1
        else:
            i += 1

    cur = hours[0]

    # phone alerts the app would send (simulated: nothing leaves this computer)
    def _t(ts):
        return pd.Timestamp(ts).strftime("%a %I:%M %p").replace(" 0", " ")
    alerts = [{"at": now.strftime("%Y-%m-%dT%H:%M"), "kind": cur["status"], "title": "Now: %s at your field" % _status_word(cur["status"]),
               "body": cur["reasons"][0]["text"] if cur["reasons"] else ""}]
    for s in slots:
        st, en = pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
        if st > pd.Timestamp(hour0):
            alerts.append({"at": (st - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M"), "kind": "GO", "title": "Spray window opens %s" % _t(st),
                           "body": "%d hour%s, %d%% confidence, wind %.0f-%.0f mph from the %s." % (s["hours"], "" if s["hours"] == 1 else "s", round(s["conf"] * 100), s["ws_lo"], s["ws_hi"], s["card"])})
        nxt_h = next((h for h in hours if h["time"] == s["end"]), None)
        if nxt_h:
            alerts.append({"at": (en - pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M"), "kind": nxt_h["status"], "title": "Window closes %s" % _t(en),
                           "body": "Finish up. Then: " + (nxt_h["reasons"][0]["text"] if nxt_h["reasons"] else _status_word(nxt_h["status"]))})
    for k in range(1, len(hours)):
        inv_now = any(r["code"] == "INVERSION" for r in hours[k]["reasons"])
        inv_before = any(r["code"] == "INVERSION" for r in hours[k - 1]["reasons"])
        if inv_now and not inv_before:
            alerts.append({"at": (pd.Timestamp(hours[k]["time"]) - pd.Timedelta(minutes=60)).strftime("%Y-%m-%dT%H:%M"), "kind": "STOP",
                           "title": "Inversion expected %s" % _t(hours[k]["time"]), "body": "Do not spray. " + hours[k]["reasons"][0]["text"]})
    alerts = sorted(alerts, key=lambda x: x["at"])[:9]

    def fmt(ts):
        t = pd.Timestamp(ts)
        return t.strftime("%a %b %d, %I:%M %p").replace(" 0", " ")
    nxt = slots[0] if slots else None
    nxt_txt = ("%s to %s (%d%% confidence)" % (fmt(nxt["start"]), pd.Timestamp(nxt["end"]).strftime("%I:%M %p").lstrip("0"), round(nxt["conf"] * 100))) if nxt else "none in the next 72 hours"
    advice = "Right now: %s. %s " % (_status_word(cur["status"]).upper(), cur["reasons"][0]["text"] if cur["reasons"] else "")
    if cur["ground"] == "GO" and cur["status"] != "GO":
        advice += "A single ground-level wind reading (%.0f mph) would look fine, but it misses this. " % cur["ws"]
    if nxt:
        advice += "Next safe window: %s, wind %.0f-%.0f mph from the %s." % (nxt_txt, nxt["ws_lo"], nxt["ws_hi"], nxt["card"])
        if nxt.get("warn"):
            advice += " Note: %s." % nxt["warn"]
        if len(slots) > 1:
            advice += " %d more window%s in the next 3 days." % (len(slots) - 1, " follows" if len(slots) == 2 else "s follow")
    else:
        advice += "No safe window is forecast in the next 72 hours with your settings."
    advice += " Always follow the product label; this tool supports your decision and does not replace it."

    usage = _usage(advice, hours, slots)
    obs_ws = [s["ws"] for s in stations if "ws" in s]
    obs_t = [s["temp"] for s in stations if "temp" in s]
    out = {
        "point": {"lat": lat, "lon": lon, "elev_m": elev,
                  "nearest": sorted([{"name": STATIONS[s]["name"], "km": round(km(lat, lon, STATIONS[s]["lat"], STATIONS[s]["lon"]), 1)} for s in M["sids"]], key=lambda x: x["km"])[:3]},
        "generated": now.strftime("%Y-%m-%d %H:%M:%S"), "source": "SIMULATED DATA" if ds.SYNTH else src,
        "settings": {"wmin": wmin, "wmax": wmax, "conf": conf}, "profile": profile, "sens": areas,
        "now": cur, "hours": hours, "slots": slots, "alerts": alerts, "advice": advice, "nudge": nudge_note,
        "stations": stations, "stations_source": st_src,
        "spread": {"ws": round(max(obs_ws) - min(obs_ws), 1) if obs_ws else None, "temp": round(max(obs_t) - min(obs_t), 1) if obs_t else None},
        "usage": usage,
    }
    if write_log:
        _log([out["generated"], round(lat, 4), round(lon, 4), cur["time"], cur["status"], round(cur["conf"] * 100), cur["ground"], cur["ws"], cur["lo"], cur["hi"],
              cur["card"], cur["temp"], cur["dT"], round(cur["p_inv"] * 100), cur["ws80"], " | ".join(r["code"] for r in cur["reasons"]), nxt_txt, profile.get("crop") or "", " ".join(x for x in (profile.get("ptype"), profile.get("product"), ("(%s volatility)" % vol) if vol else "") if x),
              "; ".join("%s %.1f mi %s" % (a["name"], a["miles"], a["card"]) for a in areas), out["source"]])
    return out


def _log(row):
    new = not os.path.exists(LOG_PATH)
    with _lock, open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_COLS)
        w.writerow(row)


def read_log(n=200):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:][::-1]


def _usage(advice, hours, slots):
    """ESTIMATE of what a chat-model advice layer would use. No chat model is connected."""
    prompt_chars = 900 + 60 * len(hours) + 120 * len(slots)   # instructions + the hourly table it would be given
    est = {"in": int(prompt_chars / 4), "out": int(len(advice) / 4)}
    tot = {"calls": 0, "in": 0, "out": 0}
    if os.path.exists(USAGE_PATH):
        try:
            tot = json.load(open(USAGE_PATH))
        except Exception:
            pass
    tot = {"calls": tot["calls"] + 1, "in": tot["in"] + est["in"], "out": tot["out"] + est["out"]}
    with _lock:
        json.dump(tot, open(USAGE_PATH, "w"))
    return {"this_call": est, "total": tot, "connected": False}
