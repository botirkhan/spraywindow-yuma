"""Local web app. Run: python server.py  then open http://127.0.0.1:8000"""
import os, sys, json, subprocess, threading
from flask import Flask, request, jsonify, send_from_directory, Response
from config import *
import datasrc as ds
import engine, store, advisor, notify

app = Flask(__name__, static_folder=os.path.join(ROOT, "static"))
_train = {"running": False, "last": None, "ok": None}


@app.route("/")
def index():
    r = send_from_directory(app.static_folder, "index.html")
    r.headers["Cache-Control"] = "no-store"
    return r


@app.route("/api/meta")
def meta():
    runs = []
    p = os.path.join(DATA, "runs.json")
    if os.path.exists(p):
        runs = json.load(open(p))
    return jsonify({"stations": [dict(id=k, **v) for k, v in STATIONS.items()], "bounds": BOUNDS, "default": DEFAULT_POINT,
                    "metrics": engine.metrics(), "runs": runs, "synthetic": ds.SYNTH, "training": _train,
                    "gemini": {"connected": advisor.connected(), "usage": advisor.usage()}})


def _forecast(a, write_log):
    lat, lon = float(a.get("lat", DEFAULT_POINT["lat"])), float(a.get("lon", DEFAULT_POINT["lon"]))
    sens = a.get("sens") or []
    if isinstance(sens, str):
        sens = json.loads(sens) if sens.strip() else []
    profile = {k: (a.get(k) or "") for k in ("crop", "ptype", "product", "vol")}
    return engine.forecast_point(lat, lon, float(a.get("wmin", 3)), float(a.get("wmax", 10)), float(a.get("conf", 0.7)), sens[:12], profile, write_log)


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        b = request.get_json(force=True)
        fc = _forecast(b.get("params") or {}, False)
        return jsonify(advisor.chat(b.get("message", ""), b.get("history", []), fc))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"connected": advisor.connected(), "reply": "The advisor hit an error: %s" % str(e)[:200]}), 200


@app.route("/api/notify", methods=["GET", "POST"])
def notify_api():
    if request.method == "POST":
        b = request.get_json(force=True)
        if b.get("action") == "test":
            fc = _forecast(b.get("params") or {}, False)
            a = fc["alerts"][0]
            ok = notify.send(a["title"], a["body"], a["kind"])
            return jsonify(dict(notify.public(), ok=ok))
        return jsonify(notify.enable(bool(b.get("on")), b.get("params") or {}))
    return jsonify(notify.public())


@app.route("/api/sprays", methods=["GET", "POST"])
def sprays():
    if request.method == "POST":
        store.add(request.get_json(force=True))
    return jsonify(store.recent())


@app.route("/api/sprays/<int:sid>", methods=["POST", "DELETE"])
def spray_one(sid):
    if request.method == "DELETE":
        store.delete(sid)
    else:
        store.set_outcome(sid, request.get_json(force=True).get("outcome", ""))
    return jsonify(store.recent())


@app.route("/api/forecast")
def forecast():
    try:
        a = request.args
        lat, lon = float(a.get("lat", DEFAULT_POINT["lat"])), float(a.get("lon", DEFAULT_POINT["lon"]))
        if not (BOUNDS["lat_min"] <= lat <= BOUNDS["lat_max"] and BOUNDS["lon_min"] <= lon <= BOUNDS["lon_max"]):
            return jsonify({"error": "That point is outside the Yuma area this model was trained for."}), 400
        return jsonify(_forecast(a, True))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/log")
def log():
    return jsonify(engine.read_log())


@app.route("/api/log.csv")
def log_csv():
    if not os.path.exists(engine.LOG_PATH):
        return Response("", mimetype="text/csv")
    return Response(open(engine.LOG_PATH, encoding="utf-8").read(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=spray_decision_log.csv"})


def _run_training():
    _train.update(running=True, ok=None)
    try:
        with open(os.path.join(LOGS, "retrain.log"), "w") as f:
            rc = subprocess.call([sys.executable, os.path.join(ROOT, "pipeline.py"), "--refresh"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        _train.update(ok=(rc == 0), last=ds.now_local().strftime("%Y-%m-%d %H:%M"))
    finally:
        _train["running"] = False


@app.route("/api/retrain", methods=["POST"])
def retrain():
    if not _train["running"]:
        threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"started": True})


@app.route("/api/retrain/status")
def retrain_status():
    return jsonify(_train)


if __name__ == "__main__":
    print("SprayWindow Yuma running at http://127.0.0.1:8000", flush=True)
    if notify.state().get("enabled"):
        notify.start()
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
