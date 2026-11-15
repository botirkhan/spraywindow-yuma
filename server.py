"""Local web app. Run: python server.py  then open http://127.0.0.1:8000"""
import os, sys, json, subprocess, threading
from flask import Flask, request, jsonify, send_from_directory, Response
from config import *
import datasrc as ds
import engine

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
                    "metrics": engine.metrics(), "runs": runs, "synthetic": ds.SYNTH, "training": _train})


@app.route("/api/forecast")
def forecast():
    try:
        a = request.args
        lat, lon = float(a.get("lat", DEFAULT_POINT["lat"])), float(a.get("lon", DEFAULT_POINT["lon"]))
        if not (BOUNDS["lat_min"] <= lat <= BOUNDS["lat_max"] and BOUNDS["lon_min"] <= lon <= BOUNDS["lon_max"]):
            return jsonify({"error": "That point is outside the Yuma area this model was trained for."}), 400
        avoid = a.get("avoid")
        out = engine.forecast_point(lat, lon, float(a.get("wmin", 3)), float(a.get("wmax", 10)), float(a.get("conf", 0.7)),
                                    float(avoid) if avoid not in (None, "", "none") else None)
        return jsonify(out)
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
    app.run(host="127.0.0.1", port=8000, debug=False, threaded=True)
