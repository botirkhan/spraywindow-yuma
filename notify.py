"""Real phone push alerts through ntfy.sh (free, no account).
The phone subscribes to a random topic name in the ntfy app; this module publishes to it."""
import os, json, time, secrets, threading
import requests
from config import DATA
import datasrc as ds

STATE = os.path.join(DATA, "notify.json")
TAGS = {"GO": ["white_check_mark"], "CAUTION": ["warning"], "STOP": ["no_entry"]}
_lock = threading.Lock()
_started = {"on": False}


def state():
    s = {}
    if os.path.exists(STATE):
        try:
            s = json.load(open(STATE))
        except Exception:
            s = {}
    if not s.get("topic"):
        s = {"topic": "spraywindow-" + secrets.token_hex(6), "enabled": False, "params": {}, "last_status": None, "sent": [], "last_error": None, "count": 0}
        _save(s)
    return s


def _save(s):
    with _lock:
        json.dump(s, open(STATE, "w"))


def public(s=None):
    s = s or state()
    return {"topic": s["topic"], "url": "https://ntfy.sh/" + s["topic"], "enabled": s["enabled"], "count": s.get("count", 0), "last_error": s.get("last_error")}


def send(title, body, kind="GO"):
    s = state()
    try:
        r = requests.post("https://ntfy.sh/", data=json.dumps({"topic": s["topic"], "title": title, "message": body or " ", "tags": TAGS.get(kind, []),
                                                               "priority": 4 if kind == "STOP" else 3}).encode("utf-8"), timeout=20)
        ok = r.status_code == 200
        s["last_error"] = None if ok else "ntfy answered %s: %s" % (r.status_code, r.text[:120])
    except Exception as e:
        ok = False
        s["last_error"] = str(e)[:160]
    if ok:
        s["count"] = s.get("count", 0) + 1
    _save(s)
    return ok


def enable(on, params):
    s = state()
    s["enabled"] = bool(on)
    s["params"] = params or {}
    s["last_status"] = None
    s["sent"] = []
    _save(s)
    if on:
        check(first=True)
    start()
    return public()


def check(first=False):
    """Send what is due: a status change right now, and scheduled alerts whose time has arrived."""
    import server  # late import: uses the same forecast path as the app
    s = state()
    if not s["enabled"]:
        return
    fc = server._forecast(s["params"], False)
    now = ds.now_local().strftime("%Y-%m-%dT%H:%M")
    cur = fc["now"]
    if first or (s["last_status"] and s["last_status"] != cur["status"]):
        title = ("Now: %s at your field" % cur["status"].title()) if first else "Status changed: %s -> %s" % (s["last_status"].title(), cur["status"].title())
        send(title, cur["reasons"][0]["text"] if cur["reasons"] else "", cur["status"])
        s = state()
    s["last_status"] = cur["status"]
    for a in fc["alerts"][1:]:
        key = a["title"]
        if a["at"] <= now and key not in s["sent"]:
            if not first:               # on first enable, do not replay alerts that were already due
                send(a["title"], a["body"], a["kind"])
                s2 = state()
                s["count"], s["last_error"] = s2.get("count", 0), s2.get("last_error")
            s["sent"] = (s["sent"] + [key])[-40:]
    s2 = state()
    s["count"], s["last_error"] = s2.get("count", 0), s2.get("last_error")
    _save(s)


def _loop():
    while True:
        time.sleep(180)
        try:
            check()
        except Exception as e:
            print("notify check failed:", e, flush=True)


def start():
    if not _started["on"]:
        _started["on"] = True
        threading.Thread(target=_loop, daemon=True).start()
