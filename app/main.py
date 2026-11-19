import json
import logging
import os
import sys
import time

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from preprocess import normalize  # noqa: E402,F401  (cần để joblib load pipeline)

from app import db  # noqa: E402

app = FastAPI(title="Lab API")

model = joblib.load(os.path.join(ROOT, "models", "best.joblib"))
TH = json.load(open(os.path.join(ROOT, "models", "threshold.json")))
TH_BLOCK = float(os.getenv("TH_BLOCK", TH["block"]))
TH_MONITOR = float(os.getenv("TH_MONITOR", TH["monitor"]))
MODE = os.getenv("GUARD_MODE", "block")          # block | monitor | off
MAX_BODY = 64 * 1024
logging.basicConfig(filename="guard.log", level=logging.INFO, format="%(message)s")


def flatten(obj, out=None):
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k)); flatten(v, out)
    elif isinstance(obj, list):
        for v in obj:
            flatten(v, out)
    elif obj is not None:
        out.append(str(obj))
    return out


@app.middleware("http")
async def sqli_guard(request: Request, call_next):
    if MODE == "off":
        return await call_next(request)
    t0 = time.perf_counter()

    body = await request.body()
    if len(body) > MAX_BODY:
        return JSONResponse({"detail": "Payload too large"}, status_code=413)

    inputs = list(request.query_params.values())
    if body:
        try:
            inputs += flatten(json.loads(body))
        except Exception:
            inputs.append(body.decode("utf-8", errors="ignore"))
    inputs = [s for s in inputs if s and s.strip()]

    score, worst = 0.0, None
    if inputs:
        try:
            probs = model.predict_proba(inputs)[:, 1]
            i = int(probs.argmax())
            score, worst = float(probs[i]), inputs[i]
        except Exception as e:                    # fail-open: ghi rõ lựa chọn này trong báo cáo
            logging.info(json.dumps({"ts": time.time(), "error": str(e)}))

    ms = (time.perf_counter() - t0) * 1000
    blocked = score >= TH_BLOCK
    if score >= TH_MONITOR:
        logging.info(json.dumps({
            "ts": time.time(), "ip": request.client.host, "path": request.url.path,
            "score": round(score, 4), "input": worst[:200],
            "action": "block" if (blocked and MODE == "block") else "log",
            "latency_ms": round(ms, 2)}, ensure_ascii=False))
    if 0.05 <= score < TH_MONITOR:
        logging.info(json.dumps({"ts": time.time(), "path": request.url.path,
                                 "score": round(score, 4), "input": (worst or "")[:200],
                                 "action": "pass_low"}, ensure_ascii=False))
    if blocked and MODE == "block":
        return JSONResponse({"detail": "Request blocked"}, status_code=403)
    return await call_next(request)


@app.get("/health")
def health():
    return {"status": "ok", "db": db.BACKEND, "mode": MODE}
@app.get("/ping")
def ping(name: str = ""):
    return {"ok": True}

# CỐ Ý CÓ LỖ HỔNG: nối chuỗi SQL. Chỉ dùng trong lab trên máy bạn.
@app.get("/vuln/products")
def vuln_products(name: str = ""):
    sql = f"SELECT id, name, price FROM products WHERE name LIKE '%{name}%'"
    try:
        return {"rows": db.run(sql)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# AN TOÀN: parameterized query
@app.get("/safe/products")
def safe_products(name: str = ""):
    sql = f"SELECT id, name, price FROM products WHERE name LIKE {db.PH}"
    return {"rows": db.run(sql, (f"%{name}%",))}