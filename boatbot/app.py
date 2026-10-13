"""スマホから操作するWeb画面とAPI。

起動: uvicorn boatbot.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import Engine, Notifier, deadline_dt, now_jst
from .model import Settings
from .scraper import VENUES
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STATIC = Path(__file__).parent / "static"
TOKEN = os.environ.get("BOATBOT_TOKEN", "")
TELEBOAT_URL = os.environ.get("TELEBOAT_URL", "https://spweb.brtb.jp/")

store = Store(os.environ.get("BOATBOT_DB", "data/boatbot.db"))
engine = Engine(store, notifier=Notifier(
    topic=os.environ.get("NTFY_TOPIC", ""),
    server=os.environ.get("NTFY_SERVER", "https://ntfy.sh"),
    click_url=os.environ.get("BOATBOT_PUBLIC_URL", ""),
))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.environ.get("BOATBOT_NO_ENGINE") != "1":
        engine.start()
    yield
    engine.stop()


app = FastAPI(title="boatbot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def auth(request: Request) -> None:
    if TOKEN and request.headers.get("x-token") != TOKEN:
        raise HTTPException(401, "token required")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state", dependencies=[Depends(auth)])
def state():
    now = now_jst()
    hd = now.strftime("%Y%m%d")
    s = store.settings()
    spent, races = store.spent(hd)
    venues = s.venue_set()
    upcoming = []
    for row in store.schedule(hd):
        if row["processed"] or (venues and row["jcd"] not in venues):
            continue
        if deadline_dt(hd, row["deadline"]) > now:
            upcoming.append({"jcd": row["jcd"], "venue": VENUES.get(row["jcd"], row["jcd"]),
                             "rno": row["rno"], "deadline": row["deadline"]})
    recs = store.recs(hd)
    for r in recs:
        r["venue"] = VENUES.get(r["jcd"], r["jcd"])
    return {
        "now": now.isoformat(timespec="seconds"),
        "settings": s.__dict__,
        "spent": spent, "races": races,
        "remaining": max(0, s.daily_budget - spent),
        "upcoming": upcoming[:8],
        "recs": recs,
        "venues": VENUES,
        "teleboat_url": TELEBOAT_URL,
        "last_error": engine.last_error,
    }


@app.post("/api/settings", dependencies=[Depends(auth)])
def save_settings(data: dict):
    s = Settings.from_dict({**store.settings().__dict__, **data})
    store.save_settings(s)
    return s.__dict__


class StatusIn(BaseModel):
    status: str


@app.post("/api/recs/{rid}/status", dependencies=[Depends(auth)])
def set_status(rid: int, body: StatusIn):
    if body.status not in ("pending", "bought", "skipped"):
        raise HTTPException(400, "bad status")
    rec = store.rec(rid)
    if not rec or rec["status"] == "pass":
        raise HTTPException(404)
    store.set_status(rid, body.status)
    return store.rec(rid)


class PredictIn(BaseModel):
    jcd: str
    rno: int


@app.post("/api/predict", dependencies=[Depends(auth)])
def predict(body: PredictIn):
    """手動で今すぐ予想（通知なし）"""
    hd = now_jst().strftime("%Y%m%d")
    engine.ensure_schedule(hd)
    dl = next((r["deadline"] for r in store.schedule(hd) if r["jcd"] == body.jcd and r["rno"] == body.rno), None)
    if not dl:
        raise HTTPException(404, "本日そのレースはありません")
    old = next((r for r in store.recs(hd) if r["jcd"] == body.jcd and r["rno"] == body.rno), None)
    if old and old["status"] == "bought":
        raise HTTPException(409, "購入済みのレースは再予想できません")
    if old and old["status"] == "pending":
        store.set_status(old["id"], "skipped")  # 古い推奨の予算を解放してから作り直す
    store.mark_processed(hd, body.jcd, body.rno)
    rec = engine.predict(hd, body.jcd, body.rno, dl, notify=False)
    rec["venue"] = VENUES.get(rec["jcd"], rec["jcd"])
    return rec


@app.get("/api/stats", dependencies=[Depends(auth)])
def stats():
    return store.stats()
