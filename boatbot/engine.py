"""定期実行：今日のスケジュール取得 → 締切前に予想 → 通知 → 結果の取り込み。"""
from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from .model import make_plan
from .scraper import VENUES, BoatraceClient
from .store import Store

JST = ZoneInfo("Asia/Tokyo")
log = logging.getLogger("boatbot")


def now_jst() -> datetime:
    return datetime.now(JST)


def deadline_dt(hd: str, hhmm: str) -> datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.strptime(hd, "%Y%m%d").replace(hour=h, minute=m, tzinfo=JST)


class Notifier:
    """ntfy.sh（無料のプッシュ通知アプリ）へ送る。トピック未設定なら何もしない。"""

    def __init__(self, topic: str = "", server: str = "https://ntfy.sh", click_url: str = ""):
        self.topic, self.server, self.click_url = topic, server.rstrip("/"), click_url

    def send(self, title: str, body: str) -> None:
        if not self.topic:
            return
        headers = {"Title": title.encode("utf-8"), "Tags": "speedboat"}
        if self.click_url:
            headers["Click"] = self.click_url
        try:
            requests.post(f"{self.server}/{self.topic}", data=body.encode("utf-8"), headers=headers, timeout=10)
        except requests.RequestException as e:
            log.warning("通知失敗: %s", e)


class Engine:
    def __init__(self, store: Store, client: BoatraceClient | None = None, notifier: Notifier | None = None):
        self.store = store
        self.client = client or BoatraceClient()
        self.notifier = notifier or Notifier()
        self._stop = threading.Event()
        self.last_error = ""

    # ---- 1日の準備
    def ensure_schedule(self, hd: str) -> None:
        if self.store.has_schedule(hd):
            return
        for jcd in self.client.venues(hd):
            self.store.add_schedule(hd, jcd, self.client.deadlines(jcd, hd))
        log.info("%s のスケジュールを取得しました", hd)

    # ---- 1レース予想
    def predict(self, hd: str, jcd: str, rno: int, deadline: str, notify: bool = True) -> dict:
        s = self.store.settings()
        spent, races = self.store.spent(hd)
        remaining = max(0, s.daily_budget - spent)
        info = self.client.race(jcd, rno, hd)
        odds = self.client.odds3t(jcd, rno, hd)
        if races >= s.max_races_per_day:
            plan = make_plan(info.boats, odds, s, 0)
            plan.tickets, plan.skip_reason = [], "本日の最大参加レース数に到達"
        else:
            plan = make_plan(info.boats, odds, s, remaining, random.Random())
            if not plan.tickets and remaining < 100 and not plan.skip_reason.startswith("自信"):
                plan.skip_reason = "本日の予算上限に到達"
        tickets = [{"combo": t.combo, "odds": t.odds, "prob": round(t.prob, 4), "amount": t.amount}
                   for t in plan.tickets]
        detail = {
            "skip_reason": plan.skip_reason,
            "hit_prob": round(plan.hit_prob, 4),
            "avg_ev": round(plan.avg_ev, 3),
            "boats": [{"waku": b.waku, "name": b.name, "grade": b.grade, "nat_win": b.nat_win,
                       "motor_2r": b.motor_2r, "ex_time": b.ex_time, "course": b.course,
                       "win_prob": round((plan.win_probs or {}).get(b.waku, 0), 3)} for b in info.boats],
            "settings": {"gamble": s.gamble, "selectivity": s.selectivity, "crowd": s.crowd},
        }
        status = "pending" if tickets else "pass"
        rid = self.store.add_rec(hd, jcd, rno, deadline, now_jst().isoformat(timespec="seconds"),
                                 status, tickets, plan.total, detail)
        if notify and tickets:
            name = VENUES.get(jcd, jcd)
            lines = [f"{t['combo']}  {t['amount']}円 ({t['odds']}倍)" for t in tickets]
            self.notifier.send(f"{name}{rno}R 締切{deadline} 計{plan.total}円",
                               "3連単\n" + "\n".join(lines) + f"\n的中率(推定) {plan.hit_prob:.0%}")
        return self.store.rec(rid)

    # ---- 結果取り込み
    def settle(self) -> None:
        now = now_jst()
        for r in self.store.awaiting_result():
            if now < deadline_dt(r["hd"], r["deadline"]) + timedelta(minutes=10):
                continue
            res = self.client.result(r["jcd"], r["rno"], r["hd"])
            if not res:
                # 中止・返還などで結果が出ない場合、翌日以降は諦める
                if now.strftime("%Y%m%d") > r["hd"]:
                    self.store.set_result(r["id"], "-", 0, 0)
                continue
            returned = sum(t["amount"] // 100 * res["payout"] for t in r["tickets"] if t["combo"] == res["combo"])
            self.store.set_result(r["id"], res["combo"], res["payout"], returned)
            if r["status"] == "bought" and returned:
                self.notifier.send(f"🎉 {VENUES.get(r['jcd'], '')}{r['rno']}R 的中！",
                                   f"{res['combo']} 払戻 {returned:,}円")

    # ---- ループ
    def tick(self) -> None:
        now = now_jst()
        hd = now.strftime("%Y%m%d")
        s = self.store.settings()
        if s.enabled and 7 <= now.hour < 22:
            self.ensure_schedule(hd)
            venues = s.venue_set()
            for row in self.store.schedule(hd):
                if row["processed"] or (venues and row["jcd"] not in venues):
                    continue
                dl = deadline_dt(hd, row["deadline"])
                if dl - timedelta(minutes=s.lead_minutes) <= now < dl - timedelta(minutes=1):
                    self.store.mark_processed(hd, row["jcd"], row["rno"])
                    try:
                        self.predict(hd, row["jcd"], row["rno"], row["deadline"])
                    except Exception as e:  # 1レースの失敗で全体を止めない
                        log.exception("予想失敗 %s %sR", row["jcd"], row["rno"])
                        self.last_error = f"{row['jcd']} {row['rno']}R: {e}"
        self.settle()

    def run_forever(self, interval: int = 30) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                log.exception("tick失敗")
                self.last_error = str(e)
            self._stop.wait(interval)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
