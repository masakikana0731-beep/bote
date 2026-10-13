"""定期実行：今日のスケジュール取得 → 締切前に予想 → 通知 → 結果の取り込み。"""
from __future__ import annotations

import logging
import os
import random
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests

from .executor import AbortPurchase, Executor, PaperExecutor, TeleboatExecutor
from .model import hard_daily_cap, make_plan
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

    def send(self, title: str, body: str, urgent: bool = False) -> None:
        if not self.topic:
            return
        headers = {"Title": title.encode("utf-8"), "Tags": "speedboat", "Priority": "urgent" if urgent else "default"}
        if self.click_url:
            headers["Click"] = self.click_url
        try:
            requests.post(f"{self.server}/{self.topic}", data=body.encode("utf-8"), headers=headers, timeout=10)
        except requests.RequestException as e:
            log.warning("通知失敗: %s", e)


MAX_CONSECUTIVE_FAILURES = 3


class Engine:
    def __init__(self, store: Store, client: BoatraceClient | None = None, notifier: Notifier | None = None,
                 executors: dict[str, Executor] | None = None):
        self.store = store
        self.client = client or BoatraceClient()
        self.notifier = notifier or Notifier()
        self.executors = executors or {
            "paper": PaperExecutor(),
            "real": TeleboatExecutor(flow_path=os.environ.get("TELEBOAT_FLOW", "teleboat_flow.py"),
                                     headless=os.environ.get("TELEBOAT_HEADLESS", "1") != "0"),
        }
        self.failures = 0
        self._predictor = None  # None=未初期化 / False=使えない / LivePredictor
        self._buy_lock = threading.Lock()
        self._stop = threading.Event()
        self.last_error = ""

    # ---- 自動投資
    def stop_auto(self, reason: str) -> None:
        s = self.store.settings()
        if s.auto_mode != "off":
            s.auto_mode = "off"
            self.store.save_settings(s)
            self.notifier.send("⛔ 自動投資を停止しました", reason, urgent=True)
        log.warning("自動投資停止: %s", reason)

    def make_guard(self, rec: dict, mode: str):
        """購入直前に呼ばれる最終チェック。1つでも引っかかれば AbortPurchase（お金は使わない）。"""
        def guard(total: int) -> None:
            s = self.store.settings()
            if s.auto_mode != mode:
                raise AbortPurchase("自動投資が停止/変更されました")
            if total <= 0 or total % 100:
                raise AbortPurchase(f"金額が不正 ({total}円)")
            if total != rec["total"] or total != sum(t["amount"] for t in rec["tickets"]):
                raise AbortPurchase("金額が予定と一致しません")
            if total > s.race_budget:
                raise AbortPurchase(f"1レース上限 {s.race_budget}円 超え")
            spent, races = self.store.spent(rec["hd"], exclude_id=rec["id"])
            if spent + total > s.daily_budget:
                raise AbortPurchase(f"1日上限 {s.daily_budget}円 超え（使用済 {spent}円）")
            if races + 1 > s.max_races_per_day:
                raise AbortPurchase("1日の最大参加レース数に到達")
            cap = hard_daily_cap()
            if cap is not None and self.store.real_spent(rec["hd"]) + total > cap:
                raise AbortPurchase(f"最終上限 BOATBOT_HARD_DAILY_CAP={cap}円 超え")
            if now_jst() > deadline_dt(rec["hd"], rec["deadline"]) - timedelta(seconds=30):
                raise AbortPurchase("締切直前のため中止")
        return guard

    def auto_buy(self, rec: dict) -> dict:
        mode = self.store.settings().auto_mode
        executor = self.executors.get(mode)
        if not executor or not rec["tickets"]:
            return rec
        with self._buy_lock:  # 同時に2つ買わない
            order = {**rec, "venue": VENUES.get(rec["jcd"], rec["jcd"])}
            res = executor.buy(order, self.make_guard(rec, mode))
            self.store.log_buy(rec["id"], now_jst().isoformat(timespec="seconds"), mode,
                               res.ok, res.uncertain, rec["total"], res.message)
            title = f"{order['venue']}{rec['rno']}R"
            lines = "\n".join(f"{t['combo']} {t['amount']}円" for t in rec["tickets"])
            if res.ok:
                self.failures = 0
                self.store.set_status(rec["id"], "bought" if mode == "real" else "paper")
                tag = "🤖 自動購入" if mode == "real" else "🧪 仮想購入"
                self.notifier.send(f"{tag} {title} 計{rec['total']}円", lines)
            elif res.uncertain:
                self.store.set_status(rec["id"], "unknown")
                self.stop_auto(f"{title}: {res.message}\nテレボートの購入履歴を確認してください")
            else:
                self.store.set_status(rec["id"], "failed")
                if not res.message.startswith("中止"):
                    self.failures += 1
                    self.last_error = f"{title}: {res.message}"
                    if self.failures >= MAX_CONSECUTIVE_FAILURES:
                        self.stop_auto(f"{self.failures}回連続で購入に失敗: {res.message}")
                    else:
                        self.notifier.send(f"⚠️ 購入失敗 {title}", res.message)
        return self.store.rec(rec["id"])

    # ---- 1日の準備
    def ensure_schedule(self, hd: str) -> None:
        if self.store.has_schedule(hd):
            return
        try:  # 学習モデル用の過去データを前日分まで更新
            from .history import update
            if update(datetime.strptime(hd, "%Y%m%d").date()):
                log.info("過去データを更新しました")
        except Exception as e:
            log.warning("過去データの更新に失敗: %s", e)
        for jcd in self.client.venues(hd):
            self.store.add_schedule(hd, jcd, self.client.deadlines(jcd, hd))
        log.info("%s のスケジュールを取得しました", hd)

    # ---- 学習モデル
    def ml_probs(self, info, hd: str) -> tuple[dict | None, str]:
        """学習モデルの3連単確率。モデルや過去データが無い・失敗したときは簡易モデルに切り替える"""
        if self._predictor is None:
            try:
                from .ml import LivePredictor
                self._predictor = LivePredictor()
            except Exception as e:
                log.warning("学習モデルを使えません（簡易モデルで予想）: %s", e)
                self._predictor = False
        if not self._predictor:
            return None, "simple"
        try:
            return self._predictor.trifecta(info, hd), "ml"
        except Exception as e:
            log.exception("学習モデルでの予想に失敗")
            self.last_error = f"学習モデル失敗（簡易モデルで予想）: {e}"
            return None, "simple"

    # ---- 1レース予想
    def predict(self, hd: str, jcd: str, rno: int, deadline: str, notify: bool = True, auto: bool = False) -> dict:
        s = self.store.settings()
        spent, races = self.store.spent(hd)
        remaining = max(0, s.daily_budget - spent)
        info = self.client.race(jcd, rno, hd)
        odds = self.client.odds3t(jcd, rno, hd)
        probs, used = self.ml_probs(info, hd) if s.predictor == "ml" else (None, "simple")
        if races >= s.max_races_per_day:
            plan = make_plan(info.boats, odds, s, 0, probs=probs, wind=info.wind)
            plan.tickets, plan.skip_reason = [], "本日の最大参加レース数に到達"
        else:
            plan = make_plan(info.boats, odds, s, remaining, random.Random(), probs=probs, wind=info.wind)
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
            "predictor": used,
        }
        status = "pending" if tickets else "pass"
        rid = self.store.add_rec(hd, jcd, rno, deadline, now_jst().isoformat(timespec="seconds"),
                                 status, tickets, plan.total, detail)
        if auto and tickets and s.auto_mode != "off":
            return self.auto_buy(self.store.rec(rid))
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
            if r["status"] in ("bought", "paper") and returned:
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
                        self.predict(hd, row["jcd"], row["rno"], row["deadline"], auto=True)
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
