from datetime import timedelta
from pathlib import Path

import pytest

from boatbot import engine as eng
from boatbot.engine import Engine
from boatbot.executor import BuyResult, Executor, PaperExecutor
from boatbot.model import Settings
from boatbot.scraper import RaceInfo, parse_beforeinfo, parse_odds3t, parse_racelist
from boatbot.store import Store

FX = Path(__file__).parent / "fixtures"
HD = "20261013"


class FakeClient:
    def race(self, jcd, rno, hd):
        boats = parse_racelist((FX / "racelist.html").read_text())
        bi = parse_beforeinfo((FX / "beforeinfo.html").read_text())
        for b in boats:
            for k, v in bi.get(b.waku, {}).items():
                setattr(b, k, v)
        return RaceInfo(jcd, rno, hd, boats)

    def odds3t(self, jcd, rno, hd):
        return parse_odds3t((FX / "odds3t.html").read_text())


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def send(self, title, body, urgent=False):
        self.sent.append((title, urgent))


class ScriptedExecutor(Executor):
    """guard を呼んでから決められた結果を返す"""
    def __init__(self, result: BuyResult):
        self.result, self.calls = result, 0

    def buy(self, order, guard):
        self.calls += 1
        guard(order["total"])
        return self.result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(eng, "now_jst", lambda: eng.deadline_dt(HD, "12:00"))
    monkeypatch.delenv("BOATBOT_HARD_DAILY_CAP", raising=False)
    store = Store(tmp_path / "t.db")
    note = FakeNotifier()

    def make(mode="paper", executor=None, **kw):
        store.save_settings(Settings(auto_mode=mode, selectivity=0, predictor="simple", **kw))
        ex = {"paper": PaperExecutor(), "real": executor or PaperExecutor()}
        return Engine(store, client=FakeClient(), notifier=note, executors=ex)
    return store, note, make


def test_paper_mode_buys_within_budget(setup):
    store, note, make = setup
    e = make("paper", daily_budget=1000, race_budget=500)
    statuses = [e.predict(HD, "01", r, "12:30", auto=True)["status"] for r in range(1, 5)]
    assert statuses[:2] == ["paper", "paper"]
    assert statuses[2:] == ["pass", "pass"]  # 予算切れ
    assert store.spent(HD)[0] <= 1000


def test_off_mode_does_not_buy(setup):
    store, note, make = setup
    e = make("off")
    assert e.predict(HD, "01", 1, "12:30", auto=True)["status"] == "pending"
    assert store.buy_log() == []


def test_guard_blocks_over_budget(setup):
    store, note, make = setup
    e = make("paper", daily_budget=500, race_budget=500)
    # 別レースで既に400円使っている状態を作る
    store.add_rec(HD, "02", 1, "11:00", "", "bought", [{"combo": "1-2-3", "amount": 400}], 400, {})
    rec = e.predict(HD, "01", 1, "12:30", auto=True)
    assert rec["status"] in ("pass", "paper")
    assert store.spent(HD)[0] <= 500


def test_guard_blocks_near_deadline(setup, monkeypatch):
    store, note, make = setup
    e = make("paper")
    monkeypatch.setattr(eng, "now_jst", lambda: eng.deadline_dt(HD, "12:30") - timedelta(seconds=10))
    rec = e.predict(HD, "01", 1, "12:30", auto=True)
    assert rec["status"] == "failed"
    assert "締切直前" in store.buy_log()[0]["message"]


def test_uncertain_stops_auto_mode(setup):
    store, note, make = setup
    ex = ScriptedExecutor(BuyResult(False, "完了画面が出ない", uncertain=True))
    e = make("real", executor=ex)
    rec = e.predict(HD, "01", 1, "12:30", auto=True)
    assert rec["status"] == "unknown"
    assert store.settings().auto_mode == "off"
    assert any(urgent for _, urgent in note.sent)
    # 不明分は予算を使ったものとして数える
    assert store.spent(HD)[0] == rec["total"]
    # 停止後は自動購入しない
    assert e.predict(HD, "01", 2, "12:30", auto=True)["status"] == "pending"
    assert ex.calls == 1


def test_consecutive_failures_stop_auto_mode(setup):
    store, note, make = setup
    ex = ScriptedExecutor(BuyResult(False, "ログイン失敗"))
    e = make("real", executor=ex)
    for r in range(1, 4):
        assert e.predict(HD, "01", r, "12:30", auto=True)["status"] == "failed"
    assert store.settings().auto_mode == "off"
    assert store.spent(HD)[0] == 0  # 失敗分は予算に数えない


def test_hard_cap_env_limits_settings(monkeypatch):
    monkeypatch.setenv("BOATBOT_HARD_DAILY_CAP", "2000")
    s = Settings.from_dict({"daily_budget": 100000, "race_budget": 50000})
    assert s.daily_budget == 2000 and s.race_budget == 2000


def test_real_executor_requires_env(monkeypatch):
    from boatbot.executor import TeleboatExecutor
    monkeypatch.delenv("BOATBOT_ALLOW_REAL", raising=False)
    res = TeleboatExecutor().buy({"total": 100}, lambda t: None)
    assert not res.ok and "BOATBOT_ALLOW_REAL" in res.message
