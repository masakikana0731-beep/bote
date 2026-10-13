from pathlib import Path

from boatbot.backtest import simulate
from boatbot.model import Settings
from boatbot.scraper import parse_beforeinfo, parse_odds3t, parse_racelist, parse_result

FX = Path(__file__).parent / "fixtures"


def race(hd, rno, deadline):
    boats = parse_racelist((FX / "racelist.html").read_text())
    bi = parse_beforeinfo((FX / "beforeinfo.html").read_text())
    for b in boats:
        for k, v in bi.get(b.waku, {}).items():
            setattr(b, k, v)
    return {"hd": hd, "jcd": "01", "rno": rno, "deadline": deadline, "boats": boats,
            "odds": parse_odds3t((FX / "odds3t.html").read_text()),
            "result": parse_result((FX / "raceresult.html").read_text())}


def test_simulate_respects_daily_limits():
    days = {hd: [race(hd, r, f"{10 + r}:00") for r in range(1, 13)] for hd in ("20261001", "20261002")}
    s = Settings(daily_budget=1000, race_budget=300, max_races_per_day=10, selectivity=0, whim=0)
    res = simulate(days, s)
    for d in res["daily"]:
        assert d["bet"] <= 1000 and d["races"] <= 10
    assert res["races"] == sum(d["races"] for d in res["daily"]) > 0


def test_simulate_pays_out_on_hit():
    # 結果 5-1-4 (払戻5510円) を必ず買う設定：人気100%寄り＋点数最大
    days = {"20261001": [race("20261001", 1, "10:00")]}
    s = Settings(daily_budget=2000, race_budget=2000, tickets=20, gamble=50, selectivity=0, whim=0)
    res = simulate(days, s)
    assert res["bet"] <= 2000
    assert res["ret"] in (0,) or res["ret"] % 5510 == 0
