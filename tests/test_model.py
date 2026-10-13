import random
from pathlib import Path

from boatbot.model import Settings, Ticket, allocate, make_plan, trifecta_probs, win_probs, strengths
from boatbot.scraper import parse_beforeinfo, parse_odds3t, parse_racelist

FX = Path(__file__).parent / "fixtures"


def race():
    boats = parse_racelist((FX / "racelist.html").read_text())
    bi = parse_beforeinfo((FX / "beforeinfo.html").read_text())
    for b in boats:
        for k, v in bi.get(b.waku, {}).items():
            setattr(b, k, v)
    return boats, parse_odds3t((FX / "odds3t.html").read_text())


def test_trifecta_sums_to_one():
    p = win_probs(strengths(race()[0], Settings()))
    assert abs(sum(p.values()) - 1) < 1e-9
    assert abs(sum(trifecta_probs(p).values()) - 1) < 1e-9


def test_allocate_never_exceeds_budget():
    for budget in (0, 100, 250, 500, 1000, 1234):
        for mode in ("equal", "dutch", "prob"):
            ts = [Ticket(f"1-2-{i}", 0.1, o) for i, o in zip(range(3, 7), (5.0, 12.0, 40.0, 150.0))]
            out = allocate(ts, budget, mode)
            assert sum(t.amount for t in out) <= budget
            assert all(t.amount % 100 == 0 and t.amount >= 100 for t in out)


def test_dutch_equalises_payout():
    ts = allocate([Ticket("a", .1, 5.0), Ticket("b", .1, 10.0)], 1500, "dutch")
    pays = [t.amount * t.odds for t in ts]
    assert max(pays) / min(pays) < 1.3


def test_gamble_slider_raises_odds():
    boats, odds = race()
    avg = []
    for g in (0, 100):
        p = make_plan(boats, odds, Settings(gamble=g, selectivity=0, whim=0), 3000, random.Random(0))
        avg.append(sum(t.odds for t in p.tickets) / len(p.tickets))
    assert avg[1] > avg[0] * 3


def test_plan_respects_race_budget():
    boats, odds = race()
    s = Settings(race_budget=300, tickets=8, selectivity=0)
    p = make_plan(boats, odds, s, 3000, random.Random(0))
    assert 0 < p.total <= 300 and len(p.tickets) == 3


def test_plan_respects_remaining_daily_budget():
    boats, odds = race()
    p = make_plan(boats, odds, Settings(selectivity=0), 50, random.Random(0))
    assert p.total == 0 and p.skip_reason == "予算切れ"
