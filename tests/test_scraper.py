from pathlib import Path

from boatbot.scraper import (parse_beforeinfo, parse_deadlines, parse_index, parse_odds3t,
                             parse_racelist, parse_result)

FX = Path(__file__).parent / "fixtures"


def html(name):
    return (FX / f"{name}.html").read_text()


def test_index():
    codes = parse_index(html("index"))
    assert "01" in codes and len(codes) >= 10


def test_deadlines():
    d = parse_deadlines(html("racelist"))
    assert d[1] == "15:19" and d[12] == "20:45"


def test_racelist():
    boats = parse_racelist(html("racelist"))
    assert [b.waku for b in boats] == [1, 2, 3, 4, 5, 6]
    b1 = boats[0]
    assert b1.toban == "4303" and b1.grade == "B1" and b1.nat_win == 4.84 and b1.motor_2r == 37.8
    assert boats[2].f_count == 1


def test_beforeinfo():
    bi = parse_beforeinfo(html("beforeinfo"))
    assert bi[1]["ex_time"] == 6.85
    assert bi[1]["course"] == 1 and bi[1]["ex_st"] == -0.03  # F.03
    assert bi[3]["ex_st"] == 0.17


def test_odds3t():
    o = parse_odds3t(html("odds3t"))
    assert len(o) == 120
    assert o["1-2-3"] == 32.8 and o["2-1-3"] == 206.1 and o["1-2-4"] == 51.3
    assert 1.2 < sum(1 / v for v in o.values()) < 1.4  # 控除率25%程度


def test_result():
    assert parse_result(html("raceresult")) == {"combo": "5-1-4", "payout": 5510}
