from itertools import permutations

from boatbot.insight import boat_insight, market_probs


def _tri(win):
    """艇ごとの強さ → 3連単確率（Harville）"""
    out = {}
    for a, b, c in permutations(range(1, 7), 3):
        pa = win[a]
        pb = win[b] / (1 - win[a])
        pc = win[c] / (1 - win[a] - win[b])
        out[f"{a}-{b}-{c}"] = pa * pb * pc
    return out


def test_market_probs_sum_to_one():
    q = market_probs({"1-2-3": 2.0, "1-3-2": 4.0, "2-1-3": 0})
    assert abs(sum(q.values()) - 1) < 1e-9 and "2-1-3" not in q


def test_boat_insight_diff_and_solid():
    model = _tri({1: 0.97, 2: 0.01, 3: 0.01, 4: 0.004, 5: 0.003, 6: 0.003})
    market = _tri({1: 0.5, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.05, 6: 0.05})
    odds = {k: 0.75 / v for k, v in market.items()}
    ins = boat_insight(model, odds)
    b1 = ins["boats"][0]
    assert abs(b1["model_win"] - 0.97) < 1e-3 and abs(b1["market_win"] - 0.5) < 1e-3
    assert b1["notable"] and not ins["boats"][1]["notable"]
    assert {(s["waku"], s["type"]) for s in ins["solid"]} == {(1, "複勝"), (1, "単勝")}


def test_boat_insight_without_model():
    ins = boat_insight(None, {"1-2-3": 3.0}, model_win={1: 0.6})
    assert ins["boats"][0]["model_win"] == 0.6 and ins["solid"] == []
