"""買い方パラメータの最適化（学習済みモデル × 実オッズ × 実結果）。

  python -m boatbot.optimize prepare            # テスト期間のレースごとに確率・オッズ・結果をまとめる
  python -m boatbot.optimize search --n 3000    # ランダムサーチ（前半で探索→後半で検証）

過学習を避けるため、テスト期間を前半(探索用)と後半(検証用)に分け、
前半で選んだパラメータを後半でそのまま使ったときの成績を「本当の実力」として見る。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np

COMBOS = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
CIDX = {c: i for i, c in enumerate(COMBOS)}
PREP = Path("data/opt_races.npz")
SPLIT = "20260828"  # 〜この日: 探索用 / 翌日〜: 検証用


@dataclass
class Strategy:
    gamble: float = 30        # 狙う配当の目安（0=8倍〜100=300倍）
    sigma: float = 0.7        # 目安からどれだけ外れてよいか（log幅）
    tickets: int = 5          # 最大点数
    min_ev: float = 0.0       # 買い目ごとの期待値(確率×オッズ)の下限。未満は買わない
    crowd: float = 0.0        # 人気(オッズ)をどれだけ信じるか 0〜1
    allocation: str = "dutch"
    skip_fav: float = 1.0     # 1号艇などの1着確率がこれを超える「堅いレース」は見送る（1.0=見送らない）
    max_wind: float = 99      # 風速がこれ以上なら見送る(m)
    race_budget: int = 500
    daily_budget: int = 3000
    max_races: int = 10


# ---------------------------------------------------------------- 準備

def prepare() -> None:
    import pandas as pd
    from .ml import VALID_END, Model, add_features, clean_races, load_history, stage_probs
    df = clean_races(add_features(load_history()))
    te = df[df["hd"] > VALID_END].sort_values(["race_id", "waku"])
    m = Model()
    te = te.assign(score=m.scores(te))
    rows = {k: [] for k in ("race_id", "hd", "deadline", "P", "odds", "res", "payout", "wind", "p1max")}
    for rid, g in te.groupby("race_id", sort=False):
        hd, jcd, rno = rid[:8], rid[8:10], int(rid[10:])
        f = Path("data/odds") / hd / f"{jcd}_{rno:02d}.json"
        if not f.exists():
            continue
        odds = json.loads(f.read_text())
        if len(odds) < 100:
            continue
        P = stage_probs(g["score"].to_numpy(), m.temps)
        first = g.iloc[0]
        if not isinstance(first["tri_combo"], str):
            continue
        rows["race_id"].append(rid)
        rows["hd"].append(hd)
        rows["deadline"].append(first["deadline"])
        rows["P"].append([P[int(c[0]) - 1, int(c[2]) - 1, int(c[4]) - 1] for c in COMBOS])
        rows["odds"].append([odds.get(c, 0.0) for c in COMBOS])
        rows["res"].append(CIDX[first["tri_combo"]])
        rows["payout"].append(first["tri_payout"])
        rows["wind"].append(first["wind"])
        p1 = np.exp(m.temps[0] * (g["score"] - g["score"].max()))
        rows["p1max"].append(float((p1 / p1.sum()).max()))
    order = np.lexsort((np.array(rows["deadline"]), np.array(rows["hd"])))
    np.savez(PREP, **{k: np.array(v)[order] for k, v in rows.items()})
    print(f"{PREP}: {len(order)} レース")


# ---------------------------------------------------------------- シミュレーション

def target_odds(g: float) -> float:
    return 8.0 * (300 / 8.0) ** (g / 100)


def simulate(data: dict, s: Strategy, mask: np.ndarray | None = None, with_bets: bool = False) -> dict:
    hd, P, O, res, pay = data["hd"], data["P"], data["odds"], data["res"], data["payout"]
    idx = np.arange(len(hd)) if mask is None else np.where(mask)[0]
    center = math.log(target_odds(s.gamble))
    bets, cur_day, spent, races = [], None, 0, 0
    for i in idx:
        if hd[i] != cur_day:
            cur_day, spent, races = hd[i], 0, 0
        if races >= s.max_races or data["wind"][i] >= s.max_wind or data["p1max"][i] > s.skip_fav:
            continue
        o = O[i]
        valid = o > 0
        p = P[i]
        if s.crowd > 0:
            q = np.where(valid, 1 / np.where(valid, o, 1), 0)
            q /= q.sum()
            p = np.where(valid, p ** (1 - s.crowd) * np.where(valid, q, 1) ** s.crowd, 0)
            p /= p.sum()
        ev = p * o
        ok = valid & (ev >= s.min_ev)
        if not ok.any():
            continue
        lo = np.log(np.where(valid, o, 1))
        score = np.where(ok, np.log(np.maximum(p, 1e-12)) - (lo - center) ** 2 / (2 * s.sigma ** 2), -np.inf)
        pick = np.argsort(-score)[: s.tickets]
        pick = pick[np.isfinite(score[pick])]
        budget = min(s.race_budget, s.daily_budget - spent)
        units = budget // 100
        pick = pick[:units]
        if len(pick) == 0:
            continue
        if s.allocation == "equal":
            w = np.ones(len(pick))
        elif s.allocation == "prob":
            w = p[pick]
        else:
            w = 1 / o[pick]
        rest = units - len(pick)
        raw = rest * w / w.sum()
        u = 1 + np.floor(raw).astype(int)
        left = units - u.sum()
        u[np.argsort(-(raw - np.floor(raw)))[:left]] += 1
        amount = u * 100
        total = int(amount.sum())
        ret = int(sum(a // 100 * pay[i] for c, a in zip(pick, amount) if c == res[i]))
        spent += total
        races += 1
        bets.append((hd[i], total, ret))
    out = summarize(bets)
    if with_bets:
        out["bets"] = bets
    return out


def summarize(bets) -> dict:
    if not bets:
        return {"races": 0, "bet": 0, "ret": 0, "roi": 0.0, "hit": 0.0, "days": 0, "plus_days": 0, "dd": 0, "streak": 0}
    bet = sum(b for _, b, _ in bets)
    ret = sum(r for _, _, r in bets)
    days: dict = {}
    cum = peak = dd = st = worst = 0
    for d, b, r in bets:
        days[d] = days.get(d, 0) + r - b
        cum += r - b
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        st = 0 if r else st + 1
        worst = max(worst, st)
    return {"races": len(bets), "bet": bet, "ret": ret, "profit": ret - bet, "roi": ret / bet,
            "hit": sum(1 for *_, r in bets if r) / len(bets), "days": len(days),
            "plus_days": sum(1 for v in days.values() if v > 0), "dd": dd, "streak": worst}


def random_strategy(rng: random.Random) -> Strategy:
    return Strategy(
        gamble=rng.choice(range(0, 101, 10)),
        sigma=rng.choice([0.4, 0.7, 1.0, 1.5, 3.0]),
        tickets=rng.choice([1, 2, 3, 4, 5, 6, 8]),
        min_ev=rng.choice([0.0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]),
        crowd=rng.choice([0.0, 0.2, 0.4, 0.6]),
        allocation=rng.choice(["dutch", "equal", "prob"]),
        skip_fav=rng.choice([1.0, 0.7, 0.6, 0.5]),
        max_wind=rng.choice([99, 6, 4]),
        race_budget=rng.choice([300, 500, 1000]),
    )


OBJECTIVES = {
    # 名前: (説明, 条件, 並べ替えキー)
    "回収率重視": ("回収率が最大", lambda r: r["races"] >= 100, lambda r: r["roi"]),
    "当たる楽しさ": ("的中率25%以上の中で回収率最大", lambda r: r["races"] >= 100 and r["hit"] >= 0.25, lambda r: r["roi"]),
    "一発狙い": ("的中率5〜15%の中で回収率最大", lambda r: r["races"] >= 100 and 0.05 <= r["hit"] < 0.15, lambda r: r["roi"]),
    "安定": ("最大下落が小さい中で回収率最大", lambda r: r["races"] >= 100, lambda r: r["roi"] - r["dd"] / 100000),
}


def search(n: int, seed: int, out: Path) -> None:
    data = dict(np.load(PREP))
    first = data["hd"] <= SPLIT
    second = ~first
    rng = random.Random(seed)
    cands = [Strategy()] + [random_strategy(rng) for _ in range(n)]
    results = []
    for k, s in enumerate(cands):
        results.append((s, simulate(data, s, first)))
        if k % 500 == 0:
            print(f"  {k}/{len(cands)}", flush=True)
    report = {"split": SPLIT, "n": len(cands),
              "explore_races": int(first.sum()), "verify_races": int(second.sum()), "objectives": {}}
    base = Strategy()
    report["default"] = {"params": asdict(base), "explore": simulate(data, base, first),
                         "verify": simulate(data, base, second)}
    market = Strategy(crowd=1.0)
    report["market_only"] = {"params": asdict(market), "explore": simulate(data, market, first),
                             "verify": simulate(data, market, second)}
    for name, (desc, cond, key) in OBJECTIVES.items():
        ok = [(s, r) for s, r in results if cond(r)]
        ok.sort(key=lambda x: -key(x[1]))
        top = []
        for s, r in ok[:5]:
            top.append({"params": asdict(s), "explore": r, "verify": simulate(data, s, second)})
        report["objectives"][name] = {"desc": desc, "top": top}
    # パラメータごとの効き方（探索期間・全候補の平均回収率）
    eff = {}
    for field in ("gamble", "sigma", "tickets", "min_ev", "crowd", "allocation", "skip_fav", "max_wind", "race_budget"):
        vals: dict = {}
        for s, r in results:
            if r["races"] >= 50:
                vals.setdefault(getattr(s, field), []).append(r["roi"])
        eff[field] = {str(k): round(float(np.mean(v)), 3) for k, v in sorted(vals.items())}
    report["param_effect"] = eff
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"保存しました {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    s = sub.add_parser("search")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/opt_report.json")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    else:
        search(a.n, a.seed, Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
