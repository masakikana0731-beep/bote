"""券種・単位・配分まで含めた買い方の最適化。

  python -m boatbot.optimize2 prepare
  python -m boatbot.optimize2 search --n 20000 --jobs 4

評価方法:
  - 前半（〜SPLIT）だけで最良の設定を選び、後半でそのまま使った成績を「本当の実力」として報告する
  - 比較用に「ランダムに買う」戦略も同じ条件で評価する
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

PREP = Path("data/opt2_races.npz")
SPLIT = "20260828"
BOATS = range(1, 7)

# 券種ごとの組番一覧
COMBOS = {
    "tri": [f"{a}-{b}-{c}" for a, b, c in permutations(BOATS, 3)],
    "trio": [f"{a}-{b}-{c}" for a, b, c in combinations(BOATS, 3)],
    "exacta": [f"{a}-{b}" for a, b in permutations(BOATS, 2)],
    "quinella": [f"{a}-{b}" for a, b in combinations(BOATS, 2)],
    "wide": [f"{a}-{b}" for a, b in combinations(BOATS, 2)],
    "win": [str(a) for a in BOATS],
    "place": [str(a) for a in BOATS],
}
TYPE_NAMES = {"tri": "3連単", "trio": "3連複", "exacta": "2連単", "quinella": "2連複",
              "wide": "拡連複", "win": "単勝", "place": "複勝", "all": "全券種から自動"}
# ギャンブル性 0〜100 → 狙う配当の目安（券種ごとの範囲）
TARGET_RANGE = {"tri": (8, 300), "trio": (3, 100), "exacta": (3, 100), "quinella": (2, 50),
                "wide": (1.3, 20), "win": (1.3, 20), "place": (1.1, 5), "all": (1.5, 300)}


def type_probs(P: np.ndarray) -> dict[str, np.ndarray]:
    """3連単確率 (120,) → 各券種の的中確率"""
    T = {c: p for c, p in zip(COMBOS["tri"], P)}
    out = {"tri": P}
    out["trio"] = np.array([sum(T[f"{x}-{y}-{z}"] for x, y, z in permutations(map(int, c.split("-"))))
                            for c in COMBOS["trio"]])
    ex = {f"{a}-{b}": sum(T[f"{a}-{b}-{c}"] for c in BOATS if c not in (a, b)) for a, b in permutations(BOATS, 2)}
    out["exacta"] = np.array([ex[c] for c in COMBOS["exacta"]])
    out["quinella"] = np.array([ex[c] + ex["-".join(reversed(c.split("-")))] for c in COMBOS["quinella"]])
    out["wide"] = np.array([sum(p for k, p in T.items() if c[0] in k and c[2] in k) for c in COMBOS["wide"]])
    out["win"] = np.array([sum(p for k, p in T.items() if k[0] == str(a)) for a in BOATS])
    out["place"] = np.array([sum(p for k, p in T.items() if str(a) in k[:3]) for a in BOATS])
    return out


# ---------------------------------------------------------------- 準備

def prepare() -> None:
    import pandas as pd
    base = dict(np.load("data/opt_races.npz"))
    hist = pd.read_parquet("data/history.parquet", columns=["hd", "jcd", "rno", "payouts"])
    hist["race_id"] = hist["hd"] + hist["jcd"] + hist["rno"].astype(str).str.zfill(2)
    pays = hist.drop_duplicates("race_id").set_index("race_id")["payouts"]
    keep, data = [], {t: {"P": [], "O": [], "R": []} for t in COMBOS}
    for i, rid in enumerate(base["race_id"]):
        f = Path("data/odds_multi") / rid[:8] / f"{rid[8:10]}_{rid[10:]}.json"
        if not f.exists() or rid not in pays.index:
            continue
        om = json.loads(f.read_text())
        om["tri"] = {c: o for c, o in zip(COMBOS["tri"], base["odds"][i])}
        pay = json.loads(pays[rid])
        if any(len(om.get(t, {})) < len(COMBOS[t]) * 0.8 for t in COMBOS) or len(pay) < 7:
            continue
        probs = type_probs(base["P"][i])
        for t in COMBOS:
            data[t]["P"].append(probs[t])
            data[t]["O"].append([om[t].get(c, 0.0) for c in COMBOS[t]])
            data[t]["R"].append([pay.get(t, {}).get(c, 0) for c in COMBOS[t]])
        keep.append(i)
    out = {k: base[k][keep] for k in ("race_id", "hd", "deadline", "wind", "p1max")}
    for t in COMBOS:
        for k in ("P", "O", "R"):
            out[f"{t}_{k}"] = np.array(data[t][k], dtype=float)
    np.savez(PREP, **out)
    print(f"{PREP}: {len(keep)} レース")


# ---------------------------------------------------------------- 戦略

@dataclass
class Strategy:
    bet_type: str = "tri"     # tri/trio/exacta/quinella/wide/win/place/all
    gamble: float = 30        # 狙う配当の目安（券種ごとの範囲内で 0〜100）
    sigma: float = 0.7        # 目安からの外れ許容（log）
    tickets: int = 5          # 最大点数
    min_ev: float = 0.0       # 買い目ごとの期待値の下限
    crowd: float = 0.0        # 人気(オッズ)を信じる度合い
    staking: str = "dutch"    # equal / dutch(払戻を揃える) / prob(確率比例) / kelly(優位性に比例)
    unit: int = 100           # 1口の金額（100/200/500）
    race_budget: int = 500
    skip_fav: float = 1.0     # 1着確率がこれを超える艇がいれば見送り
    max_wind: float = 99
    daily_budget: int = 3000
    max_races: int = 10


def target(t: str, g: float) -> float:
    lo, hi = TARGET_RANGE[t]
    return lo * (hi / lo) ** (g / 100)


def _candidates(d: dict, i: int, s: Strategy):
    """(券種, 組番index, 確率, オッズ, 払戻) の候補を、スコアの高い順に"""
    types = list(COMBOS) if s.bet_type == "all" else [s.bet_type]
    center = math.log(target(s.bet_type, s.gamble))
    cands = []
    for t in types:
        p, o, r = d[f"{t}_P"][i], d[f"{t}_O"][i], d[f"{t}_R"][i]
        valid = o > 0
        if not valid.any():
            continue
        if s.crowd > 0:
            q = np.where(valid, 1 / np.where(valid, o, 1), 0)
            q = q / q.sum() * p[valid].sum()  # 複勝・拡連複は確率の合計が1でないので合わせる
            p = np.where(valid, np.maximum(p, 1e-12) ** (1 - s.crowd) * np.maximum(q, 1e-12) ** s.crowd, 0)
        ev = p * o
        ok = valid & (ev >= s.min_ev) & (p > 0)
        lo = np.log(np.where(valid, o, 1))
        score = np.log(np.maximum(p, 1e-12)) - (lo - center) ** 2 / (2 * s.sigma ** 2)
        for k in np.where(ok)[0]:
            cands.append((score[k], t, k, p[k], o[k], r[k]))
    cands.sort(key=lambda x: -x[0])
    return cands[: s.tickets]


def simulate(d: dict, s: Strategy, mask: np.ndarray | None = None, with_bets: bool = False) -> dict:
    hd = d["hd"]
    idx = np.arange(len(hd)) if mask is None else np.where(mask)[0]
    bets, cur, spent, races = [], None, 0, 0
    for i in idx:
        if hd[i] != cur:
            cur, spent, races = hd[i], 0, 0
        if races >= s.max_races or d["wind"][i] >= s.max_wind or d["p1max"][i] > s.skip_fav:
            continue
        picks = _candidates(d, i, s)
        budget = min(s.race_budget, s.daily_budget - spent)
        units = budget // s.unit
        picks = picks[:units]
        if not picks:
            continue
        p = np.array([x[3] for x in picks])
        o = np.array([x[4] for x in picks])
        if s.staking == "equal":
            w = np.ones(len(picks))
        elif s.staking == "prob":
            w = p
        elif s.staking == "kelly":
            w = np.maximum(0, (p * o - 1) / np.maximum(o - 1, 1e-6))
            if w.sum() <= 0:
                continue
            keep = w > 0
            picks = [x for x, k in zip(picks, keep) if k]
            w, p, o = w[keep], p[keep], o[keep]
        else:
            w = 1 / o
        rest = units - len(picks)
        raw = rest * w / w.sum()
        u = 1 + np.floor(raw).astype(int)
        u[np.argsort(-(raw - np.floor(raw)))[: units - u.sum()]] += 1
        amount = u * s.unit
        total = int(amount.sum())
        ret = int(sum(a // 100 * x[5] for x, a in zip(picks, amount)))
        spent += total
        races += 1
        bets.append((hd[i], total, ret, picks[0][1]))
    out = summarize(bets)
    if with_bets:
        out["bets"] = bets
    return out


def summarize(bets) -> dict:
    if not bets:
        return {"races": 0, "bet": 0, "ret": 0, "profit": 0, "roi": 0.0, "hit": 0.0, "days": 0,
                "plus_days": 0, "dd": 0, "streak": 0}
    bet = sum(b[1] for b in bets)
    ret = sum(b[2] for b in bets)
    days: dict = {}
    cum = peak = dd = st = worst = 0
    for d, b, r, _ in bets:
        days[d] = days.get(d, 0) + r - b
        cum += r - b
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        st = 0 if r else st + 1
        worst = max(worst, st)
    return {"races": len(bets), "bet": bet, "ret": ret, "profit": ret - bet, "roi": ret / bet,
            "hit": sum(1 for b in bets if b[2]) / len(bets), "days": len(days),
            "plus_days": sum(1 for v in days.values() if v > 0), "dd": dd, "streak": worst}


def random_strategy(rng: random.Random) -> Strategy:
    return Strategy(
        bet_type=rng.choice(list(COMBOS) + ["all"]),
        gamble=rng.choice(range(0, 101, 10)),
        sigma=rng.choice([0.3, 0.5, 0.7, 1.0, 1.5, 3.0]),
        tickets=rng.choice([1, 2, 3, 4, 5, 6, 8, 10]),
        min_ev=rng.choice([0.0, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]),
        crowd=rng.choice([0.0, 0.2, 0.4, 0.6, 0.8]),
        staking=rng.choice(["equal", "dutch", "prob", "kelly"]),
        unit=rng.choice([100, 200, 500]),
        race_budget=rng.choice([200, 300, 500, 1000, 1500]),
        skip_fav=rng.choice([1.0, 0.8, 0.7, 0.6, 0.5]),
        max_wind=rng.choice([99, 6, 4]),
    )


class RandomBetter:
    """比較用：毎レース、ランダムな券種・組番を1点100円×5点買う"""


def random_baseline(d: dict, mask: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bets, cur, spent, races = [], None, 0, 0
    for i in np.where(mask)[0]:
        if d["hd"][i] != cur:
            cur, spent, races = d["hd"][i], 0, 0
        if races >= 6:
            continue
        t = rng.choice(list(COMBOS))
        o = d[f"{t}_O"][i]
        ks = rng.choice(np.where(o > 0)[0], size=min(5, int((o > 0).sum())), replace=False)
        ret = int(sum(d[f"{t}_R"][i][k] for k in ks))
        bets.append((d["hd"][i], 100 * len(ks), ret, t))
        races += 1
    return summarize(bets)


# ---------------------------------------------------------------- 探索

_D = None


def _init():
    global _D
    _D = dict(np.load(PREP))


def _eval(s: Strategy):
    first = _D["hd"] <= SPLIT
    return asdict(s), simulate(_D, s, first), simulate(_D, s, ~first)


def search(n: int, seed: int, jobs: int, out: Path) -> None:
    from multiprocessing import Pool
    rng = random.Random(seed)
    cands = [Strategy()] + [random_strategy(rng) for _ in range(n)]
    with Pool(jobs, initializer=_init) as pool:
        res = []
        for k, r in enumerate(pool.imap(_eval, cands, chunksize=20)):
            res.append(r)
            if k % 2000 == 0:
                print(f"  {k}/{len(cands)}", flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False))
    print(f"保存しました {out}（{len(res)}件）")


# ---------------------------------------------------------------- 報告

OBJECTIVES = {
    # 名前: (説明, 前半での条件, 前半での並べ替えキー)
    "回収率重視": ("回収率が最大", lambda r: r["races"] >= 100, lambda r: r["roi"]),
    "当たる楽しさ": ("的中率25%以上の中で回収率最大", lambda r: r["races"] >= 100 and r["hit"] >= 0.25, lambda r: r["roi"]),
    "一発狙い": ("的中率5〜15%の中で回収率最大", lambda r: r["races"] >= 100 and 0.05 <= r["hit"] < 0.15, lambda r: r["roi"]),
}


def bootstrap(bets, n: int = 5000, seed: int = 0) -> dict:
    """日単位でリサンプリングした回収率の90%区間とプラスになる確率"""
    days: dict = {}
    for d, b, r, _ in bets:
        x = days.setdefault(d, [0, 0])
        x[0] += b
        x[1] += r
    arr = np.array(list(days.values()), dtype=float)
    if len(arr) == 0:
        return {"lo": 0.0, "hi": 0.0, "p_plus": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    s = arr[idx].sum(axis=1)
    roi = s[:, 1] / np.maximum(s[:, 0], 1)
    return {"lo": float(np.percentile(roi, 5)), "hi": float(np.percentile(roi, 95)),
            "p_plus": float((roi > 1).mean())}


def _verify(d: dict, s: Strategy, second: np.ndarray) -> dict:
    v = simulate(d, s, second, with_bets=True)
    bets = v.pop("bets")
    v["boot"] = bootstrap(bets)
    v["roi_wo_top2"] = (v["ret"] - sum(b[2] for b in sorted(bets, key=lambda b: -b[2])[:2])) / v["bet"] if v["bet"] else 0
    return v


def report(results_path: Path, out: Path, top_k: int = 20) -> None:
    d = dict(np.load(PREP))
    first = d["hd"] <= SPLIT
    second = ~first
    res = json.loads(results_path.read_text())
    rep = {"split": SPLIT, "n": len(res), "explore_races": int(first.sum()),
           "verify_races": int(second.sum()), "objectives": {}, "by_type": {}, "baselines": {}}
    for name, (desc, cond, key) in OBJECTIVES.items():
        ok = sorted((r for r in res if cond(r[1])), key=lambda r: -key(r[1]))
        top = ok[:top_k]
        best = top[0]
        rep["objectives"][name] = {
            "desc": desc, "params": best[0], "explore": best[1],
            "verify": _verify(d, Strategy(**best[0]), second),
            # 前半の上位 top_k 件が後半でどうだったか（選択バイアスの目安）
            "topk_verify_roi_mean": float(np.mean([r[2]["roi"] for r in top])),
            "topk_explore_roi_mean": float(np.mean([r[1]["roi"] for r in top])),
        }
    for t in list(COMBOS) + ["all"]:
        ok = sorted((r for r in res if r[0]["bet_type"] == t and r[1]["races"] >= 100), key=lambda r: -r[1]["roi"])
        if not ok:
            continue
        best = ok[0]
        rep["by_type"][t] = {"params": best[0], "explore": best[1],
                             "verify": _verify(d, Strategy(**best[0]), second),
                             "all_verify_roi_mean": float(np.mean([r[2]["roi"] for r in ok]))}
    rb = [random_baseline(d, second, seed) for seed in range(20)]
    rep["baselines"]["random"] = {"verify_roi_mean": float(np.mean([r["roi"] for r in rb])),
                                  "verify_roi_min": float(min(r["roi"] for r in rb)),
                                  "verify_roi_max": float(max(r["roi"] for r in rb)),
                                  "verify_hit_mean": float(np.mean([r["hit"] for r in rb]))}
    for t in COMBOS:
        rep["baselines"][f"market_{t}"] = _verify(d, Strategy(bet_type=t, crowd=1.0, gamble=0, tickets=3), second)
    rep["baselines"]["default"] = _verify(d, Strategy(), second)
    # パラメータごとの効き方（後半・全候補の平均回収率）
    eff = {}
    for field in ("bet_type", "gamble", "sigma", "tickets", "min_ev", "crowd", "staking", "unit",
                  "race_budget", "skip_fav", "max_wind"):
        vals: dict = {}
        for p, _, v in res:
            if v["races"] >= 50:
                vals.setdefault(p[field], []).append(v["roi"])
        eff[field] = {str(k): round(float(np.mean(x)), 3) for k, x in sorted(vals.items(), key=lambda kv: str(kv[0]))}
    rep["param_effect_verify"] = eff
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"保存しました {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    rp = sub.add_parser("report")
    rp.add_argument("--results", default="data/opt2_results.json")
    rp.add_argument("--out", default="docs/results/opt2_report.json")
    s = sub.add_parser("search")
    s.add_argument("--n", type=int, default=20000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--jobs", type=int, default=4)
    s.add_argument("--out", default="data/opt2_results.json")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "report":
        report(Path(a.results), Path(a.out))
    else:
        search(a.n, a.seed, a.jobs, Path(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
