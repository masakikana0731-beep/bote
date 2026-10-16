"""多点買い（流し・ボックス・フォーメーション・券種の組み合わせ）を、モデルの確率だけで選んで採点する。
オッズは使わないので確定オッズの見かけは入らない。払戻は公式の結果。予算の上限なし、1点100円。
期間：2026/7/15〜（テスト期間。展示の進入コースで予想）"""
import json
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from boatbot.ml import VALID_END, Model, add_features, clean_races, load_history, stage_probs

df = clean_races(add_features(load_history()))
te = df[df["hd"] > VALID_END].sort_values(["race_id", "waku"])
m = Model()
S = m.scores(te).reshape(-1, 6)
rid = te["race_id"].to_numpy().reshape(-1, 6)[:, 0]
pays = te.drop_duplicates("race_id").set_index("race_id")["payouts"].map(json.loads)
B = range(1, 7)
TRI = [f"{a}-{b}-{c}" for a, b, c in permutations(B, 3)]

rows = []
for k, r in enumerate(rid):
    P = stage_probs(S[k], m.temps)
    p = {c: P[a - 1, b - 1, cc - 1] for c, (a, b, cc) in zip(TRI, permutations(B, 3))}
    trio, ex, qu, wi, win, pl = {}, {}, {}, {}, {}, {}
    for c, v in p.items():
        a, b, cc = map(int, c.split("-"))
        s3 = "-".join(map(str, sorted((a, b, cc))))
        trio[s3] = trio.get(s3, 0) + v
        ex[f"{a}-{b}"] = ex.get(f"{a}-{b}", 0) + v
        q = f"{min(a, b)}-{max(a, b)}"
        qu[q] = qu.get(q, 0) + v
        for x, y in ((a, b), (a, cc), (b, cc)):
            w = f"{min(x, y)}-{max(x, y)}"
            wi[w] = wi.get(w, 0) + v
        win[str(a)] = win.get(str(a), 0) + v
        pl[str(a)] = pl.get(str(a), 0) + v
        pl[str(b)] = pl.get(str(b), 0) + v
    probs = {"tri": p, "trio": trio, "exacta": ex, "quinella": qu, "wide": wi, "win": win, "place": pl}
    order = [str(x + 1) for x in np.argsort(-S[k])]  # 強い順の艇
    pay = pays[r]

    def ret(t, combos):
        return sum(pay.get(t, {}).get(c, 0) for c in combos), len(combos)

    def top(t, n):
        return sorted(probs[t], key=lambda c: -probs[t][c])[:n]
    strat = {}
    for t, ns in (("tri", (1, 3, 5, 10, 20)), ("trio", (1, 3, 5, 10)), ("exacta", (1, 3, 5, 10)),
                  ("quinella", (1, 3, 5)), ("wide", (1, 2, 3, 5)), ("win", (1, 2)), ("place", (1, 2))):
        for n in ns:
            strat[f"{t} 上位{n}点"] = ret(t, top(t, n))
    f, s2, s3, s4, s5 = order[0], order[1], order[2], order[3], order[4]
    strat["3連単 1着固定-2,3着流し(3艇) 6点"] = ret("tri", [f"{f}-{x}-{y}" for x, y in permutations((s2, s3, s4), 2)])
    strat["3連単 1着固定-2,3着流し(4艇) 12点"] = ret("tri", [f"{f}-{x}-{y}" for x, y in permutations((s2, s3, s4, s5), 2)])
    strat["3連単 1着固定-2着2艇-3着4艇 6点"] = ret("tri", [f"{f}-{x}-{y}" for x in (s2, s3) for y in (s2, s3, s4, s5) if y != x])
    strat["3連単 ボックス3艇 6点"] = ret("tri", ["-".join(c) for c in permutations(order[:3], 3)])
    strat["3連単 ボックス4艇 24点"] = ret("tri", ["-".join(c) for c in permutations(order[:4], 3)])
    strat["3連複 1艇軸-流し4艇 6点"] = ret("trio", ["-".join(sorted((f, x, y), key=int)) for x, y in combinations((s2, s3, s4, s5), 2)])
    strat["3連複 ボックス4艇 4点"] = ret("trio", ["-".join(sorted(c, key=int)) for c in combinations(order[:4], 3)])
    strat["3連複 ボックス5艇 10点"] = ret("trio", ["-".join(sorted(c, key=int)) for c in combinations(order[:5], 3)])
    strat["2連単 1着固定-流し3艇 3点"] = ret("exacta", [f"{f}-{x}" for x in (s2, s3, s4)])
    strat["2連単 ボックス3艇 6点"] = ret("exacta", ["-".join(c) for c in permutations(order[:3], 2)])
    strat["2連複 ボックス3艇 3点"] = ret("quinella", ["-".join(sorted(c, key=int)) for c in combinations(order[:3], 2)])
    strat["拡連複 ボックス3艇 3点"] = ret("wide", ["-".join(sorted(c, key=int)) for c in combinations(order[:3], 2)])
    # いろいろな券種の組み合わせ
    mixes = {
        "組合せA 単勝1+2連複3+3連複3+3連単5 (12点)": [("win", top("win", 1)), ("quinella", top("quinella", 3)), ("trio", top("trio", 3)), ("tri", top("tri", 5))],
        "組合せB 複勝1+拡連複2+3連複5 (8点)": [("place", top("place", 1)), ("wide", top("wide", 2)), ("trio", top("trio", 5))],
        "組合せC 2連単3+3連単10 (13点)": [("exacta", top("exacta", 3)), ("tri", top("tri", 10))],
    }
    for name, parts in mixes.items():
        rr = [ret(t, c) for t, c in parts]
        strat[name] = (sum(x for x, _ in rr), sum(n for _, n in rr))
    for name, (r_, n) in strat.items():
        rows.append((r[:8], name, n * 100, r_))

d = pd.DataFrame(rows, columns=["day", "strat", "bet", "ret"])
out = []
rng = np.random.default_rng(0)
for name, g in d.groupby("strat", sort=False):
    daily = g.groupby("day")[["bet", "ret"]].sum().to_numpy()
    s = daily[rng.integers(0, len(daily), (3000, len(daily)))].sum(1)
    roi = s[:, 1] / s[:, 0]
    half = g["day"] < "20260901"
    out.append({"買い方": name, "1Rの点数": int(g["bet"].iloc[0] // 100), "レース": len(g),
                "的中率": round((g["ret"] > 0).mean(), 3), "回収率": round(g["ret"].sum() / g["bet"].sum(), 3),
                "90%区間": f"{np.percentile(roi, 5):.2f}〜{np.percentile(roi, 95):.2f}",
                "7〜8月": round(g[half]["ret"].sum() / g[half]["bet"].sum(), 3),
                "9〜10月": round(g[~half]["ret"].sum() / g[~half]["bet"].sum(), 3),
                "1日あたり収支(全場全R)": int((g["ret"].sum() - g["bet"].sum()) / g["day"].nunique())})
res = pd.DataFrame(out).sort_values("回収率", ascending=False)
pd.set_option("display.width", 250)
print(f"期間 {d.day.min()}〜{d.day.max()}  {len(rid)} レース")
print(res.to_string(index=False))
res.to_csv("docs/results/multi_bets.csv", index=False, encoding="utf-8-sig")
