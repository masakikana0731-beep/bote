"""レースを選ぶ（捨てる）＋オッズ帯・期待値で買う目を絞るルールを、前半(7/15〜8/31)で選び後半(9/1〜)で検証する。
3連単：確定オッズあり（全レース）。単勝・複勝・2連単・2連複・3連複・拡連複：オッズなし（モデルの確率だけ）"""
import json
from itertools import permutations

import numpy as np
import pandas as pd

from boatbot.ml import VALID_END, Model, add_features, clean_races, load_history, stage_probs

SPLIT = "20260831"
df = clean_races(add_features(load_history()))
te = df[df["hd"] > VALID_END].sort_values(["race_id", "waku"])
m = Model()
S = m.scores(te).reshape(-1, 6)
rid = te["race_id"].to_numpy().reshape(-1, 6)[:, 0]
pays = te.drop_duplicates("race_id").set_index("race_id")["payouts"].map(json.loads)
TRI = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
IDX = [(a - 1, b - 1, c - 1) for a, b, c in permutations(range(1, 7), 3)]

keep, P, O, R = [], [], [], []
for k, r in enumerate(rid):
    f = f"data/odds/{r[:8]}/{r[8:10]}_{r[10:]}.json"
    try:
        o = json.load(open(f))
    except FileNotFoundError:
        continue
    if len(o) < 100:
        continue
    pm = stage_probs(S[k], m.temps)
    P.append([pm[a, b, c] for a, b, c in IDX])
    O.append([o.get(c, 0.0) for c in TRI])
    R.append([pays[r].get("tri", {}).get(c, 0) for c in TRI])
    keep.append(k)
P, O, R = np.array(P), np.array(O, float), np.array(R, float)
rids = rid[keep]
day = np.array([r[:8] for r in rids])
first = day <= SPLIT
q = np.where(O > 0, 1 / np.where(O > 0, O, 1), 0)
Q = q / q.sum(1, keepdims=True)
# 艇ごと：モデルの1着確率、市場の1着確率
pw = np.zeros((len(P), 6)); qw = np.zeros((len(P), 6))
for j, (a, b, c) in enumerate(IDX):
    pw[:, a] += P[:, j]; qw[:, a] += Q[:, j]
top = pw.argmax(1)
p1 = pw.max(1)
edge1 = p1 - qw[np.arange(len(P)), top]          # モデルの本命について、モデル − 市場
mkt_fav_q = qw.max(1)
print("レース", len(P), "前半", int(first.sum()), "後半", int((~first).sum()), flush=True)

rank = np.argsort(-P, 1)
EV = P * O


def evaluate(race_ok, n, lo, hi, e):
    sel = np.zeros_like(P, bool)
    cols = rank[:, :n]
    rows = np.repeat(np.arange(len(P)), n)
    sel[rows, cols.ravel()] = True
    sel &= race_ok[:, None] & (O >= lo) & (O < hi) & (EV >= e) & (O > 0)
    out = {}
    for name, mm in (("前半", first), ("後半", ~first)):
        s = sel[mm]
        bet = s.sum() * 100
        ret = (s * R[mm]).sum()
        races = int(s.any(1).sum())
        hits = int((s * (R[mm] > 0)).any(1).sum())
        out[name] = (bet, ret, races, hits)
    return out


filters = {"全レース": np.ones(len(P), bool)}
for t in (0.4, 0.5, 0.6, 0.7):
    filters[f"本命の1着確率≥{t}"] = p1 >= t
    filters[f"本命の1着確率<{t}"] = p1 < t
for d in (0.0, 0.05, 0.1):
    filters[f"本命がモデル>市場+{d}"] = edge1 > d
for t in (0.5, 0.6):
    filters[f"市場の本命<{t}(荒れそう)"] = mkt_fav_q < t
res = []
for fname, ok in filters.items():
    for n in (1, 2, 3, 5, 10, 20):
        for lo in (1, 10, 20, 50):
            for hi in (50, 100, 300, 1e9):
                if hi <= lo:
                    continue
                for e in (0, 0.8, 1.0, 1.2, 1.5):
                    o = evaluate(ok, n, lo, hi, e)
                    b1, r1, n1, h1 = o["前半"]; b2, r2, n2, h2 = o["後半"]
                    res.append({"レース選び": fname, "上位N点": n, "オッズ": f"{lo:g}〜{hi:g}", "期待値≥": e,
                                "前半R": n1, "前半回収": r1 / b1 if b1 else np.nan, "前半的中R": h1,
                                "後半R": n2, "後半回収": r2 / b2 if b2 else np.nan, "後半的中R": h2,
                                "後半投資": b2, "後半収支": r2 - b2})
d = pd.DataFrame(res)
d.to_pickle("data/select_grid.pkl")
pd.set_option("display.width", 250)
ok = d[(d["前半R"] >= 200) & (d["後半R"] >= 1)]
print("\n■ 前半の回収率で上位15ルール（前半200レース以上）→ 後半の成績")
print(ok.sort_values("前半回収", ascending=False).head(15).round(3).to_string(index=False))
print("\n■ 選ばれ方による見かけ：前半上位50ルールの平均 前半", round(ok.sort_values("前半回収", ascending=False).head(50)["前半回収"].mean(), 3),
      "→ 後半", round(ok.sort_values("前半回収", ascending=False).head(50)["後半回収"].mean(), 3))
print("■ 全ルール", len(ok), "のうち 前半・後半とも100%超え", int(((ok["前半回収"] > 1) & (ok["後半回収"] > 1)).sum()))
print("\n■ 前半・後半とも100%超え（後半の投資が大きい順）")
print(ok[(ok["前半回収"] > 1) & (ok["後半回収"] > 1)].sort_values("後半投資", ascending=False).head(20).round(3).to_string(index=False))
