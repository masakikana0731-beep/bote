"""オッズを使わず、モデルだけでレースを選ぶ（本命の強さ・2番手との差）× 全券種の上位N点"""
import json
from itertools import permutations
import numpy as np, pandas as pd
from boatbot.ml import VALID_END, Model, add_features, clean_races, load_history, stage_probs
SPLIT = "20260831"
df = clean_races(add_features(load_history()))
te = df[df["hd"] > VALID_END].sort_values(["race_id", "waku"])
m = Model(); S = m.scores(te).reshape(-1, 6)
rid = te["race_id"].to_numpy().reshape(-1, 6)[:, 0]
pays = te.drop_duplicates("race_id").set_index("race_id")["payouts"].map(json.loads)
rows = []
for k, r in enumerate(rid):
    P = stage_probs(S[k], m.temps)
    pw = P.sum((1, 2)); o = np.argsort(-pw)
    p1, gap = pw[o[0]], pw[o[0]] - pw[o[1]]
    pr = {}
    for a, b, c in permutations(range(6), 3):
        v = P[a, b, c]; A, B_, C = a + 1, b + 1, c + 1
        for t, key in (("tri", f"{A}-{B_}-{C}"), ("trio", "-".join(map(str, sorted((A, B_, C))))), ("exacta", f"{A}-{B_}"),
                       ("quinella", f"{min(A,B_)}-{max(A,B_)}"), ("win", str(A))):
            pr.setdefault(t, {}); pr[t][key] = pr[t].get(key, 0) + v
        for x, y in ((A, B_), (A, C), (B_, C)):
            pr.setdefault("wide", {}); kk = f"{min(x,y)}-{max(x,y)}"; pr["wide"][kk] = pr["wide"].get(kk, 0) + v
        for x in (A, B_):
            pr.setdefault("place", {}); pr["place"][str(x)] = pr["place"].get(str(x), 0) + v
    for t, ns in (("win", (1,)), ("place", (1,)), ("exacta", (1, 2, 3)), ("quinella", (1, 2, 3)), ("wide", (1, 2)), ("trio", (1, 2, 3, 5)), ("tri", (1, 2, 3, 5, 10))):
        srt = sorted(pr[t], key=lambda c: -pr[t][c])
        for n in ns:
            ret = sum(pays[r].get(t, {}).get(c, 0) for c in srt[:n])
            rows.append((r[:8], t, n, p1, gap, n * 100, ret))
d = pd.DataFrame(rows, columns=["day", "type", "n", "p1", "gap", "bet", "ret"])
d["half"] = np.where(d.day <= SPLIT, "前半", "後半")
out = []
for (t, n), g in d.groupby(["type", "n"]):
    for fname, cond in [("全レース", g.p1 >= 0)] + [(f"本命≥{x}", g.p1 >= x) for x in (0.5, 0.6, 0.7, 0.8)] + \
                       [(f"本命<{x}", g.p1 < x) for x in (0.3, 0.4)] + [(f"2番手との差≥{x}", g.gap >= x) for x in (0.3, 0.5, 0.6)]:
        h = g[cond]
        f, s = h[h.half == "前半"], h[h.half == "後半"]
        if len(f) < 200:
            continue
        out.append({"券種": t, "上位N点": n, "レース選び": fname, "前半R": len(f), "前半回収": f.ret.sum() / f.bet.sum(),
                    "後半R": len(s), "後半回収": s.ret.sum() / s.bet.sum(), "後半的中率": (s.ret > 0).mean()})
o = pd.DataFrame(out)
pd.set_option("display.width", 250)
print("■ 前半の回収率 上位15 → 後半")
print(o.sort_values("前半回収", ascending=False).head(15).round(3).to_string(index=False))
print("\n■ 前半・後半とも100%超え:", int(((o.前半回収 > 1) & (o.後半回収 > 1)).sum()), "/", len(o))
print("■ 後半の回収率 上位10（参考）")
print(o.sort_values("後半回収", ascending=False).head(10).round(3).to_string(index=False))
