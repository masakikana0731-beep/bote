"""週1回も買わないくらい厳しく絞ったら、回収率は100%を超えるか（2026-10-20）

- 予想：ウォークフォワードのモデル（四半期ごとに直前までのデータで学習、data/wf/）。オッズは使わない
- 券種ごとに「モデルが一番自信のある1点」の確率を出し、確率がしきい値以上のレースだけ買う（1点100円）
- しきい値は過去の四半期だけで決め（週1回未満になる中で回収率が最大）、次の四半期で採点
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from exp_train import tri_probs  # noqa: E402

TRI = [(a, b, c) for a, b, c in permutations(range(1, 7), 3)]
parts = []
for f in sorted(Path("data/wf").glob("scores_*.parquet")):
    info = json.loads((f.parent / f.name.replace("scores_", "info_").replace(".parquet", ".json")).read_text())
    s = pd.read_parquet(f).sort_values(["race_id", "waku"])
    parts.append((s["race_id"].to_numpy().reshape(-1, 6)[:, 0], tri_probs(s["score"].to_numpy().reshape(-1, 6), info["temps"])))
rid = np.concatenate([p[0] for p in parts]); P = np.concatenate([p[1] for p in parts])
hist = pd.read_parquet("data/history.parquet", columns=["hd", "jcd", "rno", "payouts"]).drop_duplicates(["hd", "jcd", "rno"])
hist["race_id"] = hist["hd"] + hist["jcd"] + hist["rno"].astype(str).str.zfill(2)
pays = hist.set_index("race_id")["payouts"].reindex(rid).map(lambda x: json.loads(x) if isinstance(x, str) else {}).tolist()

# 券種ごとの確率
def agg(keyf):
    keys = sorted({keyf(t) for t in TRI}, key=str)
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(TRI), len(keys)))
    for j, t in enumerate(TRI):
        for k in (keyf(t) if isinstance(keyf(t), list) else [keyf(t)]):
            M[j, idx[k]] += 1
    return keys, P @ M
types = {
    "単勝": agg(lambda t: str(t[0])),
    "複勝": (lambda: (lambda keys: (keys, P @ np.array([[1.0 if str(t[0]) == k or str(t[1]) == k else 0 for k in keys] for t in TRI])))([str(i) for i in range(1, 7)]))(),
    "2連単": agg(lambda t: f"{t[0]}-{t[1]}"),
    "2連複": agg(lambda t: f"{min(t[:2])}-{max(t[:2])}"),
    "拡連複": (lambda keys: (keys, P @ np.array([[1.0 if set(map(int, k.split('-'))) <= set(t) else 0 for k in keys] for t in TRI])))(
        [f"{a}-{b}" for a in range(1, 7) for b in range(a + 1, 7)]),
    "3連複": agg(lambda t: "-".join(map(str, sorted(t)))),
    "3連単": agg(lambda t: f"{t[0]}-{t[1]}-{t[2]}"),
}
KEY = {"単勝": "win", "複勝": "place", "2連単": "exacta", "2連複": "quinella", "拡連複": "wide", "3連複": "trio", "3連単": "tri"}
day = np.array([r[:8] for r in rid])
quarter = pd.PeriodIndex(pd.to_datetime(day, format="%Y%m%d"), freq="Q").astype(str).to_numpy()
QS = sorted(set(quarter))
weeks_q = {q: len(set(day[quarter == q])) / 7 for q in QS}
ok = np.array([bool(p) for p in pays])
print("レース", int(ok.sum()), "四半期", QS)

rows = []
for name, (keys, M) in types.items():
    best = M.argmax(1)
    p = M[np.arange(len(M)), best]
    ret = np.array([pays[i].get(KEY[name], {}).get(keys[best[i]], 0) if ok[i] else np.nan for i in range(len(M))])
    m = ok
    d = pd.DataFrame({"q": quarter[m], "p": p[m], "ret": ret[m]})
    # しきい値の候補：確率の上位から（細かく）
    ths = np.unique(np.quantile(d["p"], np.linspace(0.90, 0.9999, 200)))
    # 全期間での参考表（週あたりの購入回数ごと）
    for th in ths:
        g = d[d.p >= th]
        rows.append({"券種": name, "しきい値": th, "回数": len(g), "週あたり": len(g) / (len(set(day)) / 7),
                     "的中率": (g.ret > 0).mean() if len(g) else np.nan, "回収率": g.ret.sum() / (len(g) * 100) if len(g) else np.nan})
    # ウォークフォワード：過去の四半期で「週1回未満」かつ回収率最大のしきい値 → 次の四半期
    tb = tr = 0; det = []
    for k in range(1, len(QS)):
        past = d[d.q.isin(QS[:k])]; nw = sum(weeks_q[q] for q in QS[:k])
        cands = [(th, past[past.p >= th]) for th in ths]
        cands = [(th, g.ret.sum() / (len(g) * 100), len(g)) for th, g in cands if 5 <= len(g) <= nw]  # 週1回未満、最低5回
        if not cands:
            continue
        th, roi_past, n_past = max(cands, key=lambda x: x[1])
        cur = d[(d.q == QS[k]) & (d.p >= th)]
        tb += len(cur) * 100; tr += cur.ret.sum()
        det.append((QS[k], round(th, 3), round(roi_past, 2), len(cur), round(cur.ret.sum() / (len(cur) * 100), 2) if len(cur) else None))
    print(f"\n■ {name}：ウォークフォワード通算 回収率 {tr / tb if tb else float('nan'):.3f}（{int(tb / 100)}回）")
    for x in det:
        print(f"   {x[0]} しきい値{x[1]} 過去{x[2]} → 今期 {x[3]}回 回収{x[4]}")
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n■ 参考（全期間、週1回未満になるしきい値での回収率。選び方の見かけを含む）")
for name in types:
    g = t[(t["券種"] == name) & (t["週あたり"] < 1) & (t["回数"] >= 20)]
    if len(g):
        print(name, g.sort_values("回数", ascending=False).head(1)[["しきい値", "回数", "週あたり", "的中率", "回収率"]].round(3).to_string(index=False, header=False),
              "| 最大", g["回収率"].max().round(3))
print("\n■ 参考：しきい値を上げたときの回収率（複勝・単勝）")
for name in ("複勝", "単勝"):
    g = t[t["券種"] == name].iloc[::20]
    print(name); print(g[["しきい値", "回数", "週あたり", "的中率", "回収率"]].round(3).to_string(index=False))
