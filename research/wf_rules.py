"""数十万通りのルール（レースの選び方 × 上位N点 × オッズ帯 × 期待値）を四半期ごとに集計し、
ウォークフォワード（過去の四半期だけでルールを選ぶ → 次の四半期で採点）で本当の実力を測る。
比較：同じ手順を「市場（オッズ）の確率だけ」で行う（モデルの情報なし）。

  python wf_rules.py           # data/wf/scores_*.parquet（ウォークフォワードのモデル予想）と data/odds を使う
"""
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from exp_train import tri_probs  # noqa: E402

TRI = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
FIRST = np.array([int(c[0]) - 1 for c in TRI])

# ---- データ
parts = []
for f in sorted(Path("data/wf").glob("scores_*.parquet")):
    info = json.loads((f.parent / f.name.replace("scores_", "info_").replace(".parquet", ".json")).read_text())
    s = pd.read_parquet(f).sort_values(["race_id", "waku"])
    S = s["score"].to_numpy().reshape(-1, 6)
    P = tri_probs(S, info["temps"])
    parts.append((s["race_id"].to_numpy().reshape(-1, 6)[:, 0], P))
rid = np.concatenate([p[0] for p in parts])
P = np.concatenate([p[1] for p in parts])
hist = pd.read_parquet("data/history.parquet", columns=["hd", "jcd", "rno", "payouts"]).drop_duplicates(["hd", "jcd", "rno"])
hist["race_id"] = hist["hd"] + hist["jcd"] + hist["rno"].astype(str).str.zfill(2)
pays = hist.set_index("race_id")["payouts"]
bef = pd.read_parquet("data/before.parquet", columns=["hd", "jcd", "rno", "w_wind"]).drop_duplicates(["hd", "jcd", "rno"])
bef["race_id"] = bef["hd"] + bef["jcd"] + bef["rno"].astype(str).str.zfill(2)
wind_map = bef.set_index("race_id")["w_wind"]
keep, O, R = [], [], []
for k, r in enumerate(rid):
    f = Path("data/odds") / r[:8] / f"{r[8:10]}_{r[10:]}.json"
    if not f.exists() or r not in pays.index:
        continue
    o = json.loads(f.read_text())
    if len(o) < 100:
        continue
    pt = json.loads(pays[r]).get("tri", {})
    if not pt:
        continue
    keep.append(k)
    O.append([o.get(c, 0.0) for c in TRI])
    R.append([pt.get(c, 0) for c in TRI])
keep = np.array(keep)
rid, P = rid[keep], P[keep]
O, R = np.array(O, float), np.array(R, float)
q = np.where(O > 0, 1 / np.where(O > 0, O, 1), 0)
Q = q / q.sum(1, keepdims=True)
day = np.array([r[:8] for r in rid])
quarter = pd.PeriodIndex(pd.to_datetime(day, format="%Y%m%d"), freq="Q").astype(str).to_numpy()
QS = sorted(set(quarter))
Qm = np.stack([(quarter == x).astype(float) for x in QS], 1)          # (レース, 四半期)
rno = np.array([int(r[10:]) for r in rid])
wind = pd.Series(rid).map(wind_map).to_numpy(dtype=float)
print("レース", len(rid), "四半期", {x: int((quarter == x).sum()) for x in QS}, flush=True)


def race_conditions(Pm):
    """レースの選び方の部品（次元ごと）。Pm = 判断に使う確率（モデル or 市場）"""
    pw = np.zeros((len(Pm), 6)); qw = np.zeros((len(Pm), 6))
    np.add.at(pw.T, FIRST, Pm.T); np.add.at(qw.T, FIRST, Q.T)
    top = pw.argmax(1); p1 = pw.max(1)
    edge = p1 - qw[np.arange(len(Pm)), top]
    qmax = qw.max(1)
    agree = top == qw.argmax(1)
    dims = {
        "本命の強さ": {f"{lo}≤p1<{hi}": (p1 >= lo) & (p1 < hi) for lo, hi in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01))}
        | {f"p1<{x}": p1 < x for x in (.4, .5, .6)} | {f"p1≥{x}": p1 >= x for x in (.5, .6, .7)},
        "モデル−市場": {f"差<{x}": edge < x for x in (-.1, 0)} | {f"差≥{x}": edge >= x for x in (0, .05, .1, .15)},
        "市場の本命": {f"市場本命<{x}": qmax < x for x in (.4, .5, .6)} | {f"市場本命≥{x}": qmax >= x for x in (.5, .6, .7)},
        "本命の一致": {"一致": agree, "不一致": ~agree},
        "レース番号": {"1〜4R": rno <= 4, "5〜8R": (rno >= 5) & (rno <= 8), "9〜12R": rno >= 9},
        "風": {"風<3m": wind < 3, "風3〜4m": (wind >= 3) & (wind < 5), "風≥5m": wind >= 5},
    }
    filters = {"全レース": np.ones(len(Pm), bool)}
    flat = [(d, n, m) for d, cs in dims.items() for n, m in cs.items()]
    for d, n, m in flat:
        filters[n] = m
    for i, (d1, n1, m1) in enumerate(flat):
        for d2, n2, m2 in flat[i + 1:]:
            if d1 != d2:
                filters[f"{n1}&{n2}"] = m1 & m2
    names = list(filters)
    return names, np.stack([filters[n] for n in names]).astype(np.float32)


def evaluate(Pm, label, stakes=("均等",)):
    """stakes：賭け方。均等=1点100円 / 確率比例=確率に応じて1〜10口 / ケリー=資金1万円のケリー基準（期待値1未満は0）"""
    names, F = race_conditions(Pm)
    rank = np.argsort(-Pm, 1)
    EV = Pm * O
    W = {"均等": np.ones_like(Pm, dtype=np.float32),
         "確率比例": np.clip(np.round(Pm * 50), 1, 10).astype(np.float32),
         "ケリー": np.round(np.clip((EV - 1) / np.maximum(O - 1, 1e-6), 0, 1) * 100).astype(np.float32)}
    rows = []
    for n in (1, 2, 3, 5, 10, 20):
        topn = np.zeros_like(Pm, bool)
        topn[np.repeat(np.arange(len(Pm)), n), rank[:, :n].ravel()] = True
        for lo in (1, 10, 20, 30, 50, 100):
            for hi in (20, 30, 50, 100, 200, 1e9):
                if hi <= lo:
                    continue
                band = topn & (O >= lo) & (O < hi)
                for ev in (0, 1.0, 1.2, 1.5):
                    sel = band & (EV >= ev)
                    for st in stakes:
                        w = sel * W[st]
                        b = w.sum(1).astype(np.float32) * 100
                        r = (w * R).sum(1).astype(np.float32)
                        h = ((w * R) > 0).any(1).astype(np.float32)
                        bet = F @ (b[:, None] * Qm)            # (ルール, 四半期)
                        ret = F @ (r[:, None] * Qm)
                        hit = F @ (h[:, None] * Qm)
                        rows.append((n, lo, hi, ev, st, bet, ret, hit))
    return names, rows


def walk_forward(names, rows, label):
    """各四半期 k について、それより前の四半期の合計で上位のルールを選び、k で採点"""
    nF = len(names)
    BET = np.concatenate([x[5] for x in rows]); RET = np.concatenate([x[6] for x in rows]); HIT = np.concatenate([x[7] for x in rows])
    meta = [(names[i % nF], x[0], x[1], x[2], x[3], x[4]) for x in rows for i in range(nF)]
    out = {}
    for top_k in (1, 10, 100):
        tot_b = tot_r = 0.0
        detail = []
        for k in range(1, len(QS)):
            pb, pr, ph = BET[:, :k].sum(1), RET[:, :k].sum(1), HIT[:, :k].sum(1)
            ok = (pb >= 30000) & (ph >= 15)                 # 過去に300点以上・15レース以上的中
            roi = np.where(ok, pr / np.maximum(pb, 1), -1)
            idx = np.argsort(-roi)[:top_k]
            b, r = BET[idx, k].sum(), RET[idx, k].sum()
            tot_b += b; tot_r += r
            detail.append({"採点する四半期": QS[k], "選んだルールの過去回収率": round(float(roi[idx].mean()), 3),
                           "採点": round(float(r / b), 3) if b else None, "投資": int(b)})
            if top_k == 1:
                n, nn, lo, hi, ev, st = meta[idx[0]]
                detail[-1]["ルール"] = f"{n} / 上位{nn}点 / {lo:g}〜{hi:g}倍 / 期待値≥{ev} / {st}"
        out[top_k] = (tot_r / tot_b if tot_b else None, detail)
    # 全期間で100%を超えたルールの数（参考：選ばれ方の見かけ）
    allb, allr = BET.sum(1), RET.sum(1)
    big = allb >= 100000
    stable = big & np.all((RET > BET) | (BET == 0), 1) & (np.count_nonzero(BET, 1) == len(QS))
    print(f"\n===== {label}：ルール {len(meta):,} 通り =====")
    print("全四半期で100%超えのルール（投資10万円以上）:", int(stable.sum()), "/", int(big.sum()))
    for top_k, (roi, detail) in out.items():
        print(f"■ 過去の上位{top_k}ルールを使った場合（ウォークフォワード）：通算回収率 {roi:.3f}")
        for d in detail:
            print("   ", d)
    return out, (BET, RET, HIT, meta)


import pickle
res_model = walk_forward(*evaluate(P, "モデル", ("均等", "確率比例", "ケリー")), "モデルの確率で選ぶ（賭け方3種類を含む）")
res_market = walk_forward(*evaluate(Q, "市場"), "市場の確率だけで選ぶ（比較用：モデルの情報なし）")
pickle.dump({"QS": QS, "model": res_model[1], "market": res_market[1], "day": day, "quarter": quarter},
            open("data/wf_rules.pkl", "wb"))
