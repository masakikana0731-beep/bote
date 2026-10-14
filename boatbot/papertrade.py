"""締切前のオッズでの「紙上の売買」（実際には買わない）。

確定オッズで買い目を選ぶシミュレーションは、締切後に決まるオッズを先に知っている分だけ良く見える。
ここでは collect watch が記録した締切5分前・1分前のオッズで買い目を選び、実際の結果で採点する。
同じ選び方を確定オッズでやった場合とも比べ、確定オッズを使うことでどれだけ良く見えていたかを測る。

  python -m boatbot.papertrade              # 過去データを更新し、記録のある日をすべて採点
  python -m boatbot.papertrade --no-update  # 過去データの更新をしない

買い方（検証中の仮説）:
  確率 = 学習モデル^W_MODEL × 市場(1/オッズ)^W_MARKET を正規化
  期待値 = 確率 × オッズ が閾値以上の3連単を、1レース最大5点・100円ずつ。1日3,000円・最大10レース
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import ODDS, ODDS_LIVE, before_path
from .scraper import Fetcher, parse_beforeinfo_full, parse_odds3t

TRI = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
W_MODEL, W_MARKET = 0.12, 0.929  # 展示の進入コースで予想し、2026/7/25〜8/31 の確定オッズで推定
THRESHOLDS = (0.9, 1.0, 1.1)
PER_RACE, DAILY, MAX_RACES = 5, 3000, 10
OUT = Path("data/papertrade")


def blend(p_model: np.ndarray, odds: np.ndarray) -> np.ndarray:
    q = np.where(odds > 0, 1 / np.where(odds > 0, odds, 1), 0)
    q = q / q.sum()
    z = W_MODEL * np.log(np.maximum(p_model, 1e-12)) + W_MARKET * np.log(np.maximum(q, 1e-12))
    z = np.where(odds > 0, z, -np.inf)
    e = np.exp(z - z.max())
    return e / e.sum()


def day_probs(hist: pd.DataFrame, hd: str, race_ids: list[str]) -> dict[str, np.ndarray]:
    """その日の朝時点の情報で、各レースの3連単確率（TRI 順）"""
    from .ml import Model, live_features, snapshot, stage_probs
    model = Model()
    priors = model.meta["priors"]
    snap = snapshot(hist, hd, priors)
    day = hist[(hist["hd"] == hd) & (hist["race_id"].isin(race_ids))].copy()
    # 本番と同じく、進入コースは直前情報（スタート展示）のものを使う（保存してあれば）
    for rid, g in day.groupby("race_id"):
        p = before_path(hd, rid[8:10], int(rid[10:]))
        if p.exists():
            boats = parse_beforeinfo_full(gzip.decompress(p.read_bytes()).decode("utf-8"))["boats"]
            for i, row in g.iterrows():
                c = boats.get(int(row["waku"]), {}).get("ex_course")
                if c:
                    day.at[i, "course"] = c
    cols = ["hd", "jcd", "rno", "race_id", "race_name", "waku", "toban", "age", "weight", "grade", "nat_win", "nat_2r",
            "loc_win", "loc_2r", "motor_2r", "boat_2r", "meet_results", "ex_time", "course", "wind", "wave",
            "deadline", "motor_no", "boat_no"]
    f = live_features(day[cols].reset_index(drop=True), snap, priors)
    out = {}
    for rid, g in f.groupby("race_id"):
        if len(g) != 6:
            continue
        P = stage_probs(model.scores(g.sort_values("waku")), model.temps)
        out[rid] = np.array([P[a - 1, b - 1, c - 1] for a, b, c in permutations(range(1, 7), 3)])
    return out


def final_odds(hd: str, jcd: str, rno: int, f: Fetcher | None) -> np.ndarray | None:
    p = ODDS / hd / f"{jcd}_{rno:02d}.json"
    if not p.exists() and f is not None:
        try:
            o = parse_odds3t(f.get("odds3t", rno=rno, jcd=jcd, hd=hd))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(o))
        except Exception:
            return None
    if not p.exists():
        return None
    o = json.loads(p.read_text())
    return np.array([o.get(c, 0.0) for c in TRI], dtype=float) if len(o) >= 100 else None


def score_day(hist: pd.DataFrame, hd: str, fetch_final: bool = True) -> dict | None:
    races = hist[hist["hd"] == hd].drop_duplicates("race_id").set_index("race_id")
    if races.empty:
        return None
    snaps = {}
    for p in sorted((ODDS_LIVE / hd).glob("*_m*.json")):
        jcd, rno, lead = p.stem.split("_")
        d = json.loads(p.read_text())
        if d.get("late") or len(d.get("tri", {})) < 100:
            continue
        snaps.setdefault(f"{hd}{jcd}{rno}", {})[int(lead[1:])] = np.array([d["tri"].get(c, 0.0) for c in TRI])
    rids = [r for r in snaps if r in races.index]
    if not rids:
        return None
    probs = day_probs(hist, hd, rids)
    fetcher = Fetcher() if fetch_final else None
    rows = []
    for rid in sorted(rids, key=lambda r: str(races.at[r, "deadline"])):
        if rid not in probs:
            continue
        pay = json.loads(races.at[rid, "payouts"]).get("tri", {})
        R = np.array([pay.get(c, 0) for c in TRI], dtype=float)
        fo = final_odds(hd, rid[8:10], int(rid[10:]), fetcher)
        for lead, O in snaps[rid].items():
            for src, odds in (("締切前", O), ("確定", fo)):
                if odds is None:
                    continue
                B = blend(probs[rid], odds)
                ev = B * odds
                for th in THRESHOLDS:
                    pick = [int(j) for j in np.argsort(-ev)[:PER_RACE] if ev[j] >= th and odds[j] > 0]
                    if pick:
                        rows.append({"race_id": rid, "deadline": str(races.at[rid, "deadline"]), "lead": lead, "src": src,
                                     "th": th, "picks": [TRI[j] for j in pick], "ev": [round(float(ev[j]), 3) for j in pick],
                                     "odds_pre": [float(O[j]) for j in pick],
                                     "odds_final": [float(fo[j]) for j in pick] if fo is not None else None,
                                     "rets": [float(R[j]) for j in pick]})
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"hd": hd, "races": len(rids), "rows": rows}
    (OUT / f"{hd}.json").write_text(json.dumps(res, ensure_ascii=False))
    return res


def summarize() -> pd.DataFrame:
    rows = []
    for p in sorted(OUT.glob("*.json")):
        rows += json.loads(p.read_text())["rows"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["day"] = df["race_id"].str[:8]
    out = []
    for (lead, src, th), g in df.groupby(["lead", "src", "th"]):
        # 予算：締切順に、1日3,000円・最大10レースまで
        g = g.sort_values(["day", "deadline"])
        taken, spent, races, cur = [], 0, 0, None
        for _, r in g.iterrows():
            if r["day"] != cur:
                cur, spent, races = r["day"], 0, 0
            n = min(len(r["picks"]), (DAILY - spent) // 100)
            if races >= MAX_RACES or n <= 0:
                continue
            # 予算で削った場合は期待値の高い順に先頭 n 点だけ買う
            taken.append((r["day"], 100 * n, sum(r["rets"][:n]), r))
            spent += 100 * n
            races += 1
        t = pd.DataFrame([(d, b, x) for d, b, x, _ in taken], columns=["day", "bet", "ret"])
        drift = [np.log(f / p) for _, _, _, r in taken if r["odds_final"] for p, f in zip(r["odds_pre"], r["odds_final"])
                 if p > 0 and f > 0]
        out.append({"記録": f"{lead}分前", "選ぶオッズ": src, "期待値≥": th, "日数": t["day"].nunique(),
                    "レース": len(t), "投資": int(t["bet"].sum()), "払戻": int(t["ret"].sum()),
                    "回収率": round(t["ret"].sum() / max(t["bet"].sum(), 1), 3),
                    "的中レース": int((t["ret"] > 0).sum()),
                    "オッズの変化(確定/締切前,中央値)": round(float(np.exp(np.median(drift))), 3) if drift else None})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    a = ap.parse_args()
    if not a.no_update:
        from .history import update
        update(date.today())
    from .ml import load_history
    hist = load_history()
    for d in sorted(p.name for p in ODDS_LIVE.iterdir() if p.is_dir()):
        if (OUT / f"{d}.json").exists() and d < max(hist["hd"]):
            continue
        r = score_day(hist, d)
        print(d, "採点なし（結果がまだ無い）" if r is None else f"{r['races']} レース", flush=True)
    s = summarize()
    print(s.to_string(index=False) if len(s) else "まだ記録がありません")
    return 0


if __name__ == "__main__":
    sys.exit(main())
