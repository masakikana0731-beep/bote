"""締切前のオッズでの「紙上の売買」（実際には買わない）。

確定オッズで買い目を選ぶシミュレーションは、締切後に決まるオッズを先に知っている分だけ良く見える。
ここでは collect watch が記録した締切5分前・1分前のオッズで買い目を選び、実際の結果で採点する。
同じ選び方を確定オッズでやった場合とも比べ、確定オッズを使うことでどれだけ良く見えていたかを測る。

  python -m boatbot.papertrade              # 過去データを更新し、記録のある日をすべて採点
  python -m boatbot.papertrade --no-update  # 過去データの更新をしない
  python -m boatbot.papertrade --no-final   # 確定オッズを取りに行かない（1日約1時間かかるので）

買い方（検証中の仮説）:
  3連単：確率 = 学習モデル^W_MODEL × 市場(1/オッズ)^W_MARKET を正規化。
         期待値 = 確率 × オッズ が閾値以上の目を、1レース最大5点・100円ずつ
  単勝・複勝・拡連複：学習モデルの確率 × オッズ が一番高い1点を、閾値以上なら100円
         （確定オッズのシミュレーションでは期待値の下限を上げるほど回収率が上がった。それが本物かを確かめる）
  どれも1日3,000円・最大10レース（券種・条件ごとに別々に数える）
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

from .collect import LEADS, ODDS, ODDS_LIVE, before_path

ODDS_MULTI = Path("data/odds_multi")
from .scraper import Fetcher, parse_beforeinfo_full, parse_odds3t

TRI = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
W_MODEL, W_MARKET = 0.12, 0.929  # 展示の進入コースで予想し、2026/7/25〜8/31 の確定オッズで推定
THRESHOLDS = (0.9, 1.0, 1.1)        # 3連単
TYPE_THRESHOLDS = (1.0, 1.2, 1.5)   # 単勝・複勝・拡連複（確定オッズのシミュレーションで良く見えた条件）
TYPE_NAMES = {"tri": "3連単", "tri_open": "3連単(荒れ)", "win": "単勝", "place": "複勝", "wide": "拡連複"}
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


def type_probs(p_tri: np.ndarray) -> dict[str, dict[str, float]]:
    """3連単確率 → 単勝・複勝（2着以内）・拡連複（2艇とも3着以内）の確率"""
    win, place, wide = {}, {}, {}
    for (a, b, c), p in zip(permutations(range(1, 7), 3), p_tri):
        win[str(a)] = win.get(str(a), 0) + p
        for x in (a, b):
            place[str(x)] = place.get(str(x), 0) + p
        for x, y in ((a, b), (a, c), (b, c)):
            k = f"{min(x, y)}-{max(x, y)}"
            wide[k] = wide.get(k, 0) + p
    return {"win": win, "place": place, "wide": wide}


def final_multi(hd: str, jcd: str, rno: int, f: Fetcher | None) -> dict | None:
    """単勝・複勝・拡連複の確定オッズ（data/odds_multi に保存。無ければ取得）"""
    from .scraper import parse_odds_tables, parse_win_place
    p = ODDS_MULTI / hd / f"{jcd}_{rno:02d}.json"
    if not p.exists() and f is not None:
        try:
            wp = parse_win_place(f.get("oddstf", rno=rno, jcd=jcd, hd=hd))
            wide = parse_odds_tables(f.get("oddsk", rno=rno, jcd=jcd, hd=hd))
            data = {"win": wp[0] if wp else {}, "place": wp[1] if len(wp) > 1 else {}, "wide": wide[0] if wide else {}}
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data))
        except Exception:
            return None
    return json.loads(p.read_text()) if p.exists() else None


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
        snaps.setdefault(f"{hd}{jcd}{rno}", {})[int(lead[1:])] = d
    rids = [r for r in snaps if r in races.index]
    if not rids:
        return None
    probs = day_probs(hist, hd, rids)
    fetcher = Fetcher() if fetch_final else None
    rows = []
    for rid in sorted(rids, key=lambda r: str(races.at[r, "deadline"])):
        if rid not in probs:
            continue
        base = {"race_id": rid, "deadline": str(races.at[rid, "deadline"])}
        pays = json.loads(races.at[rid, "payouts"])
        jcd, rno = rid[8:10], int(rid[10:])
        # --- 3連単（モデルと市場の混合確率 × オッズ）
        R = np.array([pays.get("tri", {}).get(c, 0) for c in TRI], dtype=float)
        fo = final_odds(hd, jcd, rno, fetcher)
        for lead, snap in snaps[rid].items():
            O = np.array([snap["tri"].get(c, 0.0) for c in TRI], dtype=float)
            for src, odds in (("締切前", O), ("確定", fo)):
                if odds is None:
                    continue
                ev = blend(probs[rid], odds) * odds
                for th in THRESHOLDS:
                    pick = [int(j) for j in np.argsort(-ev)[:PER_RACE] if ev[j] >= th and odds[j] > 0]
                    if pick:
                        rows.append({**base, "type": "tri", "lead": lead, "src": src, "th": th,
                                     "picks": [TRI[j] for j in pick], "ev": [round(float(ev[j]), 3) for j in pick],
                                     "odds_pre": [float(O[j]) for j in pick],
                                     "odds_final": [float(fo[j]) for j in pick] if fo is not None else None,
                                     "rets": [float(R[j]) for j in pick]})
        # --- 3連単「荒れそうなレース」ルール（2026-10-16：7〜8月・9〜10月とも確定オッズで100%超えだった唯一の候補）
        #     モデルの本命の1着確率 < 0.4 のレースだけ、モデル確率の上位5点のうち、オッズ20〜50倍かつ モデル確率×オッズ ≥ 1.2
        pw = {}
        for c, v in zip(TRI, probs[rid]):
            pw[c[0]] = pw.get(c[0], 0) + v
        if max(pw.values()) < 0.4:
            top5 = np.argsort(-probs[rid])[:5]
            for lead, snap in snaps[rid].items():
                O = np.array([snap["tri"].get(c, 0.0) for c in TRI], dtype=float)
                for src, odds in (("締切前", O), ("確定", fo)):
                    if odds is None:
                        continue
                    pick = [int(j) for j in top5 if 20 <= odds[j] < 50 and probs[rid][j] * odds[j] >= 1.2]
                    if pick:
                        rows.append({**base, "type": "tri_open", "lead": lead, "src": src, "th": 1.2,
                                     "picks": [TRI[j] for j in pick], "ev": [round(float(probs[rid][j] * odds[j]), 3) for j in pick],
                                     "odds_pre": [float(O[j]) for j in pick],
                                     "odds_final": [float(fo[j]) for j in pick] if fo is not None else None,
                                     "rets": [float(R[j]) for j in pick]})
        # --- 単勝・複勝・拡連複（モデルの確率 × オッズ が一番高い1点）
        tp = type_probs(probs[rid])
        fm = None
        for lead, snap in snaps[rid].items():
            for t in ("win", "place", "wide"):
                pre = snap.get(t) or {}
                if not pre:
                    continue
                if fm is None:
                    fm = final_multi(hd, jcd, rno, fetcher) or {}
                for src, odds in (("締切前", pre), ("確定", fm.get(t) or {})):
                    evs = {k: tp[t].get(k, 0) * o for k, o in odds.items() if o and o > 0}
                    if not evs:
                        continue
                    k = max(evs, key=evs.get)
                    for th in TYPE_THRESHOLDS:
                        if evs[k] >= th:
                            rows.append({**base, "type": t, "lead": lead, "src": src, "th": th, "picks": [k],
                                         "ev": [round(evs[k], 3)], "odds_pre": [float(pre.get(k, 0))],
                                         "odds_final": [float((fm.get(t) or {}).get(k, 0))] if fm.get(t) else None,
                                         "rets": [float(pays.get(t, {}).get(k, 0))]})
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
    if "type" not in df:
        df["type"] = "tri"
    df["type"] = df["type"].fillna("tri")
    df["day"] = df["race_id"].str[:8]
    out = []
    for (t, lead, src, th), g in df.groupby(["type", "lead", "src", "th"]):
        # 予算：締切順に、1日3,000円・最大10レースまで（券種・条件ごとに別々に）
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
        tb = pd.DataFrame([(d, b, x) for d, b, x, _ in taken], columns=["day", "bet", "ret"])
        drift = [np.log(f / p) for _, _, _, r in taken if r["odds_final"] for p, f in zip(r["odds_pre"], r["odds_final"])
                 if p > 0 and f > 0]
        out.append({"券種": TYPE_NAMES[t], "記録": f"{lead}分前", "選ぶオッズ": src, "期待値≥": th,
                    "日数": tb["day"].nunique(), "レース": len(tb), "投資": int(tb["bet"].sum()), "払戻": int(tb["ret"].sum()),
                    "回収率": round(tb["ret"].sum() / max(tb["bet"].sum(), 1), 3),
                    "的中レース": int((tb["ret"] > 0).sum()),
                    "オッズの変化(確定/締切前,中央値)": round(float(np.exp(np.median(drift))), 3) if drift else None})
    return pd.DataFrame(out)


def score_pending(fetch_final: bool = False, hist: pd.DataFrame | None = None) -> list[str]:
    """締切前オッズの記録があり、結果が出ていて、まだ採点していない日を採点する（アプリが毎朝呼ぶ）"""
    if not ODDS_LIVE.exists():
        return []
    if hist is None:
        from .ml import load_history
        hist = load_history()
    last = max(hist["hd"])
    done = []
    for d in sorted(p.name for p in ODDS_LIVE.iterdir() if p.is_dir()):
        if d > last or ((OUT / f"{d}.json").exists() and d < last):
            continue
        if score_day(hist, d, fetch_final=fetch_final) is not None:
            done.append(d)
    return done


# アプリの成績タブに出す買い方（締切5分前のオッズで選び、1点100円・上限なし）
APP_STRATEGIES = [
    ("place", 1.0, "複勝", "モデルの確率×オッズが一番高い1点（期待値1.0以上）"),
    ("win", 1.0, "単勝", "同上"),
    ("wide", 1.0, "拡連複", "同上"),
    ("tri", 0.9, "3連単（モデル＋市場）", "モデルと市場を混ぜた確率で期待値0.9以上の目（最大5点）"),
    ("tri_open", 1.2, "3連単（荒れそうなレース）", "本命の1着確率40%未満のレースで、20〜50倍・期待値1.2以上"),
]


def app_summary() -> dict:
    """成績タブ用：買い方ごとの通算と日別の収支"""
    rows = []
    for p in sorted(OUT.glob("*.json")):
        for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
            if r["src"] == "締切前" and r["lead"] == max(LEADS):
                rows.append((p.stem, r.get("type") or "tri", r["th"], len(r["picks"]), sum(r["rets"]),
                             sum(1 for x in r["rets"] if x > 0)))
    out = []
    for t, th, name, desc in APP_STRATEGIES:
        days: dict = {}
        for d, tt, tth, n, ret, hit in rows:
            if tt == t and tth == th:
                x = days.setdefault(d, {"day": d, "races": 0, "bet": 0, "ret": 0, "hits": 0})
                x["races"] += 1; x["bet"] += 100 * n; x["ret"] += int(ret); x["hits"] += hit
        daily = sorted(days.values(), key=lambda x: x["day"], reverse=True)
        bet, ret = sum(x["bet"] for x in daily), sum(x["ret"] for x in daily)
        out.append({"name": name, "desc": desc, "days": len(daily), "races": sum(x["races"] for x in daily),
                    "hits": sum(x["hits"] for x in daily), "bet": bet, "ret": ret,
                    "roi": round(ret / bet, 3) if bet else None, "daily": daily})
    scored = sorted(p.stem for p in OUT.glob("*.json")) if OUT.exists() else []
    return {"strategies": out, "scored_days": len(scored), "last_day": scored[-1] if scored else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--no-final", action="store_true", help="確定オッズを取りに行かない（速い。締切前オッズでの採点だけ）")
    a = ap.parse_args()
    if not a.no_update:
        from .history import update
        update(date.today())
    for d in score_pending(fetch_final=not a.no_final):
        print(d, "採点しました", flush=True)
    s = summarize()
    print(s.to_string(index=False) if len(s) else "まだ記録がありません")
    return 0


if __name__ == "__main__":
    sys.exit(main())
