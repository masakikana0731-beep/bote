"""過去データから学習する予測モデル。

流れ:
  1. history.parquet（番組表＋競走成績）から、各艇の「レース前に分かる」特徴量を作る
     - 過去成績系の特徴量は必ず「その日より前」のデータだけで計算する（未来の情報を使わない）
  2. LightGBM（LambdaRank）で「どの艇が上位に来やすいか」のスコアを学習
  3. スコア → 1着確率（softmax）。2着・3着は段階ごとに温度を変えた Plackett-Luce で計算
     （温度は検証期間で最尤推定。素の Harville 法は人気薄の2・3着を過大評価しがちなため）
  4. 3連単120通りの確率を出す

  python -m boatbot.ml train          # 学習（data/model/ に保存）
  python -m boatbot.ml evaluate       # テスト期間での精度評価
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

HIST = Path("data/history.parquet")
MODEL_DIR = Path("data/model")

TRAIN_START = "20231001"  # 学習: この日〜（それ以前の3か月は過去成績系の特徴量を溜めるためだけに使う）
TRAIN_END = "20260430"   # 学習: 〜この日
VALID_END = "20260714"   # 検証(温度・木の本数の決定): TRAIN_END の翌日〜この日
# テスト: VALID_END の翌日以降（オッズを取得してある期間。戦略シミュレーションに使う）

GRADES = ["A1", "A2", "B1", "B2"]
RACE_KINDS = ["優勝", "準優", "選抜", "特選", "特賞", "予選", "一般"]

# モデルに入れる特徴量（レース前に分かる情報だけ）
FEATURES = [
    # 枠・進入
    "course", "waku", "course_changed",
    # 選手の基本情報（番組表）
    "grade_num", "age", "weight", "nat_win", "nat_2r", "loc_win", "loc_2r", "loc_missing",
    # 機材（番組表）
    "motor_2r", "boat_2r",
    # 直前情報
    "ex_time_diff", "ex_time_rank",
    # 過去成績（その日より前のデータから計算）
    "r_course_win", "r_course_top3", "r_course_n", "r_st_mean", "r_f_180", "r_finish_60", "r_n_365",
    "meet_mean",
    # 場・水面
    "venue_course_win", "wind", "wave",
    # レース内での相対値
    "nat_win_rel", "motor_2r_rel", "r_st_rel",
    # レース種別
    "race_kind", "rno",
]


# ---------------------------------------------------------------- 前処理

def _st(x: str) -> float:
    """ST文字列 → 数値。F（フライング）はマイナス、L/不明は NaN"""
    if not isinstance(x, str):
        return np.nan
    x = x.strip()
    try:
        return -float(x[1:]) if x.startswith("F") else float(x)
    except ValueError:
        return np.nan


def _meet_mean(s: str) -> float:
    """今節成績（例: '1 3 2'）の平均着順"""
    digits = [int(c) for c in str(s) if c in "123456"]
    return float(np.mean(digits)) if digits else np.nan


def _race_kind(name: str) -> int:
    for i, k in enumerate(RACE_KINDS):
        if k in str(name):
            return i
    return len(RACE_KINDS)


def load_history() -> pd.DataFrame:
    df = pd.read_parquet(HIST)
    df["date"] = pd.to_datetime(df["hd"], format="%Y%m%d")
    df["race_id"] = df["hd"] + df["jcd"] + df["rno"].astype(str).str.zfill(2)
    df["st"] = df["st_raw"].map(_st)
    df["is_f"] = df["st_raw"].astype(str).str.startswith("F").astype(int)
    df = df.sort_values(["date", "race_id", "waku"]).reset_index(drop=True)
    return df


def _rolling_by(df: pd.DataFrame, keys: list[str], cols: dict[str, str], window: str) -> pd.DataFrame:
    """keys ごとに、その日より前 window 期間の集計（closed='left' で当日を含まない）"""
    out = pd.DataFrame(index=df.index)
    g = df[keys + ["date"] + list(cols)].copy()
    g = g.sort_values(keys + ["date"])
    for col, how in cols.items():
        rolled = (g.set_index("date").groupby(keys, sort=False)[col]
                  .rolling(window, closed="left").agg(how))
        out.loc[g.index, f"{col}_{how}_{window}"] = rolled.to_numpy()
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _basic(df)
    # --- 過去成績（当日を含まない）
    rc = _rolling_by(df, ["toban", "course"], {"win": "sum", "top3": "sum", "one": "sum"}, "365D")
    rr = _rolling_by(df, ["toban"], {"st": "mean", "is_f": "sum"}, "180D")
    r60 = _rolling_by(df, ["toban"], {"finish_f": "mean"}, "60D")
    r365 = _rolling_by(df, ["toban"], {"one": "sum"}, "365D")
    vc = _rolling_by(df, ["jcd", "course"], {"win": "mean"}, "365D")
    df["rc_win"], df["rc_top3"], df["rc_n"] = rc["win_sum_365D"], rc["top3_sum_365D"], rc["one_sum_365D"]
    df["r_st_mean"], df["r_f_180"] = rr["st_mean_180D"], rr["is_f_sum_180D"]
    df["r_finish_60"], df["r_n_365"] = r60["finish_f_mean_60D"], r365["one_sum_365D"]
    df["venue_course_win"] = vc["win_mean_365D"]
    return _finish(df, course_priors(df))


def course_priors(df: pd.DataFrame) -> dict:
    """縮小推定の事前値（コース別の平均）。未来の情報が混ざらないよう学習期間だけで計算"""
    past = df[df["hd"] <= TRAIN_END]
    past = past if len(past) else df
    return {"win": {int(k): float(v) for k, v in past.groupby("course")["win"].mean().items()},
            "top3": {int(k): float(v) for k, v in past.groupby("course")["top3"].mean().items()}}


def _finish(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """集計済みの過去成績から最終的な特徴量を作る（学習時・本番時で共通）"""
    k = 10.0  # 少ないサンプルは全体平均に寄せる（縮小推定）
    n = df["rc_n"].fillna(0)
    pw = df["course"].map(lambda c: priors["win"].get(int(c), 1 / 6))
    pt = df["course"].map(lambda c: priors["top3"].get(int(c), 0.5))
    df["r_course_win"] = (df["rc_win"].fillna(0) + k * pw) / (n + k)
    df["r_course_top3"] = (df["rc_top3"].fillna(0) + k * pt) / (n + k)
    df["r_course_n"] = n
    df["r_f_180"] = df["r_f_180"].fillna(0)
    df["r_n_365"] = df["r_n_365"].fillna(0)
    # --- レース内の相対値
    for col, rel in (("nat_win", "nat_win_rel"), ("motor_2r", "motor_2r_rel"), ("r_st_mean", "r_st_rel")):
        df[rel] = df[col] - df.groupby("race_id")[col].transform("mean")
    return df


def _basic(df: pd.DataFrame) -> pd.DataFrame:
    """そのレースの情報だけで作れる特徴量"""
    df = df.copy()
    df["course"] = df["course"].fillna(df["waku"]).astype(int)
    df["course_changed"] = (df["course"] != df["waku"]).astype(int)
    df["grade_num"] = df["grade"].map({g: i for i, g in enumerate(GRADES)}).fillna(3)
    df["loc_missing"] = (df["loc_win"] <= 0).astype(int)
    df.loc[df["loc_win"] <= 0, ["loc_win", "loc_2r"]] = np.nan
    df.loc[df["motor_2r"] <= 0, "motor_2r"] = np.nan  # 期の初めは0.00
    df.loc[df["boat_2r"] <= 0, "boat_2r"] = np.nan
    df["meet_mean"] = df["meet_results"].map(_meet_mean)
    df["race_kind"] = df["race_name"].map(_race_kind)
    # --- 展示タイム（レース内の差・順位）
    ex = df["ex_time"].where(df["ex_time"] > 0)
    df["ex_time_diff"] = ex - ex.groupby(df["race_id"]).transform("mean")
    df["ex_time_rank"] = ex.groupby(df["race_id"]).rank(method="average")
    if "finish" in df:
        df["win"] = (df["finish"] == 1).astype(float)
        df["top3"] = (df["finish"] <= 3).astype(float)
        df["finish_f"] = df["finish"].astype(float)
    df["one"] = 1.0
    return df


# ---------------------------------------------------------------- 本番（当日のレース）用

def snapshot(hist: pd.DataFrame, asof: str) -> dict:
    """asof 日の朝時点の過去成績集計。add_features の rolling(closed='left') と同じ定義"""
    t = pd.Timestamp(asof)
    h = _basic(hist[(hist["date"] < t) & (hist["date"] >= t - pd.Timedelta("365D"))])
    h180 = h[h["date"] >= t - pd.Timedelta("180D")]
    h60 = h[h["date"] >= t - pd.Timedelta("60D")]
    return {
        "rc": h.groupby(["toban", "course"]).agg(rc_win=("win", "sum"), rc_top3=("top3", "sum"), rc_n=("one", "sum")),
        "rr": h180.groupby("toban").agg(r_st_mean=("st", "mean"), r_f_180=("is_f", "sum")),
        "r60": h60.groupby("toban").agg(r_finish_60=("finish_f", "mean")),
        "r365": h.groupby("toban").agg(r_n_365=("one", "sum")),
        "vc": h.groupby(["jcd", "course"]).agg(venue_course_win=("win", "mean")),
    }


def live_features(race: pd.DataFrame, snap: dict, priors: dict) -> pd.DataFrame:
    """当日レース（番組表相当の列＋展示）→ 特徴量"""
    df = _basic(race)
    df = df.join(snap["rc"], on=["toban", "course"]).join(snap["rr"], on="toban")
    df = df.join(snap["r60"], on="toban").join(snap["r365"], on="toban").join(snap["vc"], on=["jcd", "course"])
    df = _finish(df, priors)
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    return df


def race_frame(info, hd: str) -> pd.DataFrame:
    """scraper.RaceInfo → live_features に渡す DataFrame"""
    rows = []
    for b in info.boats:
        rows.append({
            "hd": hd, "jcd": info.jcd, "rno": info.rno, "race_id": f"{hd}{info.jcd}{info.rno:02d}",
            "race_name": info.race_name, "waku": b.waku, "toban": b.toban, "age": b.age, "weight": b.weight,
            "grade": b.grade, "nat_win": b.nat_win, "nat_2r": b.nat_2r, "loc_win": b.loc_win or 0.0,
            "loc_2r": b.loc_2r or 0.0, "motor_2r": b.motor_2r or 0.0, "boat_2r": b.boat_2r or 0.0,
            "meet_results": b.meet_results, "ex_time": b.ex_time if b.ex_time else np.nan,
            "course": b.course if b.course else b.waku, "wind": info.wind, "wave": info.wave,
        })
    return pd.DataFrame(rows)


class LivePredictor:
    """本番用：その日の朝に過去成績を集計しておき、レースごとに3連単確率を返す"""

    def __init__(self):
        self.model = Model()
        self.priors = self.model.meta["priors"]
        self.hist = None
        self.snap_day = None
        self.snap = None

    def _ensure(self, hd: str) -> None:
        if self.snap_day == hd:
            return
        if self.hist is None or self.hist["date"].max() < pd.Timestamp(hd) - pd.Timedelta("1D"):
            self.hist = load_history()
        self.snap = snapshot(self.hist, hd)
        self.snap_day = hd

    def trifecta(self, info, hd: str) -> dict[str, float]:
        self._ensure(hd)
        return self.model.trifecta(live_features(race_frame(info, hd), self.snap, self.priors))


def clean_races(df: pd.DataFrame) -> pd.DataFrame:
    """6艇そろって1〜3着が確定しているレースだけ使う"""
    g = df.groupby("race_id")
    ok = (g["waku"].transform("count") == 6) & g["finish"].transform(lambda s: {1, 2, 3} <= set(s.dropna()))
    return df[ok]


# ---------------------------------------------------------------- 確率

def stage_probs(scores: np.ndarray, temps: tuple[float, float, float]) -> np.ndarray:
    """scores(6,) → 3連単確率 (6,6,6)。段階ごとに温度 temps を変えた Plackett-Luce"""
    out = np.zeros((6, 6, 6))
    e1 = np.exp(temps[0] * (scores - scores.max()))
    p1 = e1 / e1.sum()
    e2 = np.exp(temps[1] * (scores - scores.max()))
    e3 = np.exp(temps[2] * (scores - scores.max()))
    for a in range(6):
        rest2 = e2.sum() - e2[a]
        for b in range(6):
            if b == a:
                continue
            p2 = e2[b] / rest2
            rest3 = e3.sum() - e3[a] - e3[b]
            for c in range(6):
                if c in (a, b):
                    continue
                out[a, b, c] = p1[a] * p2 * e3[c] / rest3
    return out


def top3_loglik(S: np.ndarray, Y: np.ndarray, temps) -> float:
    """S: (N,6) スコア, Y: (N,3) 1〜3着の艇index。平均対数尤度"""
    t1, t2, t3 = temps
    N = len(S)
    idx = np.arange(N)
    a, b, c = Y[:, 0], Y[:, 1], Y[:, 2]
    def lse(x):
        m = x.max(1, keepdims=True)
        return (m + np.log(np.exp(x - m).sum(1, keepdims=True)))[:, 0]
    l1 = t1 * S[idx, a] - lse(t1 * S)
    m2 = t2 * S.copy(); m2[idx, a] = -np.inf
    l2 = t2 * S[idx, b] - lse(m2)
    m3 = t3 * S.copy(); m3[idx, a] = -np.inf; m3[idx, b] = -np.inf
    l3 = t3 * S[idx, c] - lse(m3)
    return float(np.mean(l1 + l2 + l3))


def fit_temps(S: np.ndarray, Y: np.ndarray) -> tuple[float, float, float]:
    from scipy.optimize import minimize
    res = minimize(lambda t: -top3_loglik(S, Y, t), x0=[1.0, 1.0, 1.0], method="Nelder-Mead")
    return tuple(float(x) for x in res.x)


def to_matrix(df: pd.DataFrame, scores: np.ndarray):
    """艇単位のスコアをレース単位 (N,6) と着順 (N,3) に"""
    d = df.assign(score=scores).sort_values(["race_id", "waku"])
    S = d["score"].to_numpy().reshape(-1, 6)
    fin = d["finish"].to_numpy().reshape(-1, 6)
    Y = np.stack([np.argmax(fin == k, axis=1) for k in (1, 2, 3)], axis=1)
    return d["race_id"].to_numpy().reshape(-1, 6)[:, 0], S, Y


# ---------------------------------------------------------------- 学習

def train() -> None:
    import lightgbm as lgb
    print("特徴量を作成中…", flush=True)
    df = clean_races(add_features(load_history()))
    df = df[df["hd"] >= TRAIN_START]
    df["label"] = (7 - df["finish"]).clip(lower=0).fillna(0).astype(int)  # 1着=6 … 6着=1
    tr = df[df["hd"] <= TRAIN_END].sort_values(["race_id", "waku"])
    va = df[(df["hd"] > TRAIN_END) & (df["hd"] <= VALID_END)].sort_values(["race_id", "waku"])
    print(f"学習 {tr['race_id'].nunique()} R / 検証 {va['race_id'].nunique()} R", flush=True)
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 3], learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambdarank_truncation_level=6, verbose=-1)
    dtr = lgb.Dataset(tr[FEATURES], tr["label"], group=tr.groupby("race_id", sort=False).size().to_numpy())
    dva = lgb.Dataset(va[FEATURES], va["label"], group=va.groupby("race_id", sort=False).size().to_numpy())
    booster = lgb.train(params, dtr, 3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    _, S, Y = to_matrix(va, booster.predict(va[FEATURES]))
    temps = fit_temps(S, Y)
    print(f"温度 {temps}  検証の対数尤度 {top3_loglik(S, Y, temps):.4f}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_DIR / "lgb.txt"))
    imp = dict(zip(FEATURES, booster.feature_importance("gain").round(1).tolist()))
    (MODEL_DIR / "meta.json").write_text(json.dumps(
        {"temps": temps, "features": FEATURES, "train_end": TRAIN_END, "valid_end": VALID_END,
         "best_iteration": booster.best_iteration, "importance": imp, "priors": course_priors(df)}, ensure_ascii=False, indent=1))
    print("保存しました", MODEL_DIR)


class Model:
    def __init__(self, model_dir: Path | None = None):
        import lightgbm as lgb
        model_dir = model_dir or MODEL_DIR
        self.booster = lgb.Booster(model_file=str(model_dir / "lgb.txt"))
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.temps = tuple(self.meta["temps"])

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df[FEATURES])

    def trifecta(self, race_df: pd.DataFrame) -> dict[str, float]:
        """1レース6艇（waku順）→ {'1-2-3': 確率}"""
        r = race_df.sort_values("waku")
        P = stage_probs(self.scores(r), self.temps)
        return {f"{a+1}-{b+1}-{c+1}": float(P[a, b, c]) for a, b, c in permutations(range(6), 3)}


# ---------------------------------------------------------------- 評価

def evaluate() -> dict:
    df = clean_races(add_features(load_history()))
    te = df[df["hd"] > VALID_END].sort_values(["race_id", "waku"])
    m = Model()
    rid, S, Y = to_matrix(te, m.scores(te))
    # ベースライン: コース別1着率だけ
    base = te.sort_values(["race_id", "waku"])["venue_course_win"].fillna(1 / 6).to_numpy().reshape(-1, 6)
    Sb = np.log(np.clip(base, 1e-3, None))
    tb = fit_temps(Sb, Y)
    res = {"races": len(S),
           "model_loglik": top3_loglik(S, Y, m.temps), "base_loglik": top3_loglik(Sb, Y, tb),
           "uniform_loglik": -math.log(120)}
    # 1着的中率・3連単の順位
    res["win_acc"] = float(np.mean(S.argmax(1) == Y[:, 0]))
    res["base_win_acc"] = float(np.mean(Sb.argmax(1) == Y[:, 0]))
    ranks, mranks = [], []
    races = te.groupby("race_id").first()
    for i in range(len(S)):
        P = stage_probs(S[i], m.temps).ravel()
        actual = P.reshape(6, 6, 6)[Y[i, 0], Y[i, 1], Y[i, 2]]
        ranks.append(int((P > actual).sum()) + 1)
        mranks.append(races.loc[rid[i], "tri_pop"])
    res["model_tri_rank_median"] = float(np.median(ranks))
    res["market_tri_rank_median"] = float(np.nanmedian(np.array(mranks, dtype=float)))
    res["model_top10_rate"] = float(np.mean(np.array(ranks) <= 10))
    res["market_top10_rate"] = float(np.nanmean(np.array(mranks, dtype=float) <= 10))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "evaluate"])
    a = ap.parse_args()
    train() if a.cmd == "train" else evaluate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
