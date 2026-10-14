"""過去データから学習する予測モデル。

流れ:
  1. history.parquet（番組表＋競走成績）から、各艇の「レース前に分かる」特徴量を作る
     - 過去成績系の特徴量は必ず「その日より前」のデータだけで計算する（未来の情報を使わない）
  2. LightGBM で各艇の「強さスコア」を学習。目的関数は 1〜3着の Plackett-Luce 尤度
     （以前は LambdaRank。確率の当てやすさを直接学習するほうがテスト期間の対数尤度が良かった）
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
BEFORE = Path("data/before.parquet")  # 直前情報（boatbot.collect で収集）。無ければ使わない
MODEL_DIR = Path("models")

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
    # 調子（コースの有利不利を除いた3着以内率・1着率。その日より前）
    "form90_top3", "form90_win", "form30_top3", "n30", "form365_top3",
    # スタート（そのコースでの平均ST・STのばらつき）
    "st_course", "st_std",
    # その場との相性・場・時間帯・今節
    "venue_res_s", "jcd_num", "hour", "meet_n", "meet_last",
    # 隣の艇（内側・外側）と1コースの艇
    "in_r_st_mean", "in_r_course_win", "in_st_course", "in_nat_win", "in_form90_top3",
    "out_r_st_mean", "out_r_course_win", "out_st_course", "out_nat_win", "out_form90_top3",
    "st_vs_in", "st_vs_out", "c1_course_win", "c1_st",
    # レース内の順位・相対値
    "rank_r_course_win", "rank_form90_top3", "rank_ex_time", "form_rel",
    # モーター・ボート（場×番号ごとの最近の成績。コース補正後）と展示タイムの「いつもとの差」
    "motor_res_s", "motor_n", "motor_res30_s", "boat_res_s", "motor_exd_s", "ex_vs_motor", "ex_vs_racer",
    "racer_exd", "motor_rel", "rank_motor",
]
# 直前情報（スタート展示・チルト・部品交換・気象）。data/before.parquet がある期間だけ値が入る
BEFORE_FEATURES = [
    "ex_st", "ex_st_rel", "ex_st_rank", "ex_st_vs_avg", "ex_course_changed",
    "tilt", "tilt_rel", "prop_new", "parts_n", "parts_major", "parts_ring", "parts_elec", "parts_carb",
    "adj_weight", "weight_diff", "temp", "water_temp", "temp_diff", "wind_rel", "wind_along", "wind_cross",
    "in_ex_st", "out_ex_st", "ex_st_vs_in",
]
BI_COLS = ["bi_ex_weight", "bi_adj_weight", "bi_tilt", "bi_propeller", "bi_parts", "bi_ex_course", "bi_ex_st",
           "bi_w_temp", "bi_w_wind", "bi_w_wind_dir", "bi_w_water_temp", "bi_w_course_dir"]
PL_TEMPS = (1.0, 0.75, 0.55)  # 学習時の段階ごとの温度（学習後に検証期間で改めて推定する）


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
    if BEFORE.exists():
        b = pd.read_parquet(BEFORE)
        b = b.rename(columns={c: f"bi_{c}" for c in b.columns if c not in ("hd", "jcd", "rno", "waku")})
        df = df.merge(b[["hd", "jcd", "rno", "waku"] + [c for c in BI_COLS if c in b]],
                      on=["hd", "jcd", "rno", "waku"], how="left")
    df = df.sort_values(["date", "race_id", "waku"]).reset_index(drop=True)
    return df


def _rolling_by(df: pd.DataFrame, keys: list[str], cols: dict[str, str], window: str) -> pd.DataFrame:
    """keys ごとに、その日より前 window 期間の集計（closed='left' で当日を含まない）"""
    out = pd.DataFrame(index=df.index)
    g = df[keys + ["date"] + list(cols)].copy()
    g = g.sort_values(keys + ["date"])
    for col, how in cols.items():
        # sort=True: 結果の並びが g（keys→date 順）と一致する。sort=False は複数キーで並びが崩れることがある
        rolled = (g.set_index("date").groupby(keys, sort=True)[col]
                  .rolling(window, closed="left").agg(how))
        for lv, k in enumerate(keys):
            if not (rolled.index.get_level_values(lv).to_numpy() == g[k].to_numpy()).all():
                raise RuntimeError(f"rolling の並びが一致しません: {keys}")
        out.loc[g.index, f"{col}_{how}_{window}"] = rolled.to_numpy()
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _basic(df)
    priors = course_priors(df)
    df = _residuals(df, priors)
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
    f90 = _rolling_by(df, ["toban"], {"res_top3": "mean", "res_win": "mean"}, "90D")
    f30 = _rolling_by(df, ["toban"], {"res_top3": "mean", "one": "sum"}, "30D")
    f365 = _rolling_by(df, ["toban"], {"res_top3": "mean"}, "365D")
    stc = _rolling_by(df, ["toban", "course"], {"st": "mean"}, "365D")
    sts = _rolling_by(df, ["toban"], {"st": "std"}, "180D")
    ven = _rolling_by(df, ["toban", "jcd"], {"res_top3": "mean", "one": "sum"}, "730D")
    df["form90_top3"], df["form90_win"] = f90["res_top3_mean_90D"], f90["res_win_mean_90D"]
    df["form30_top3"], df["n30"] = f30["res_top3_mean_30D"], f30["one_sum_30D"]
    df["form365_top3"] = f365["res_top3_mean_365D"]
    df["st_course"], df["st_std"] = stc["st_mean_365D"], sts["st_std_180D"]
    df["venue_res"], df["venue_n"] = ven["res_top3_mean_730D"], ven["one_sum_730D"]
    m = df["motor_no"].notna()
    mo = _rolling_by(df[m], ["jcd", "motor_no"], {"res_top3": "mean", "one": "sum", "ex_time_diff": "mean"}, "90D")
    mo30 = _rolling_by(df[m], ["jcd", "motor_no"], {"res_top3": "mean", "one": "sum"}, "30D")
    df.loc[m, "motor_res"], df.loc[m, "motor_n"] = mo["res_top3_mean_90D"], mo["one_sum_90D"]
    df.loc[m, "motor_exd"] = mo["ex_time_diff_mean_90D"]
    df.loc[m, "motor_res30"], df.loc[m, "motor_n30"] = mo30["res_top3_mean_30D"], mo30["one_sum_30D"]
    b = df["boat_no"].notna()
    bo = _rolling_by(df[b], ["jcd", "boat_no"], {"res_top3": "mean", "one": "sum"}, "90D")
    df.loc[b, "boat_res"], df.loc[b, "boat_n"] = bo["res_top3_mean_90D"], bo["one_sum_90D"]
    df["racer_exd"] = _rolling_by(df, ["toban"], {"ex_time_diff": "mean"}, "90D")["ex_time_diff_mean_90D"]
    return _finish(df, priors)


def _residuals(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """着順からコースの有利不利を除いた成績（調子の集計用）"""
    if "top3" in df:
        df["res_top3"] = df["top3"] - df["course"].map(lambda c: priors["top3"].get(int(c), 0.5))
        df["res_win"] = df["win"] - df["course"].map(lambda c: priors["win"].get(int(c), 1 / 6))
    return df


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
    df["n30"] = df["n30"].fillna(0)
    vn = df["venue_n"].fillna(0)
    df["venue_res_s"] = df["venue_res"].fillna(0) * vn / (vn + 15)  # 少ないサンプルは0に寄せる
    df["form_rel"] = df["form90_top3"] - df.groupby("race_id")["form90_top3"].transform("mean")
    # --- モーター・ボート（少ないサンプルは0に寄せる）
    def shrink(v, n, k):
        n = df[n].fillna(0) if n in df else 0
        return df[v].fillna(0) * n / (n + k) if v in df else np.nan
    df["motor_n"] = df["motor_n"].fillna(0) if "motor_n" in df else 0.0
    df["motor_res_s"] = shrink("motor_res", "motor_n", 20)
    df["motor_res30_s"] = shrink("motor_res30", "motor_n30", 10)
    df["boat_res_s"] = shrink("boat_res", "boat_n", 20)
    df["motor_exd_s"] = shrink("motor_exd", "motor_n", 20)
    df["ex_vs_motor"] = df["ex_time_diff"] - df["motor_exd_s"]  # 今日の展示がそのモーターのいつもより速いか
    df["ex_vs_racer"] = df["ex_time_diff"] - df["racer_exd"]    # 今日の展示がその選手のいつもより速いか
    df["motor_rel"] = df["motor_res_s"] - df.groupby("race_id")["motor_res_s"].transform("mean")
    df["rank_motor"] = df.groupby("race_id")["motor_res_s"].rank(ascending=False, method="average")
    # --- 隣の艇（進入コース±1）と1コースの艇の値
    keyed = df.set_index([df["race_id"], df["course"]])

    def at(col, course):
        v = keyed[col]
        v = v[~v.index.duplicated()]
        return v.reindex(pd.MultiIndex.from_arrays([df["race_id"], course])).to_numpy()
    for col in ("r_st_mean", "r_course_win", "st_course", "nat_win", "form90_top3"):
        df[f"in_{col}"] = at(col, df["course"] - 1)
        df[f"out_{col}"] = at(col, df["course"] + 1)
    df["st_vs_in"] = df["st_course"] - df["in_st_course"]
    df["st_vs_out"] = df["st_course"] - df["out_st_course"]
    one = pd.Series(1, index=df.index)
    df["c1_course_win"] = at("r_course_win", one)
    df["c1_st"] = at("st_course", one)
    for col in ("r_course_win", "form90_top3"):
        df[f"rank_{col}"] = df.groupby("race_id")[col].rank(ascending=False, method="average")
    ex = df["ex_time"].where(df["ex_time"] > 0)
    df["rank_ex_time"] = ex.groupby(df["race_id"]).rank(ascending=False, method="average")
    df = df.copy()  # 列の追加が多く断片化するのでまとめる
    # --- 直前情報（スタート展示）と普段のST・隣の艇との比較
    df["ex_st_vs_avg"] = df["ex_st"] - df["st_course"]
    df["in_ex_st"] = at("ex_st", df["course"] - 1)
    df["out_ex_st"] = at("ex_st", df["course"] + 1)
    df["ex_st_vs_in"] = df["ex_st"] - df["in_ex_st"]
    return df


MAJOR_PARTS = ("ピストン", "シリンダ", "シャフト", "キャリボディ", "ギア")


def _before_basic(df: pd.DataFrame) -> pd.DataFrame:
    """直前情報の列（bi_*）→ 特徴量。列が無ければ NaN"""
    for c in BI_COLS:
        if c not in df:
            df[c] = np.nan
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    df["ex_st"] = num("bi_ex_st")
    df["ex_st_rel"] = df["ex_st"] - df.groupby("race_id")["ex_st"].transform("mean")
    df["ex_st_rank"] = df.groupby("race_id")["ex_st"].rank(method="average")
    exc = num("bi_ex_course")
    df["ex_course_changed"] = np.where(exc.notna(), (exc != df["waku"]).astype(float), np.nan)
    df["tilt"] = num("bi_tilt")
    df["tilt_rel"] = df["tilt"] - df.groupby("race_id")["tilt"].transform("mean")
    prop = df["bi_propeller"]
    df["prop_new"] = np.where(prop.notna(), prop.astype(str).str.contains("新").astype(float), np.nan)
    parts = df["bi_parts"]
    has = parts.notna()
    ps = parts.fillna("").astype(str)
    df["parts_n"] = np.where(has, ps.map(lambda x: len([t for t in x.split("/") if t])), np.nan)
    df["parts_major"] = np.where(has, ps.map(lambda x: float(any(k in x for k in MAJOR_PARTS))), np.nan)
    for name, key in (("parts_ring", "リング"), ("parts_elec", "電気"), ("parts_carb", "キャブ")):
        df[name] = np.where(has, ps.str.contains(key).astype(float), np.nan)
    df["adj_weight"] = num("bi_adj_weight")
    df["weight_diff"] = num("bi_ex_weight") - pd.to_numeric(df["weight"], errors="coerce")
    df["temp"], df["water_temp"] = num("bi_w_temp"), num("bi_w_water_temp")
    df["temp_diff"] = df["temp"] - df["water_temp"]
    # 風向き：公式サイトの画像番号（16方位）。コースの向きとの差で「追い風・向かい風・横風」を表す
    rel = (num("bi_w_wind_dir") - num("bi_w_course_dir")) % 16
    wind = num("bi_w_wind")
    df["wind_rel"] = rel
    df["wind_along"] = wind * np.cos(2 * np.pi * rel / 16)
    df["wind_cross"] = wind * np.sin(2 * np.pi * rel / 16)
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
    df["jcd_num"] = pd.to_numeric(df["jcd"], errors="coerce")
    df["hour"] = (pd.to_numeric(df["deadline"].astype(str).str.split(":").str[0], errors="coerce")
                  if "deadline" in df else np.nan)
    df["meet_n"] = df["meet_results"].map(lambda s: sum(c in "123456" for c in str(s)))
    df["meet_last"] = df["meet_results"].map(
        lambda s: next((float(c) for c in reversed(str(s)) if c in "123456"), np.nan))
    # --- 展示タイム（レース内の差・順位）
    ex = df["ex_time"].where(df["ex_time"] > 0)
    df["ex_time_diff"] = ex - ex.groupby(df["race_id"]).transform("mean")
    df["ex_time_rank"] = ex.groupby(df["race_id"]).rank(method="average")
    if "finish" in df:
        df["win"] = (df["finish"] == 1).astype(float)
        df["top3"] = (df["finish"] <= 3).astype(float)
        df["finish_f"] = df["finish"].astype(float)
    df["one"] = 1.0
    df = _before_basic(df)
    return df


# ---------------------------------------------------------------- 本番（当日のレース）用

def snapshot(hist: pd.DataFrame, asof: str, priors: dict) -> dict:
    """asof 日の朝時点の過去成績集計。add_features の rolling(closed='left') と同じ定義"""
    t = pd.Timestamp(asof)
    h730 = _residuals(_basic(hist[(hist["date"] < t) & (hist["date"] >= t - pd.Timedelta("730D"))]), priors)
    h = h730[h730["date"] >= t - pd.Timedelta("365D")]
    h180 = h[h["date"] >= t - pd.Timedelta("180D")]
    h90 = h[h["date"] >= t - pd.Timedelta("90D")]
    h60 = h[h["date"] >= t - pd.Timedelta("60D")]
    h30 = h[h["date"] >= t - pd.Timedelta("30D")]
    return {
        "f90": h90.groupby("toban").agg(form90_top3=("res_top3", "mean"), form90_win=("res_win", "mean")),
        "f30": h30.groupby("toban").agg(form30_top3=("res_top3", "mean"), n30=("one", "sum")),
        "f365": h.groupby("toban").agg(form365_top3=("res_top3", "mean")),
        "stc": h.groupby(["toban", "course"]).agg(st_course=("st", "mean")),
        "sts": h180.groupby("toban").agg(st_std=("st", "std")),
        "ven": h730.groupby(["toban", "jcd"]).agg(venue_res=("res_top3", "mean"), venue_n=("one", "sum")),
        "mo": h90.groupby(["jcd", "motor_no"]).agg(motor_res=("res_top3", "mean"), motor_n=("one", "sum"),
                                                   motor_exd=("ex_time_diff", "mean")),
        "mo30": h30.groupby(["jcd", "motor_no"]).agg(motor_res30=("res_top3", "mean"), motor_n30=("one", "sum")),
        "bo": h90.groupby(["jcd", "boat_no"]).agg(boat_res=("res_top3", "mean"), boat_n=("one", "sum")),
        "rex": h90.groupby("toban").agg(racer_exd=("ex_time_diff", "mean")),
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
    df = df.join(snap["f90"], on="toban").join(snap["f30"], on="toban").join(snap["f365"], on="toban")
    df = df.join(snap["stc"], on=["toban", "course"]).join(snap["sts"], on="toban")
    df = df.join(snap["ven"], on=["toban", "jcd"])
    df["motor_no"] = pd.to_numeric(df["motor_no"], errors="coerce").astype(float)
    df["boat_no"] = pd.to_numeric(df["boat_no"], errors="coerce").astype(float)
    df = df.join(snap["mo"], on=["jcd", "motor_no"]).join(snap["mo30"], on=["jcd", "motor_no"])
    df = df.join(snap["bo"], on=["jcd", "boat_no"]).join(snap["rex"], on="toban")
    df = _finish(df, priors)
    cols = FEATURES + BEFORE_FEATURES
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
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
            "deadline": info.deadlines.get(info.rno, ""),
            "motor_no": b.motor_no, "boat_no": b.boat_no,
            **{f"bi_{k}": v for k, v in (getattr(b, "before", None) or {}).items()},
            **{f"bi_w_{k}": v for k, v in (getattr(info, "weather", None) or {}).items()},
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
        self.snap = snapshot(self.hist, hd, self.priors)
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


def pl_objective(Y: np.ndarray, temps=PL_TEMPS):
    """LightGBM 用の目的関数：1〜3着の Plackett-Luce 負の対数尤度の勾配と（対角の）ヘッセ行列。
    データは race_id・waku 順に1レース6行で並んでいること"""
    def f(preds, _ds):
        S = preds.reshape(-1, 6)
        i = np.arange(len(S))
        grad, hess = np.zeros_like(S), np.zeros_like(S)
        alive = np.ones_like(S, dtype=bool)
        for k, t in enumerate(temps):
            z = np.where(alive, t * S, -np.inf)
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            grad += t * p
            grad[i, Y[:, k]] -= t
            hess += t * t * p * (1 - p)
            alive[i, Y[:, k]] = False
        return grad.ravel(), np.maximum(hess.ravel(), 1e-6)
    return f


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
    feats = FEATURES + (BEFORE_FEATURES if BEFORE.exists() else [])
    print(f"特徴量を作成中…（{len(feats)}個）", flush=True)
    hist = add_features(load_history())
    priors = course_priors(hist)
    df = clean_races(hist)
    df = df[df["hd"] >= TRAIN_START]
    tr = df[df["hd"] <= TRAIN_END].sort_values(["race_id", "waku"])
    va = df[(df["hd"] > TRAIN_END) & (df["hd"] <= VALID_END)].sort_values(["race_id", "waku"])
    print(f"学習 {tr['race_id'].nunique()} R / 検証 {va['race_id'].nunique()} R", flush=True)
    _, _, Ytr = to_matrix(tr, np.zeros(len(tr)))
    _, _, Yva = to_matrix(va, np.zeros(len(va)))
    params = dict(objective=pl_objective(Ytr), metric="None", learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, seed=0)
    dtr = lgb.Dataset(tr[feats], free_raw_data=False)
    dva = lgb.Dataset(va[feats], reference=dtr)

    def feval(preds, _ds):
        return "pl_loglik", top3_loglik(preds.reshape(-1, 6), Yva, PL_TEMPS), True
    booster = lgb.train(params, dtr, 5000, valid_sets=[dva], feval=feval,
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    _, S, Y = to_matrix(va, booster.predict(va[feats], num_iteration=booster.best_iteration))
    temps = fit_temps(S, Y)
    print(f"温度 {temps}  検証の対数尤度 {top3_loglik(S, Y, temps):.4f}", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_DIR / "lgb.txt"), num_iteration=booster.best_iteration)
    imp = dict(zip(feats, booster.feature_importance("gain").round(1).tolist()))
    (MODEL_DIR / "meta.json").write_text(encoding="utf-8", data=json.dumps(
        {"temps": temps, "features": feats, "train_end": TRAIN_END, "valid_end": VALID_END,
         "objective": "plackett_luce", "best_iteration": booster.best_iteration, "importance": imp,
         "priors": priors}, ensure_ascii=False, indent=1))
    print("保存しました", MODEL_DIR)


class Model:
    def __init__(self, model_dir: Path | None = None):
        import lightgbm as lgb
        model_dir = model_dir or MODEL_DIR
        self.booster = lgb.Booster(model_file=str(model_dir / "lgb.txt"))
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        # JSON ではキーが文字列になるので、コース番号(int)に戻す（戻さないと本番で事前値が使われない）
        self.meta["priors"] = {k: {int(c): v for c, v in d.items()} for k, d in self.meta["priors"].items()}
        self.temps = tuple(self.meta["temps"])

    def scores(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df[self.meta["features"]])

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
