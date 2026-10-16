"""実験用：設定ごとに学習→検証期間で温度を決め→テスト期間・オッズのあるレースで市場と比較"""
import json
import sys
import time
from itertools import permutations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from boatbot.ml import FEATURES, TRAIN_END, TRAIN_START, VALID_END, fit_temps, to_matrix, top3_loglik

SPLIT = "20260828"
NEW = ["x_form90_top3", "x_form90_win", "x_form30_top3", "x_n30", "x_form365_top3", "x_st_course", "x_st_std",
       "x_motor_res_s", "x_motor_n", "x_boat_res", "x_venue_res_s", "x_jcd", "x_wind_dir", "x_hour",
       "x_meet_n", "x_meet_last", "x_ex_time_dev",
       "x_in_r_st_mean", "x_in_r_course_win", "x_in_x_st_course", "x_in_nat_win", "x_in_x_form90_top3",
       "x_out_r_st_mean", "x_out_r_course_win", "x_out_x_st_course", "x_out_nat_win", "x_out_x_form90_top3",
       "x_st_vs_in", "x_st_vs_out", "x_c1_course_win", "x_c1_st",
       "x_rank_r_course_win", "x_rank_x_form90_top3", "x_rank_x_motor_res_s", "x_rank_ex_time",
       "x_form_rel", "x_motor_rel"]
LIVE_NEW = [f for f in NEW if not any(k in f for k in ("motor", "boat", "wind_dir", "ex_time_dev"))]
BASE_PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 3], learning_rate=0.05,
                   num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                   bagging_freq=1, lambdarank_truncation_level=6, verbose=-1, num_threads=16)
TRI = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
TRI_IDX = [(a - 1) * 36 + (b - 1) * 6 + (c - 1) for a, b, c in permutations(range(1, 7), 3)]


def tri_probs(S, temps):
    """S (N,6) → (N,6,6,6) をベクトル化して計算"""
    t1, t2, t3 = temps
    S = S - S.max(1, keepdims=True)
    e1, e2, e3 = np.exp(t1 * S), np.exp(t2 * S), np.exp(t3 * S)
    p1 = e1 / e1.sum(1, keepdims=True)
    N = len(S)
    out = np.zeros((N, 6, 6, 6))
    for a in range(6):
        r2 = e2.sum(1) - e2[:, a]
        for b in range(6):
            if b == a:
                continue
            p2 = e2[:, b] / r2
            r3 = e3.sum(1) - e3[:, a] - e3[:, b]
            for c in range(6):
                if c in (a, b):
                    continue
                out[:, a, b, c] = p1[:, a] * p2 * e3[:, c] / r3
    return out.reshape(N, 216)[:, TRI_IDX]


class _Ens:
    """複数の booster の平均（seed違い）"""
    def __init__(self, bsts):
        self.bsts = bsts
        self.best_iteration = int(np.mean([b.best_iteration for b in bsts]))

    def predict(self, X, num_iteration=None):
        return np.mean([b.predict(X, num_iteration=b.best_iteration) for b in self.bsts], axis=0)

    def feature_importance(self, kind):
        return np.sum([b.feature_importance(kind) for b in self.bsts], axis=0)


def load():
    df = pd.read_parquet("data/exp_features.parquet")
    df = df[df["hd"] >= TRAIN_START].copy()
    df["label"] = (7 - df["finish"]).clip(lower=0).fillna(0).astype(int)
    return df


def market_table(te_rid):
    """オッズのあるレース: race_id → 市場確率(120,)"""
    rows = {}
    for rid in te_rid:
        f = Path("data/odds") / rid[:8] / f"{rid[8:10]}_{rid[10:]}.json"
        if not f.exists():
            continue
        o = json.loads(f.read_text())
        if len(o) < 100:
            continue
        v = np.array([o.get(c, 0.0) for c in TRI], dtype=float)
        q = np.where(v > 0, 1 / np.where(v > 0, v, 1), 0)
        rows[rid] = q / q.sum()
    return rows


def _Y(d):
    fin = d["finish"].to_numpy().reshape(-1, 6)
    return np.stack([np.argmax(fin == k, axis=1) for k in (1, 2, 3)], axis=1)


def pl_objective(Y, temps):
    """Plackett-Luce（1〜3着）の負の対数尤度の勾配・ヘッセ行列（対角近似）"""
    def f(preds, ds):
        S = preds.reshape(-1, 6)
        N = len(S)
        i = np.arange(N)
        grad, hess = np.zeros_like(S), np.zeros_like(S)
        alive = np.ones_like(S, dtype=bool)
        for k, t in enumerate(temps):
            z = np.where(alive, t * S, -np.inf)
            z = z - z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            grad += t * p
            grad[i, Y[:, k]] -= t
            hess += t * t * p * (1 - p)
            alive[i, Y[:, k]] = False
        return grad.ravel(), np.maximum(hess.ravel(), 1e-6)
    return f


def run(name, feats, params, df, rounds=5000, pl_temps=None, seeds=(0,)):
    t0 = time.time()
    tr = df[df["hd"] <= TRAIN_END].sort_values(["race_id", "waku"])
    va = df[(df["hd"] > TRAIN_END) & (df["hd"] <= VALID_END)].sort_values(["race_id", "waku"])
    te = df[df["hd"] > VALID_END].sort_values(["race_id", "waku"])
    grp = lambda d: d.groupby("race_id", sort=False).size().to_numpy()
    if pl_temps is None:
        dtr = lgb.Dataset(tr[feats], tr["label"], group=grp(tr))
        dva = lgb.Dataset(va[feats], va["label"], group=grp(va))
        bst = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        Ytr, Yva = _Y(tr), _Y(va)
        dtr = lgb.Dataset(tr[feats], tr["label"], free_raw_data=False)
        dva = lgb.Dataset(va[feats], va["label"], reference=dtr)

        def feval(preds, ds):
            return "pl", top3_loglik(preds.reshape(-1, 6), Yva, pl_temps), True
        p = {k: v for k, v in params.items() if k not in ("objective", "metric", "ndcg_eval_at", "lambdarank_truncation_level")}
        bsts = []
        for sd in seeds:
            b = lgb.train({**p, "objective": pl_objective(Ytr, pl_temps), "metric": "None", "seed": sd},
                          dtr, rounds, valid_sets=[dva], feval=feval,
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            bsts.append(b)
        bst = _Ens(bsts)
    _, Sv, Yv = to_matrix(va, bst.predict(va[feats], num_iteration=bst.best_iteration))
    temps = fit_temps(Sv, Yv)
    rid, St, Yt = to_matrix(te, bst.predict(te[feats], num_iteration=bst.best_iteration))
    res = {"name": name, "iters": bst.best_iteration, "temps": [round(t, 3) for t in temps],
           "valid_ll": round(top3_loglik(Sv, Yv, temps), 4), "test_ll": round(top3_loglik(St, Yt, temps), 4),
           "test_win_acc": round(float(np.mean(St.argmax(1) == Yt[:, 0])), 4), "sec": round(time.time() - t0)}
    return res, bst, (rid, St, Yt, temps)


def market_compare(rid, St, Yt, temps, mk):
    keep = np.array([r in mk for r in rid])
    rid, St, Yt = rid[keep], St[keep], Yt[keep]
    Pm = tri_probs(St, temps)
    Q = np.stack([mk[r] for r in rid])
    act = np.array([TRI.index(f"{a+1}-{b+1}-{c+1}") for a, b, c in Yt])
    i = np.arange(len(act))
    lm, lq = np.log(np.maximum(Pm[i, act], 1e-12)), np.log(np.maximum(Q[i, act], 1e-12))
    # 混ぜる: p ∝ Pm^w1 * Q^w2（前半で w を決め、後半で評価）
    first = np.array([r[:8] <= SPLIT for r in rid])
    LPm, LQ = np.log(np.maximum(Pm, 1e-12)), np.log(np.maximum(Q, 1e-12))

    def ll(w, m):
        z = w[0] * LPm[m] + w[1] * LQ[m]
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return float(lp[np.arange(m.sum()), act[m]].mean())
    w = minimize(lambda w: -ll(w, first), x0=[0.3, 0.7], method="Nelder-Mead").x
    wall = minimize(lambda w: -ll(w, np.ones_like(first)), x0=[0.3, 0.7], method="Nelder-Mead").x
    second = ~first
    # 後半での差の標準誤差（レースごとの差の平均の SE）
    z = w[0] * LPm + w[1] * LQ
    z = z - z.max(1, keepdims=True)
    lb = (z - np.log(np.exp(z).sum(1, keepdims=True)))[i, act]
    d = lb[second] - lq[second]
    return {"races": int(len(act)), "model_ll": round(float(lm.mean()), 4), "market_ll": round(float(lq.mean()), 4),
            "blend_w": [round(x, 3) for x in w], "blend_w_all": [round(x, 3) for x in wall],
            "verify_races": int(second.sum()),
            "verify_model_ll": round(float(lm[second].mean()), 4), "verify_market_ll": round(float(lq[second].mean()), 4),
            "verify_blend_ll": round(float(lb[second].mean()), 4),
            "verify_blend_gain": round(float(d.mean()), 4), "verify_blend_gain_se": round(float(d.std() / np.sqrt(len(d))), 4)}


if __name__ == "__main__":
    df = load()
    te_rid = df[df["hd"] > VALID_END]["race_id"].unique()
    mk = market_table(te_rid)
    configs = {
        "v0_現行": (FEATURES, BASE_PARAMS),
        "v1_特徴追加": (FEATURES + NEW, BASE_PARAMS),
        "v2_特徴追加_大きめ": (FEATURES + NEW, {**BASE_PARAMS, "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 100}),
        "v3_特徴追加_xendcg": (FEATURES + NEW, {**BASE_PARAMS, "objective": "rank_xendcg"}),
        "v4_PL直接": (FEATURES + NEW, BASE_PARAMS, dict(pl_temps=(1.0, 0.75, 0.55))),
        "v5_PL直接_lr02": (FEATURES + NEW, {**BASE_PARAMS, "learning_rate": 0.02, "min_data_in_leaf": 400}, dict(pl_temps=(1.0, 0.75, 0.55))),
        "v6_PL直接_現行特徴": (FEATURES, BASE_PARAMS, dict(pl_temps=(1.0, 0.75, 0.55))),
        "v8_PL直接_本番で使える特徴": (FEATURES + LIVE_NEW, BASE_PARAMS, dict(pl_temps=(1.0, 0.75, 0.55))),
        "v7_PL直接_3seed": (FEATURES + NEW, {**BASE_PARAMS, "learning_rate": 0.02, "min_data_in_leaf": 400}, dict(pl_temps=(1.0, 0.75, 0.55), seeds=(0, 1, 2))),
    }
    sel = sys.argv[1:] or list(configs)
    out = []
    for name in sel:
        feats, params, *kw = configs[name]
        res, bst, (rid, St, Yt, temps) = run(name, feats, params, df, **(kw[0] if kw else {}))
        res["market"] = market_compare(rid, St, Yt, temps, mk)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        imp = pd.Series(bst.feature_importance("gain"), index=feats)
        print("  重要度上位:", ", ".join(f"{k} {v:.1%}" for k, v in (imp / imp.sum()).sort_values(ascending=False).head(15).items()), flush=True)
        out.append(res)
