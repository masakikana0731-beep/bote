"""ウォークフォワード：四半期ごとに「その直前までのデータだけ」で学習したモデルで、次の四半期を予想する。
  python wf_models.py prepare            # 特徴量を作って保存
  python wf_models.py fold 20251231 20260331   # cutoff までで学習（最後の75日は検証）→ (cutoff, next] を予想
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from boatbot.ml import (FEATURES, PL_TEMPS, add_features, clean_races, fit_temps, load_history, pl_objective,
                        to_matrix, top3_loglik)

FEAT = Path("data/wf_features.parquet")
OUT = Path("data/wf")
START = "20180401"

if sys.argv[1] == "prepare":
    df = clean_races(add_features(load_history()))
    df = df[df["hd"] >= START]
    cols = list(dict.fromkeys(FEATURES + ["race_id", "hd", "waku", "finish"]))
    df[cols].to_parquet(FEAT)
    print("saved", len(df))
    sys.exit(0)

cutoff, nxt = sys.argv[2], sys.argv[3]
valid_from = (datetime.strptime(cutoff, "%Y%m%d") - timedelta(days=75)).strftime("%Y%m%d")
df = pd.read_parquet(FEAT)
tr = df[(df.hd <= valid_from)].sort_values(["race_id", "waku"])
va = df[(df.hd > valid_from) & (df.hd <= cutoff)].sort_values(["race_id", "waku"])
te = df[(df.hd > cutoff) & (df.hd <= nxt)].sort_values(["race_id", "waku"])
del df
_, _, Ytr = to_matrix(tr, np.zeros(len(tr)))
_, _, Yva = to_matrix(va, np.zeros(len(va)))
params = dict(objective=pl_objective(Ytr), metric="None", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0, num_threads=6)
dtr = lgb.Dataset(tr[FEATURES], free_raw_data=False)
dva = lgb.Dataset(va[FEATURES], reference=dtr)
bst = lgb.train(params, dtr, 5000, valid_sets=[dva],
                feval=lambda p, _d: ("pl", top3_loglik(p.reshape(-1, 6), Yva, PL_TEMPS), True),
                callbacks=[lgb.early_stopping(150, verbose=False)])
_, Sv, Yv = to_matrix(va, bst.predict(va[FEATURES], num_iteration=bst.best_iteration))
temps = fit_temps(Sv, Yv)
OUT.mkdir(parents=True, exist_ok=True)
sc = te[["race_id", "hd", "waku", "finish"]].copy()
sc["score"] = bst.predict(te[FEATURES], num_iteration=bst.best_iteration)
sc.to_parquet(OUT / f"scores_{cutoff}.parquet")
rid, S, Y = to_matrix(te, sc["score"].to_numpy())
info = {"cutoff": cutoff, "next": nxt, "train_races": int(tr.race_id.nunique()), "valid_races": int(va.race_id.nunique()),
        "test_races": len(S), "iters": bst.best_iteration, "temps": temps, "test_ll": top3_loglik(S, Y, temps)}
(OUT / f"info_{cutoff}.json").write_text(json.dumps(info))
print(json.dumps(info), flush=True)
