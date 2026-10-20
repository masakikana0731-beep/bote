"""予想の見せ方（方針2）：モデルの見立てと市場（オッズ）の見立てを艇ごとに並べる。

- 市場の確率 = 3連単オッズの逆数を合計1に直したもの（控除の分は均等に割り戻している）
- 「ほぼ確実な本命」：モデルが2着以内（複勝）・1着（単勝）をほぼ確実と見る艇。
  2年分の検証（docs/policy1_findings.md）では、この条件で買うと回収率100〜104%（ほぼ元返し）だった。
  儲けるためではなく「負けにくい遊び方」の目安として表示する
"""
from __future__ import annotations

SOLID_PLACE = 0.99   # 複勝：モデルの2着以内の確率がこれ以上（2025〜26年の検証で 0.985〜0.993）
SOLID_WIN = 0.95     # 単勝：モデルの1着確率がこれ以上（同 0.925〜0.954）
NOTABLE_DIFF = 0.05  # モデルが市場より5ポイント以上高く見ている艇に印


def _by_boat(tri: dict[str, float]) -> dict[int, list[float]]:
    """3連単の確率 → 艇ごとの [1着, 2着以内, 3着以内] の確率"""
    out = {w: [0.0, 0.0, 0.0] for w in range(1, 7)}
    for combo, p in tri.items():
        a, b, c = (int(x) for x in combo.split("-"))
        out[a][0] += p
        out[a][1] += p
        out[b][1] += p
        for x in (a, b, c):
            out[x][2] += p
    return out


def market_probs(odds: dict[str, float]) -> dict[str, float]:
    inv = {k: 1 / o for k, o in odds.items() if o and o > 0}
    tot = sum(inv.values())
    return {k: v / tot for k, v in inv.items()} if tot else {}


def boat_insight(model_tri: dict[str, float] | None, odds: dict[str, float],
                 model_win: dict[int, float] | None = None) -> dict:
    """艇ごとのモデルと市場の確率、市場より高く評価している艇、ほぼ確実な本命"""
    mk = _by_boat(market_probs(odds)) if odds else None
    md = _by_boat(model_tri) if model_tri else None
    boats = []
    for w in range(1, 7):
        mw = md[w][0] if md else (model_win or {}).get(w)
        row = {"waku": w,
               "model_win": round(mw, 3) if mw is not None else None,
               "model_top2": round(md[w][1], 3) if md else None,
               "market_win": round(mk[w][0], 3) if mk else None,
               "market_top2": round(mk[w][1], 3) if mk else None}
        if row["model_win"] is not None and row["market_win"] is not None:
            row["diff_win"] = round(row["model_win"] - row["market_win"], 3)
            row["notable"] = row["diff_win"] >= NOTABLE_DIFF
        boats.append(row)
    solid = []
    if md:
        for w in range(1, 7):
            if md[w][1] >= SOLID_PLACE:
                solid.append({"waku": w, "type": "複勝", "prob": round(md[w][1], 3)})
            if md[w][0] >= SOLID_WIN:
                solid.append({"waku": w, "type": "単勝", "prob": round(md[w][0], 3)})
    return {"boats": boats, "solid": solid}
