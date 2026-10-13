"""予想ロジック：各艇の強さ → 1着確率 → 3連単確率 → 買い目選び → 金額配分。

精度を追求したモデルではなく「設定スライダで味付けを変えて遊ぶ」ためのもの。
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from itertools import permutations

from .scraper import Boat

# 全国平均のコース別1着率（おおよその値。年や場によって変わる）
COURSE_WIN_RATE = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03}


def hard_daily_cap() -> int | None:
    """環境変数 BOATBOT_HARD_DAILY_CAP（円）。画面からはこれ以上に設定できない最終上限。"""
    v = os.environ.get("BOATBOT_HARD_DAILY_CAP", "").strip()
    return int(v) if v.isdigit() else None


@dataclass
class Settings:
    # --- お金まわり（円） ---
    daily_budget: int = 3000
    race_budget: int = 500
    max_races_per_day: int = 10
    # --- スライダ（0〜100） ---
    gamble: int = 30          # ギャンブル性：狙う配当の目安 0=約8倍 / 100=約300倍
    odds_width: int = 30      # 目安の幅：0=目安ぴったり / 100=幅広く
    selectivity: int = 0      # 厳選度：買い目ごとの期待値(確率×オッズ)の下限 0=制限なし / 100=1.5以上
    skip_fav: int = 0         # 堅いレースを見送る：0=見送らない / 100=1着確率50%超の艇がいたら見送る
    max_wind: int = 0         # 風速が何m以上なら見送るか（0=見送らない）
    crowd: int = 0            # 人気(オッズ)を信じる度合い：0=独自予想のみ / 100=人気通り
    whim: int = 0             # 気まぐれ度：買い目にランダム性を混ぜる
    predictor: str = "ml"     # ml=過去3年で学習したモデル（無ければ簡易版）/ simple=重視スライダの簡易版
    w_course: int = 70        # 重視：コース(イン有利)
    w_national: int = 60      # 重視：全国勝率
    w_local: int = 30         # 重視：当地勝率
    w_motor: int = 40         # 重視：モーター
    w_exhibit: int = 40       # 重視：展示タイム
    w_start: int = 30         # 重視：スタート
    # --- その他 ---
    tickets: int = 5          # 1レースの点数（上限）
    allocation: str = "dutch"  # equal=均等 / dutch=どれが当たっても払戻が近くなる / prob=確率比例
    lead_minutes: int = 8     # 締切何分前に予想するか
    venues: str = ""          # 対象場コード（カンマ区切り、空=全場）
    enabled: bool = True
    auto_mode: str = "off"    # 自動投資：off=通知だけ / paper=仮想で自動購入 / real=テレボートで自動購入

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                cur = getattr(s, k)
                setattr(s, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
        return s.clamp()

    def clamp(self) -> "Settings":
        for k in ("gamble", "odds_width", "selectivity", "skip_fav", "crowd", "whim", "w_course", "w_national",
                  "w_local", "w_motor", "w_exhibit", "w_start"):
            setattr(self, k, max(0, min(100, int(getattr(self, k)))))
        self.max_wind = max(0, min(30, self.max_wind))
        if self.predictor not in ("ml", "simple"):
            self.predictor = "ml"
        self.daily_budget = max(0, self.daily_budget)
        self.race_budget = max(100, self.race_budget)
        self.tickets = max(1, min(20, self.tickets))
        self.lead_minutes = max(2, min(30, self.lead_minutes))
        if self.allocation not in ("equal", "dutch", "prob"):
            self.allocation = "dutch"
        if self.auto_mode not in ("off", "paper", "real"):
            self.auto_mode = "off"
        cap = hard_daily_cap()
        if cap is not None:
            self.daily_budget = min(self.daily_budget, cap)
            self.race_budget = min(self.race_budget, max(100, cap))
        return self

    def venue_set(self) -> set[str]:
        return {v.strip().zfill(2) for v in self.venues.split(",") if v.strip()}


def _z(values: list[float | None], invert: bool = False) -> list[float]:
    """平均0・標準偏差1に正規化。欠損は0（平均扱い）。"""
    xs = [v for v in values if v is not None]
    if len(xs) < 2:
        return [0.0] * len(values)
    mu = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs)) or 1.0
    out = [0.0 if v is None else (v - mu) / sd for v in values]
    return [-x for x in out] if invert else out


def strengths(boats: list[Boat], s: Settings) -> dict[int, float]:
    loc = [b.loc_win if b.loc_win else None for b in boats]  # 当地0.00は未出走なので欠損扱い
    st = [b.ex_st if b.ex_st is not None and b.ex_st >= 0 else b.avg_st for b in boats]
    feats = {
        "national": _z([b.nat_win for b in boats]),
        "local": _z(loc),
        "motor": _z([b.motor_2r for b in boats]),
        "exhibit": _z([b.ex_time for b in boats], invert=True),
        "start": _z(st, invert=True),
    }
    w = {"national": s.w_national, "local": s.w_local, "motor": s.w_motor,
         "exhibit": s.w_exhibit, "start": s.w_start}
    out = {}
    for i, b in enumerate(boats):
        course = b.course or b.waku
        v = (s.w_course / 100) * math.log(COURSE_WIN_RATE.get(course, 0.03) / (1 / 6))
        v += sum((w[k] / 100) * 0.6 * feats[k][i] for k in feats)
        if b.f_count:
            v -= 0.15 * b.f_count  # F持ちはスタートを控えがち
        if b.ex_st is not None and b.ex_st < 0:
            v -= 0.1  # 展示でフライング
        out[b.waku] = v
    return out


def win_probs(st: dict[int, float]) -> dict[int, float]:
    m = max(st.values())
    e = {k: math.exp(v - m) for k, v in st.items()}
    tot = sum(e.values())
    return {k: v / tot for k, v in e.items()}


def trifecta_probs(p: dict[int, float]) -> dict[str, float]:
    """Harville法：1着確率から3連単確率を作る"""
    out = {}
    for a, b, c in permutations(p.keys(), 3):
        pa, pb, pc = p[a], p[b], p[c]
        if pa >= 1 or pa + pb >= 1:
            continue
        out[f"{a}-{b}-{c}"] = pa * pb / (1 - pa) * pc / (1 - pa - pb)
    return out


def blend_with_market(model: dict[str, float], odds: dict[str, float], crowd: int) -> dict[str, float]:
    """独自確率と市場確率(1/オッズ)を幾何平均で混ぜる"""
    if not odds or crowd <= 0:
        return model
    inv = {k: 1 / v for k, v in odds.items()}
    tot = sum(inv.values())
    m = crowd / 100
    mixed = {}
    for k, p in model.items():
        q = inv.get(k, 0) / tot
        if q <= 0:
            continue
        mixed[k] = (p ** (1 - m)) * (q ** m)
    tot = sum(mixed.values()) or 1
    return {k: v / tot for k, v in mixed.items()}


@dataclass
class Ticket:
    combo: str
    prob: float
    odds: float
    amount: int = 0

    @property
    def ev(self) -> float:
        return self.prob * self.odds


TARGET_MIN, TARGET_MAX = 8.0, 300.0  # ギャンブル性 0 / 100 のときの狙う配当（倍）


def odds_sigma(width: int) -> float:
    """目安の幅(0〜100) → 目安からどれくらい外れてよいか（log単位）。0=0.4(約1.5倍) / 100=1.5(約4.5倍)"""
    return 0.4 + 1.1 * max(0, min(100, width)) / 100


def skip_fav_threshold(v: int) -> float:
    """堅いレースを見送る(0〜100) → 1着確率がこれを超える艇がいたら見送る。0=1.0(見送らない) / 100=0.5"""
    return 1.0 - 0.5 * max(0, min(100, v)) / 100


def target_odds(gamble: int) -> float:
    """ギャンブル性(0〜100) → 狙う配当の目安（倍）。対数で等間隔：0=8倍, 50≒49倍, 100=300倍"""
    g = max(0, min(100, gamble)) / 100
    return TARGET_MIN * (TARGET_MAX / TARGET_MIN) ** g


def pick_tickets(probs: dict[str, float], odds: dict[str, float], s: Settings,
                 rng: random.Random | None = None) -> list[Ticket]:
    """「狙う配当の目安」に近いオッズの中から、当たりやすい順に選ぶ。

    score = log(確率) − (log(オッズ) − log(目安))² / (2σ²)
    目安から離れたオッズほど減点される。
    """
    rng = rng or random.Random()
    center = math.log(target_odds(s.gamble))
    sigma = odds_sigma(s.odds_width)
    min_ev = 1.5 * s.selectivity / 100
    noise = s.whim / 100
    scored = []
    for k, p in probs.items():
        o = odds.get(k)
        if not o or p <= 0 or p * o < min_ev:
            continue
        dist = math.log(o) - center
        score = math.log(p) - dist * dist / (2 * sigma ** 2) + rng.gauss(0, 1.5 * noise)
        scored.append((score, Ticket(k, p, o)))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[: s.tickets]]


def allocate(tickets: list[Ticket], budget: int, mode: str) -> list[Ticket]:
    """100円単位で配分。予算は絶対に超えない。予算不足なら点数を減らす。"""
    units = budget // 100
    tickets = tickets[: max(0, units)]
    if not tickets:
        return []
    if mode == "equal":
        w = [1.0] * len(tickets)
    elif mode == "prob":
        w = [t.prob for t in tickets]
    else:  # dutch: 払戻(=金額×オッズ)を揃える
        w = [1 / t.odds for t in tickets]
    tot = sum(w)
    # 各1口は確保し、残りを重みで分配（最大剰余法）
    rest = units - len(tickets)
    raw = [rest * x / tot for x in w]
    base = [1 + int(r) for r in raw]
    left = units - sum(base)
    order = sorted(range(len(tickets)), key=lambda i: -(raw[i] - int(raw[i])))
    for i in order[:left]:
        base[i] += 1
    for t, u in zip(tickets, base):
        t.amount = u * 100
    return tickets


@dataclass
class Plan:
    tickets: list[Ticket]
    skip_reason: str = ""
    hit_prob: float = 0.0
    avg_ev: float = 0.0
    win_probs: dict | None = None

    @property
    def total(self) -> int:
        return sum(t.amount for t in self.tickets)


def make_plan(boats: list[Boat], odds: dict[str, float], s: Settings, budget: int,
              rng: random.Random | None = None, probs: dict[str, float] | None = None,
              wind: float | None = None) -> Plan:
    """probs を渡せば（学習モデルの3連単確率）それを使い、無ければ重視スライダの簡易モデルで計算する"""
    if len(boats) < 6:
        return Plan([], "出走表が取得できない/欠場あり")
    if not odds:
        return Plan([], "オッズ未発表")
    if probs is None:
        wp = win_probs(strengths(boats, s))
        probs = trifecta_probs(wp)
    else:
        wp = {w: 0.0 for w in range(1, 7)}
        for k, p in probs.items():
            wp[int(k[0])] += p
    if s.max_wind and wind is not None and wind >= s.max_wind:
        return Plan([], f"風が強いので見送り（{wind:.0f}m）", win_probs=wp)
    fav = max(wp.values())
    if fav > skip_fav_threshold(s.skip_fav):
        return Plan([], f"堅いレースなので見送り（1着確率{fav:.0%}の艇あり）", win_probs=wp)
    probs = blend_with_market(probs, odds, s.crowd)
    picks = pick_tickets(probs, odds, s, rng)
    if not picks:
        reason = "期待値の条件を満たす買い目なし" if s.selectivity else "候補なし"
        return Plan([], reason, win_probs=wp)
    hit = sum(t.prob for t in picks)
    avg_ev = sum(t.ev for t in picks) / len(picks)
    tickets = allocate(picks, min(budget, s.race_budget), s.allocation)
    if not tickets:
        return Plan([], "予算切れ", hit, avg_ev, wp)
    hit = sum(t.prob for t in tickets)
    return Plan(tickets, "", hit, avg_ev, wp)
