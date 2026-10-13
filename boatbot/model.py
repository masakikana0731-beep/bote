"""予想ロジック：各艇の強さ → 1着確率 → 3連単確率 → 買い目選び → 金額配分。

精度を追求したモデルではなく「設定スライダで味付けを変えて遊ぶ」ためのもの。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import permutations

from .scraper import Boat

# 全国平均のコース別1着率（おおよその値。年や場によって変わる）
COURSE_WIN_RATE = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.10, 5: 0.06, 6: 0.03}


@dataclass
class Settings:
    # --- お金まわり（円） ---
    daily_budget: int = 3000
    race_budget: int = 500
    max_races_per_day: int = 10
    # --- スライダ（0〜100） ---
    gamble: int = 30          # ギャンブル性：0=本命党 / 100=大穴狙い
    selectivity: int = 20     # 厳選度：0=見送りなし / 100=自信がある時だけ
    crowd: int = 30           # 人気(オッズ)を信じる度合い：0=独自予想のみ / 100=人気通り
    whim: int = 10            # 気まぐれ度：買い目にランダム性を混ぜる
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

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                cur = getattr(s, k)
                setattr(s, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
        return s.clamp()

    def clamp(self) -> "Settings":
        for k in ("gamble", "selectivity", "crowd", "whim", "w_course", "w_national",
                  "w_local", "w_motor", "w_exhibit", "w_start"):
            setattr(self, k, max(0, min(100, int(getattr(self, k)))))
        self.daily_budget = max(0, self.daily_budget)
        self.race_budget = max(100, self.race_budget)
        self.tickets = max(1, min(20, self.tickets))
        self.lead_minutes = max(2, min(30, self.lead_minutes))
        if self.allocation not in ("equal", "dutch", "prob"):
            self.allocation = "dutch"
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


def pick_tickets(probs: dict[str, float], odds: dict[str, float], s: Settings,
                 rng: random.Random | None = None) -> list[Ticket]:
    """ギャンブル性 g に応じて score = p × odds^(2g) で並べて上位を選ぶ。
    g=0 なら当たりやすさ順、g=0.5 で期待値順、g=1 で高配当寄り。
    """
    rng = rng or random.Random()
    g = s.gamble / 100
    noise = s.whim / 100
    scored = []
    for k, p in probs.items():
        o = odds.get(k)
        if not o:
            continue
        score = math.log(p) + 2 * g * math.log(o) + rng.gauss(0, 1.5 * noise)
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
              rng: random.Random | None = None) -> Plan:
    if len(boats) < 6:
        return Plan([], "出走表が取得できない/欠場あり")
    if not odds:
        return Plan([], "オッズ未発表")
    wp = win_probs(strengths(boats, s))
    probs = blend_with_market(trifecta_probs(wp), odds, s.crowd)
    picks = pick_tickets(probs, odds, s, rng)
    if not picks:
        return Plan([], "候補なし", win_probs=wp)
    hit = sum(t.prob for t in picks)
    avg_ev = sum(t.ev for t in picks) / len(picks)
    # 厳選度：平均期待値(独自確率×オッズ)がしきい値未満なら見送り
    threshold = 0.4 + 0.8 * s.selectivity / 100  # 0.4〜1.2
    if s.selectivity > 0 and avg_ev < threshold:
        return Plan([], f"自信不足（期待値{avg_ev:.2f} < {threshold:.2f}）", hit, avg_ev, wp)
    tickets = allocate(picks, min(budget, s.race_budget), s.allocation)
    if not tickets:
        return Plan([], "予算切れ", hit, avg_ev, wp)
    hit = sum(t.prob for t in tickets)
    return Plan(tickets, "", hit, avg_ev, wp)
