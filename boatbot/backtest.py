"""過去レースでのシミュレーション（バックテスト）。

  # 1) 過去データを取得してキャッシュ（初回のみ時間がかかる。2回目以降は取得済みを再利用）
  python -m boatbot.backtest fetch --days 30
  python -m boatbot.backtest fetch --days 30 --venues-per-day 1   # 1日1場だけ抜き取り（速い）

  # 2) 今の設定（またはプリセット）で、1日上限などのルール込みでシミュレーション
  python -m boatbot.backtest run
  python -m boatbot.backtest run --compare        # 本命/バランス/大穴 など複数プリセットを比較
  python -m boatbot.backtest run --set gamble=80 --set tickets=3

注意: オッズは「確定オッズ」を使う。実際の運用では締切数分前のオッズで買うので多少ずれる。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .model import Settings, make_plan
from .scraper import Boat, BoatraceClient, Fetcher, VENUES

CACHE = Path("data/cache")

PRESETS = {
    "本命党": {"gamble": 0, "selectivity": 0, "tickets": 3},
    "バランス": {"gamble": 30, "selectivity": 20, "tickets": 5},
    "期待値重視": {"gamble": 50, "selectivity": 40, "tickets": 5},
    "大穴狙い": {"gamble": 90, "selectivity": 20, "tickets": 5},
    "人気に乗る": {"gamble": 20, "selectivity": 0, "crowd": 100, "tickets": 3},
}


# ---------------------------------------------------------------- 取得

def _race_path(hd: str, jcd: str, rno: int) -> Path:
    return CACHE / hd / f"{jcd}_{rno:02d}.json"


def fetch_race(client: BoatraceClient, hd: str, jcd: str, rno: int, deadline: str) -> str:
    p = _race_path(hd, jcd, rno)
    if p.exists():
        return "cached"
    info = client.race(jcd, rno, hd)
    odds = client.odds3t(jcd, rno, hd)
    result = client.result(jcd, rno, hd)
    data = {"hd": hd, "jcd": jcd, "rno": rno, "deadline": deadline,
            "boats": [asdict(b) for b in info.boats], "odds": odds, "result": result}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False))
    return "ok"


def fetch(days: int, venues_per_day: int | None, workers: int, end: date | None, seed: int) -> None:
    end = end or date.today() - timedelta(days=1)
    rng = random.Random(seed)
    clients = [BoatraceClient(Fetcher()) for _ in range(workers)]
    jobs = []
    for i in range(days):
        hd = (end - timedelta(days=i)).strftime("%Y%m%d")
        day_file = CACHE / hd / "_day.json"
        day = json.loads(day_file.read_text()) if day_file.exists() else {"venues": None, "deadlines": {}}
        if day["venues"] is None:
            day["venues"] = clients[0].venues(hd)
        codes = sorted(day["venues"])
        if venues_per_day:
            codes = sorted(rng.sample(codes, min(venues_per_day, len(codes))))
        for jcd in codes:
            if jcd not in day["deadlines"]:
                day["deadlines"][jcd] = clients[0].deadlines(jcd, hd)
        day_file.parent.mkdir(parents=True, exist_ok=True)
        day_file.write_text(json.dumps(day))
        for jcd in codes:
            day[jcd] = day["deadlines"][jcd]
            for rno, dl in day[jcd].items():
                jobs.append((hd, jcd, int(rno), dl))
        print(f"{hd}: {len(codes)}場 {[VENUES.get(c, c) for c in codes]}", flush=True)
    print(f"対象 {len(jobs)} レース", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_race, clients[i % workers], *job): job for i, job in enumerate(jobs)}
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:  # 中止レースなど
                print(f"  失敗 {futs[f]}: {e}", flush=True)
            if done % 20 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)


ODDS = Path("data/odds")


def fetch_odds(days: int, venues_per_day: int, workers: int, end: date | None, seed: int) -> None:
    """3連単確定オッズだけを取得（出走表・結果は公式一括データ history.py から取る）"""
    end = end or date.today() - timedelta(days=1)
    rng = random.Random(seed)
    clients = [BoatraceClient(Fetcher()) for _ in range(workers)]
    jobs = []
    for i in range(days):
        hd = (end - timedelta(days=i)).strftime("%Y%m%d")
        vf = ODDS / hd / "_venues.json"
        if vf.exists():
            codes = json.loads(vf.read_text())
        else:
            allv = sorted(clients[0].venues(hd))
            codes = sorted(rng.sample(allv, min(venues_per_day, len(allv))))
            vf.parent.mkdir(parents=True, exist_ok=True)
            vf.write_text(json.dumps(codes))
        jobs += [(hd, jcd, rno) for jcd in codes for rno in range(1, 13)
                 if not (ODDS / hd / f"{jcd}_{rno:02d}.json").exists()]
    print(f"対象 {len(jobs)} レース", flush=True)

    def one(i_job):
        i, (hd, jcd, rno) = i_job
        odds = clients[i % workers].odds3t(jcd, rno, hd)
        (ODDS / hd / f"{jcd}_{rno:02d}.json").write_text(json.dumps(odds))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, x) for x in enumerate(jobs)]
        for f in as_completed(futs):
            done += 1
            try:
                f.result()
            except Exception as e:
                print(f"  失敗: {e}", flush=True)
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)


ODDS_MULTI = Path("data/odds_multi")


def fetch_odds_multi(workers: int) -> None:
    """data/odds にあるレースについて、3連単以外の券種のオッズも取得"""
    from .scraper import parse_odds_tables, parse_win_place
    clients = [BoatraceClient(Fetcher()) for _ in range(workers)]
    jobs = []
    for f in sorted(ODDS.glob("*/[0-9]*_*.json")):
        hd, (jcd, rno) = f.parent.name, f.stem.split("_")
        out = ODDS_MULTI / hd / f.name
        if not out.exists():
            jobs.append((hd, jcd, int(rno), out))
    print(f"対象 {len(jobs)} レース", flush=True)

    def one(i_job):
        i, (hd, jcd, rno, out) = i_job
        f = clients[i % workers].f
        trio = parse_odds_tables(f.get("odds3f", rno=rno, jcd=jcd, hd=hd))
        two = parse_odds_tables(f.get("odds2tf", rno=rno, jcd=jcd, hd=hd))
        wide = parse_odds_tables(f.get("oddsk", rno=rno, jcd=jcd, hd=hd))
        wp = parse_win_place(f.get("oddstf", rno=rno, jcd=jcd, hd=hd))
        data = {"trio": trio[0] if trio else {}, "exacta": two[0] if two else {},
                "quinella": two[1] if len(two) > 1 else {}, "wide": wide[0] if wide else {},
                "win": wp[0] if wp else {}, "place": wp[1] if len(wp) > 1 else {}}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data))

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, x) for x in enumerate(jobs)]
        for fu in as_completed(futs):
            done += 1
            try:
                fu.result()
            except Exception as e:
                print(f"  失敗: {e}", flush=True)
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)


# ---------------------------------------------------------------- シミュレーション

def load_races() -> dict[str, list[dict]]:
    days: dict[str, list[dict]] = {}
    for p in sorted(CACHE.glob("*/*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        r["boats"] = [Boat(**b) for b in r["boats"]]
        days.setdefault(r["hd"], []).append(r)
    for rs in days.values():
        rs.sort(key=lambda r: (r["deadline"], r["jcd"]))
    return days


def simulate(days: dict[str, list[dict]], s: Settings, seed: int = 0) -> dict:
    rng = random.Random(seed)
    venues = s.venue_set()
    daily, bets = [], []
    for hd in sorted(days):
        spent = races = ret = 0
        for r in days[hd]:
            if venues and r["jcd"] not in venues:
                continue
            if races >= s.max_races_per_day or not r["result"] or not r["odds"]:
                continue
            plan = make_plan(r["boats"], r["odds"], s, s.daily_budget - spent, rng)
            if not plan.tickets:
                continue
            res = r["result"]
            payout = sum(t.amount // 100 * res["payout"] for t in plan.tickets if t.combo == res["combo"])
            spent += plan.total
            races += 1
            ret += payout
            bets.append({"hd": hd, "jcd": r["jcd"], "rno": r["rno"], "bet": plan.total, "ret": payout,
                         "combo": res["combo"], "odds": res["payout"] / 100})
        daily.append({"hd": hd, "bet": spent, "ret": ret, "races": races})
    return summarize(daily, bets)


def summarize(daily: list[dict], bets: list[dict]) -> dict:
    bet = sum(d["bet"] for d in daily)
    ret = sum(d["ret"] for d in daily)
    hits = [b for b in bets if b["ret"] > 0]
    # 累積収支の最大下落幅・最長連敗
    cum = peak = dd = streak = worst = 0
    for b in bets:
        cum += b["ret"] - b["bet"]
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        streak = 0 if b["ret"] else streak + 1
        worst = max(worst, streak)
    return {
        "days": len(daily), "races": len(bets), "bet": bet, "ret": ret, "profit": ret - bet,
        "roi": ret / bet if bet else 0, "hit_rate": len(hits) / len(bets) if bets else 0,
        "plus_days": sum(1 for d in daily if d["ret"] > d["bet"]),
        "max_drawdown": dd, "longest_losing": worst,
        "best": max(hits, key=lambda b: b["ret"], default=None),
        "daily": daily,
    }


def fmt(name: str, r: dict) -> str:
    best = r["best"]
    b = f"{VENUES.get(best['jcd'], best['jcd'])}{best['rno']}R {best['combo']} +{best['ret']:,}円" if best else "-"
    return (f"[{name}] {r['days']}日 {r['races']}R  投資{r['bet']:,}円 → 払戻{r['ret']:,}円 "
            f"収支{r['profit']:+,}円 回収率{r['roi']:.0%}  的中率{r['hit_rate']:.0%}  "
            f"プラスの日 {r['plus_days']}/{r['days']}  最大下落{r['max_drawdown']:,}円  最長連敗{r['longest_losing']}R  最高配当 {b}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--days", type=int, default=30)
    f.add_argument("--venues-per-day", type=int, default=None)
    f.add_argument("--workers", type=int, default=1, help="同時アクセス数（公式サイトに配慮して小さく）")
    f.add_argument("--end", default=None, help="最終日 YYYYMMDD（既定=昨日）")
    f.add_argument("--seed", type=int, default=1)
    fo = sub.add_parser("fetch-odds")
    fo.add_argument("--days", type=int, default=90)
    fo.add_argument("--venues-per-day", type=int, default=3)
    fo.add_argument("--workers", type=int, default=1)
    fo.add_argument("--end", default=None)
    fo.add_argument("--seed", type=int, default=1)
    fm = sub.add_parser("fetch-odds-multi")
    fm.add_argument("--workers", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--compare", action="store_true")
    r.add_argument("--set", action="append", default=[], help="設定の上書き 例: gamble=80")
    r.add_argument("--db", default="data/boatbot.db", help="現在の設定を読むDB")
    r.add_argument("--json", default=None, help="結果をJSONで保存")
    r.add_argument("--seeds", type=int, default=5, help="気まぐれ度の乱数を変えて平均する回数")
    a = ap.parse_args()

    if a.cmd == "fetch-odds-multi":
        fetch_odds_multi(a.workers)
        return 0
    if a.cmd in ("fetch", "fetch-odds"):
        end = date(int(a.end[:4]), int(a.end[4:6]), int(a.end[6:])) if a.end else None
        (fetch if a.cmd == "fetch" else fetch_odds)(a.days, a.venues_per_day, a.workers, end, a.seed)
        return 0

    days = load_races()
    if not days:
        print("キャッシュがありません。先に fetch を実行してください")
        return 1
    base = {}
    if Path(a.db).exists():
        from .store import Store
        base = Store(a.db).settings().__dict__
    for kv in a.set:
        k, v = kv.split("=", 1)
        base[k] = v
    targets = {"今の設定": base}
    if a.compare:
        targets.update({k: {**base, **v} for k, v in PRESETS.items()})
    out = {}
    for name, d in targets.items():
        s = Settings.from_dict(d)
        runs = [simulate(days, s, seed) for seed in range(a.seeds if s.whim else 1)]
        res = min(runs, key=lambda x: abs(x["profit"] - sorted(y["profit"] for y in runs)[len(runs) // 2]))  # 中央値の回
        out[name] = res
        print(fmt(name, res))
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
