"""公式サイト（boatrace.jp）から過去レースのデータを大量に集める。何度実行しても取得済みは飛ばす。

  python -m boatbot.collect before --start 20230701 --end 20261013 --workers 3
      直前情報（展示・チルト・部品交換・スタート展示・気象）。新しい日から順に。HTMLを gzip で保存
  python -m boatbot.collect odds --start 20251014 --end 20261013 --workers 1
      3連単確定オッズ（全場・全レース）→ data/odds/
  python -m boatbot.collect parse-before
      保存した直前情報 → data/before.parquet（1艇1行）
  python -m boatbot.collect watch
      当日の全レースについて、締切の数分前のオッズ（3連単・単勝複勝・拡連複）を記録し続ける
      → data/odds_live/。確定オッズとの差（シミュレーションの偏り）を確かめるため

対象レースは data/history.parquet（公式一括データ）にあるレース。
公式サイトに配慮して、1接続あたり MIN_INTERVAL 秒以上あけ、接続数も小さくする。
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .scraper import (BoatraceClient, Fetcher, parse_beforeinfo_full, parse_odds3t, parse_odds_tables,
                      parse_win_place)

BEFORE = Path("data/before")
ODDS = Path("data/odds")
ODDS_LIVE = Path("data/odds_live")
JST = timezone(timedelta(hours=9))


def races(start: str, end: str) -> list[tuple[str, str, int]]:
    """history.parquet にあるレース (hd, jcd, rno)。新しい順"""
    import pandas as pd
    h = pd.read_parquet("data/history.parquet", columns=["hd", "jcd", "rno"]).drop_duplicates()
    h = h[(h["hd"] >= start) & (h["hd"] <= end)].sort_values(["hd", "jcd", "rno"], ascending=[False, True, True])
    return list(h.itertuples(index=False, name=None))


def _run(jobs, workers: int, one, label: str) -> None:
    """jobs を workers 本の接続で取得。失敗したものは最後にもう一度だけやり直す"""
    print(f"[{label}] 対象 {len(jobs)} 件", flush=True)
    fetchers = [Fetcher() for _ in range(workers)]
    t0 = time.time()
    for attempt in range(2):
        failed, done = [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, fetchers[i % workers], job): job for i, job in enumerate(jobs)}
            for fu in as_completed(futs):
                done += 1
                try:
                    fu.result()
                except Exception as e:
                    failed.append(futs[fu])
                    if len(failed) <= 20 or len(failed) % 100 == 0:
                        print(f"  失敗({len(failed)}) {futs[fu]}: {str(e)[:120]}", flush=True)
                if done % 500 == 0:
                    rate = done / (time.time() - t0)
                    print(f"  {done}/{len(jobs)}  {rate:.2f}件/秒  残り約{(len(jobs) - done) / max(rate, 1e-6) / 3600:.1f}時間",
                          flush=True)
        if not failed:
            break
        print(f"[{label}] 失敗 {len(failed)} 件をやり直します", flush=True)
        time.sleep(60)
        jobs, t0 = failed, time.time()
    print(f"[{label}] 完了（失敗 {len(failed)} 件）", flush=True)


# ---------------------------------------------------------------- 直前情報

def before_path(hd: str, jcd: str, rno: int) -> Path:
    return BEFORE / hd / f"{jcd}_{rno:02d}.html.gz"


def fetch_before(start: str, end: str, workers: int) -> None:
    jobs = [r for r in races(start, end) if not before_path(*r).exists()]

    def one(f: Fetcher, job):
        hd, jcd, rno = job
        html = f.get("beforeinfo", rno=rno, jcd=jcd, hd=hd)
        if "table1" not in html:  # メンテナンス画面など。保存しないで後でやり直す
            raise RuntimeError("直前情報の表がありません")
        p = before_path(hd, jcd, rno)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(html.encode("utf-8")))
        tmp.replace(p)
    _run(jobs, workers, one, "直前情報")


def _parse_file(p: Path) -> list[dict]:
    hd, (jcd, rno) = p.parent.name, p.name.split(".")[0].split("_")
    d = parse_beforeinfo_full(gzip.decompress(p.read_bytes()).decode("utf-8"))
    w = d["weather"]
    rows = []
    for waku, b in d["boats"].items():
        rows.append({"hd": hd, "jcd": jcd, "rno": int(rno), "waku": waku, **b,
                     **{f"w_{k}": v for k, v in w.items()}})
    return rows


def parse_before(out: Path = Path("data/before.parquet"), jobs: int = 8) -> None:
    import pandas as pd
    from multiprocessing import Pool
    files = sorted(BEFORE.glob("*/*.html.gz"))
    print(f"{len(files)} レース分をパースします", flush=True)
    rows = []
    with Pool(jobs) as pool:
        for k, r in enumerate(pool.imap(_parse_file, files, chunksize=200)):
            rows += r
            if k % 20000 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    print(f"{out}: {len(df)} 艇 / {df.groupby(['hd', 'jcd', 'rno']).ngroups} レース")


# ---------------------------------------------------------------- 確定オッズ

def fetch_odds(start: str, end: str, workers: int) -> None:
    jobs = [r for r in races(start, end) if not (ODDS / r[0] / f"{r[1]}_{r[2]:02d}.json").exists()]

    def one(f: Fetcher, job):
        hd, jcd, rno = job
        odds = parse_odds3t(f.get("odds3t", rno=rno, jcd=jcd, hd=hd))
        p = ODDS / hd / f"{jcd}_{rno:02d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(odds))
    _run(jobs, workers, one, "3連単オッズ")


# ---------------------------------------------------------------- 締切前のオッズを記録

LEADS = (5, 1)  # 締切の何分前に記録するか


def _snap_odds(f: Fetcher, hd: str, jcd: str, rno: int, dl: datetime, lead: int) -> None:
    """締切前のオッズ（3連単・単勝複勝・拡連複）を1回記録する。締切を過ぎたら記録しない"""
    p = ODDS_LIVE / hd / f"{jcd}_{rno:02d}_m{lead}.json"
    if datetime.now(JST) >= dl:
        return
    data = {"ts": datetime.now(JST).isoformat(), "deadline": dl.isoformat(), "lead": lead,
            "tri": parse_odds3t(f.get("odds3t", rno=rno, jcd=jcd, hd=hd))}
    wp = parse_win_place(f.get("oddstf", rno=rno, jcd=jcd, hd=hd))
    data["win"], data["place"] = (wp[0] if wp else {}), (wp[1] if len(wp) > 1 else {})
    wide = parse_odds_tables(f.get("oddsk", rno=rno, jcd=jcd, hd=hd))
    data["wide"] = wide[0] if wide else {}
    data["ts_end"] = datetime.now(JST).isoformat()
    data["late"] = datetime.now(JST) >= dl  # 取り終わる前に締切を過ぎた（確定オッズに近い）
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def watch(interval: int = 10, workers: int = 4) -> None:
    """当日の全レースについて、締切 LEADS 分前のオッズを記録する（止めるまで動き続ける）。

    公式サイトの応答が遅い（1ページ数秒〜10秒）ので、締切が重なっても間に合うよう複数の接続で並行して取る
    """
    client = BoatraceClient(Fetcher())
    pool = ThreadPoolExecutor(max_workers=workers)
    fetchers = [Fetcher() for _ in range(workers)]
    submitted: dict = {}
    day, schedule, loaded = None, {}, None
    while True:
        now = datetime.now(JST)
        hd = now.strftime("%Y%m%d")
        # 日付が変わったとき・30分ごと（締切の変更に追従）・まだ空のときに日程を読み直す
        if hd != day or not schedule or now - loaded > timedelta(minutes=30):
            if hd != day:
                submitted = {}
            day, loaded = hd, now
            try:
                new = {}
                for jcd in client.venues(hd):
                    for rno, hhmm in client.deadlines(jcd, hd).items():
                        h, m = map(int, hhmm.split(":"))
                        new[(jcd, rno)] = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if len(new) != len(schedule):
                    print(f"{hd}: {len(new)} レース", flush=True)
                schedule = new
            except Exception as e:
                print(f"{hd}: 日程の取得に失敗 {e}", flush=True)
            if not schedule:
                time.sleep(600)
                continue
        for (jcd, rno), dl in sorted(schedule.items(), key=lambda x: x[1]):
            for lead in LEADS:
                key = (hd, jcd, rno, lead)
                p = ODDS_LIVE / hd / f"{jcd}_{rno:02d}_m{lead}.json"
                if key in submitted or p.exists() or not (dl - timedelta(minutes=lead) <= now < dl):
                    continue
                fu = pool.submit(_snap_odds, fetchers[len(submitted) % workers], hd, jcd, rno, dl, lead)
                fu.add_done_callback(lambda x, k=key: x.exception() and print(f"  失敗 {k}: {x.exception()}", flush=True))
                submitted[key] = fu
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    for name in ("before", "odds"):
        p = sub.add_parser(name)
        p.add_argument("--start", default="20180101")
        p.add_argument("--end", default=yesterday)
        p.add_argument("--workers", type=int, default=2)
    pb = sub.add_parser("parse-before")
    pb.add_argument("--jobs", type=int, default=8)
    sub.add_parser("watch")
    a = ap.parse_args()
    if a.cmd == "before":
        fetch_before(a.start, a.end, a.workers)
    elif a.cmd == "odds":
        fetch_odds(a.start, a.end, a.workers)
    elif a.cmd == "parse-before":
        parse_before(jobs=a.jobs)
    else:
        watch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
