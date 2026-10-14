"""公式の一括データ（番組表 B ファイル / 競走成績 K ファイル）の取得とパース。

  python -m boatbot.history download --start 20230701 --end 20261012
  python -m boatbot.history build                 # data/history.parquet を作る

ファイルは https://www1.mbrace.or.jp/od2/{B,K}/YYYYMM/{b,k}YYMMDD.lzh（1日1ファイル・全場分）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

RAW = Path("data/raw")
GRADES = ("A1", "A2", "B1", "B2")
BET_LABELS = {"単勝": "win", "複勝": "place", "２連単": "exacta", "２連複": "quinella",
              "拡連複": "wide", "３連単": "tri", "３連複": "trio"}
URL = "https://www1.mbrace.or.jp/od2/{kind}/{ym}/{k}{ymd}.lzh"
_ZEN = str.maketrans("０１２３４５６７８９ＲＡＢＨｍ：", "0123456789RABHm:")


# ---------------------------------------------------------------- download

def _raw_path(kind: str, d: date) -> Path:
    return RAW / kind / f"{kind.lower()}{d:%y%m%d}.lzh"


def download(start: date, end: date, workers: int = 3, interval: float = 1.0) -> None:
    days = [start + timedelta(i) for i in range((end - start).days + 1)]
    jobs = [(k, d) for d in days for k in ("B", "K") if not _raw_path(k, d).exists()]
    print(f"{len(jobs)} ファイルを取得します", flush=True)
    lock, last = threading.Lock(), [0.0]
    sess = requests.Session()
    sess.headers["User-Agent"] = "Mozilla/5.0 (personal boatbot; low-frequency)"
    done = [0]

    def one(job):
        kind, d = job
        with lock:  # アクセス開始間隔を空ける
            wait = interval - (time.monotonic() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
        url = URL.format(kind=kind, ym=f"{d:%Y%m}", k=kind.lower(), ymd=f"{d:%y%m%d}")
        for attempt in range(3):
            try:
                r = sess.get(url, timeout=60)
                if r.status_code == 404:
                    content = b""  # 開催なし等
                else:
                    r.raise_for_status()
                    content = r.content
                p = _raw_path(kind, d)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(content)
                break
            except requests.RequestException as e:
                if attempt == 2:
                    print(f"  失敗 {url}: {e}", flush=True)
                time.sleep(5)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(jobs):
                print(f"  {done[0]}/{len(jobs)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


def update(today: date) -> bool:
    """前日分までの不足ファイルを取得し、増えていれば history.parquet を作り直す（毎朝1回）"""
    have = sorted((RAW / "K").glob("k*.lzh"))
    if not have:
        return False  # 初回の一括取得は手動で（README参照）
    last = have[-1].stem[1:]
    start = date(2000 + int(last[:2]), int(last[2:4]), int(last[4:])) + timedelta(1)
    end = today - timedelta(1)
    if start > end:
        return False
    download(start, end, workers=1)
    build()
    return True


def read_lzh(path: Path) -> str:
    import lhafile
    if path.stat().st_size == 0:
        return ""
    a = lhafile.Lhafile(str(path))
    return "".join(a.read(n).decode("cp932", errors="replace") for n in a.namelist())


# ---------------------------------------------------------------- parse B（番組表）

def _f(s: str) -> float | None:
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def parse_b(text: str, d: date) -> list[dict]:
    """出走表の各艇 -> dict"""
    rows = []
    jcd = None
    rno = None
    race_meta: dict = {}
    for line in text.splitlines():
        m = re.match(r"^(\d{2})BBGN", line)
        if m:
            jcd = m.group(1)
            continue
        if "BEND" in line[:6]:
            jcd = None
            continue
        t = line.translate(_ZEN)
        m = re.match(r"^\s*(\d{1,2})R\s+(\S+)\s+.*?H(\d+)m.*?(\d{2}:\d{2})", t)
        if m and jcd:
            rno = int(m.group(1))
            race_meta = {"race_name": m.group(2), "distance": int(m.group(3)), "deadline": m.group(4)}
            continue
        # 固定幅（全角も1文字）: 艇 登番 名前4 年齢2 支部2 体重2 級別2 全国勝率 2率 当地勝率 2率 ﾓｰﾀｰNo 2率 ﾎﾞｰﾄNo 2率 今節成績
        if jcd and rno and re.match(r"^[1-6] \d{4}", line) and len(line) >= 58 and line[16:18] in GRADES:
            def num(a, b):
                return _f(line[a:b])
            rows.append({
                "hd": f"{d:%Y%m%d}", "jcd": jcd, "rno": rno, **race_meta,
                "waku": int(line[0]), "toban": line[2:6], "age": int(num(10, 12) or 0),
                "branch": line[12:14].strip(), "weight": num(14, 16), "grade": line[16:18],
                "nat_win": num(18, 23), "nat_2r": num(23, 29), "loc_win": num(29, 34), "loc_2r": num(34, 40),
                "motor_no": num(40, 43), "motor_2r": num(43, 49), "boat_no": num(49, 52), "boat_2r": num(52, 58),
                "meet_results": line[58:72].strip(),
            })
    return rows


# ---------------------------------------------------------------- parse K（競走成績）

def parse_k(text: str, d: date) -> tuple[list[dict], list[dict]]:
    """(各艇の成績, レースごとの情報と3連単払戻)"""
    boats, races = [], []
    jcd = None
    rno = None
    cur: dict | None = None
    bet_type = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\d{2})KBGN", line)
        if m:
            jcd = m.group(1)
        elif "KEND" in line[:6]:
            jcd = None
        t = line.translate(_ZEN)
        # レース名に全角スペースが入ることがある（例：「予　選」）ので、名前は H○○m までの任意の文字列とする
        m = re.match(r"^\s+(\d{1,2})R\s+.*?\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm", t)
        if m and jcd:
            rno = int(m.group(1))
            cur = {"hd": f"{d:%Y%m%d}", "jcd": jcd, "rno": rno, "weather": m.group(3),
                   "wind_dir": m.group(4), "wind": int(m.group(5)), "wave": int(m.group(6)),
                   "tri_combo": None, "tri_payout": None, "tri_pop": None}
            races.append(cur)
        m = re.match(r"^\s+(\S{1,2})\s+([1-6])\s+(\d{4})\s.{8,12}?\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d)\s+([F\-L]?[\d.]+|\S+)", line)
        if m and jcd and cur and re.match(r"^\s+(0[1-6]|[FLKSE]\S?)\s", line):
            pos, waku, toban, _mot, _bt, ex, course, st = m.groups()
            boats.append({"hd": cur["hd"], "jcd": jcd, "rno": rno, "waku": int(waku), "toban": toban,
                          "finish": int(pos) if pos.isdigit() else None, "finish_raw": pos,
                          "ex_time": _f(ex), "course": int(course), "st_raw": st})
        m = re.match(r"^\s+３連単\s+(\d-\d-\d)\s+(\d+)\s+人気\s+(\d+)", line)
        if m and cur:
            cur["tri_combo"], cur["tri_payout"], cur["tri_pop"] = m.group(1), int(m.group(2)), int(m.group(3))
        # 全券種の払戻（100円あたり）。拡連複・複勝は複数、同着なら他の券種も複数になる
        m = re.match(r"^\s+(単勝|複勝|２連単|２連複|拡連複|３連単|３連複)\s", line)
        if m and cur:
            bet_type = BET_LABELS[m.group(1)]
            cur.setdefault("payouts", {}).setdefault(bet_type, {})
            body = line[m.end():]
        elif cur and bet_type and line.startswith(" " * 17) and re.match(r"^\s+\d(-\d){0,2}\s+\d+", line):
            body = line  # 拡連複などの2行目以降
        else:
            body = None
            if not line.strip():
                bet_type = None
        if body is not None and bet_type:
            for combo, pay in re.findall(r"(?<![\d-])(\d(?:-\d){0,2})\s+(\d+)(?!\s*人気\s*$)?", body.split("人気")[0]):
                cur["payouts"][bet_type][combo] = int(pay)
        i += 1
    for r in races:
        r["payouts"] = json.dumps(r.get("payouts", {}))
    return boats, races


def build(out: Path = Path("data/history.parquet")) -> None:
    import pandas as pd
    bs, ks, rs = [], [], []
    files = sorted((RAW / "B").glob("*.lzh"))
    for n, bp in enumerate(files):
        ymd = bp.stem[1:]
        d = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:]))
        kp = RAW / "K" / f"k{ymd}.lzh"
        if not kp.exists():
            continue
        try:
            bs += parse_b(read_lzh(bp), d)
            kb, kr = parse_k(read_lzh(kp), d)
            ks += kb
            rs += kr
        except Exception as e:
            print(f"  パース失敗 {ymd}: {e}", flush=True)
        if n % 100 == 0:
            print(f"  {n}/{len(files)}", flush=True)
    b = pd.DataFrame(bs)
    k = pd.DataFrame(ks)
    r = pd.DataFrame(rs)
    df = b.merge(k, on=["hd", "jcd", "rno", "waku", "toban"], how="inner").merge(r, on=["hd", "jcd", "rno"], how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"{out}: {len(df)} 艇 / {df.groupby(['hd', 'jcd', 'rno']).ngroups} レース")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--start", required=True)
    d.add_argument("--end", required=True)
    d.add_argument("--workers", type=int, default=3)
    sub.add_parser("build")
    a = ap.parse_args()
    if a.cmd == "download":
        p = lambda s: date(int(s[:4]), int(s[4:6]), int(s[6:]))
        download(p(a.start), p(a.end), a.workers)
    else:
        build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
