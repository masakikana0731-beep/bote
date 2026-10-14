"""boatrace.jp（公式サイト）から出走表・直前情報・オッズ・結果を取得してパースする。

公式サイトに負荷をかけないよう、リクエスト間に待ち時間を入れている。
HTML構造が変わるとパースに失敗するので、その場合はこのファイルを直す。
"""
from __future__ import annotations

import re
import time
import threading
import warnings
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

BASE = "https://www.boatrace.jp/owpc/pc/race"
UA = "Mozilla/5.0 (personal boatbot; low-frequency)"
MIN_INTERVAL = 1.5  # 秒。連続アクセスの最小間隔

VENUES = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

_ZEN = str.maketrans("０１２３４５６７８９", "0123456789")


@dataclass
class Boat:
    waku: int
    name: str = ""
    toban: str = ""
    grade: str = ""
    f_count: int = 0
    avg_st: float | None = None
    nat_win: float | None = None
    loc_win: float | None = None
    motor_2r: float | None = None
    boat_2r: float | None = None
    ex_time: float | None = None   # 展示タイム
    tilt: float | None = None
    course: int | None = None      # スタート展示の進入コース
    ex_st: float | None = None     # スタート展示のST（Fはマイナス）
    nat_2r: float | None = None
    loc_2r: float | None = None
    age: int | None = None
    weight: float | None = None
    meet_results: str = ""         # 今節成績（着順を空白区切り）
    motor_no: int | None = None
    boat_no: int | None = None
    before: dict = field(default_factory=dict)  # 直前情報の全項目（parse_beforeinfo_full の boats[枠]）


@dataclass
class RaceInfo:
    jcd: str
    rno: int
    hd: str
    boats: list[Boat] = field(default_factory=list)
    deadlines: dict[int, str] = field(default_factory=dict)
    race_name: str = ""
    wind: float | None = None
    wave: float | None = None
    weather: dict = field(default_factory=dict)  # 直前情報の気象（parse_beforeinfo_full の weather）


# ---------------------------------------------------------------- HTTP

class Fetcher:
    def __init__(self, session: requests.Session | None = None):
        self.s = session or requests.Session()
        self.s.headers["User-Agent"] = UA
        if session is None:
            # 長く使わなかった接続をサーバーが切っていることがあるので、接続エラーは少し待って2回までやり直す
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            retry = Retry(total=2, connect=2, read=1, status=0, backoff_factor=2, allowed_methods=["GET"])
            self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self._last = 0.0
        self._lock = threading.Lock()

    def get(self, page: str, **params) -> str:
        with self._lock:
            wait = MIN_INTERVAL - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.s.get(f"{BASE}/{page}", params=params, timeout=20)
            finally:
                self._last = time.monotonic()
        r.raise_for_status()
        r.encoding = "utf-8"
        return r.text


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _num(text: str) -> float | None:
    text = text.strip().translate(_ZEN)
    m = re.search(r"-?\d*\.?\d+", text)
    return float(m.group()) if m else None


# ---------------------------------------------------------------- parsers

def parse_index(html: str) -> list[str]:
    """本日の開催場コード一覧"""
    s = _soup(html)
    codes = []
    for a in s.select('div.table1 a[href*="raceindex?jcd="]'):
        m = re.search(r"jcd=(\d{2})", a["href"])
        if m and m.group(1) not in codes:
            codes.append(m.group(1))
    return codes


def parse_deadlines(html: str) -> dict[int, str]:
    """出走表等の上部にある「締切予定時刻」行 -> {rno: 'HH:MM'}"""
    s = _soup(html)
    for tr in s.select("div.table1 tr"):
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if cells and cells[0] == "締切予定時刻":
            return {i + 1: t for i, t in enumerate(cells[1:]) if re.fullmatch(r"\d{1,2}:\d{2}", t)}
    return {}


def parse_racelist(html: str) -> list[Boat]:
    s = _soup(html)
    boats = []
    for tb in s.select("div.table1.is-tableFixed__3rdadd tbody"):
        tds = tb.find("tr").find_all("td", recursive=False)
        if len(tds) < 8:
            continue
        waku = int(_num(tds[0].get_text()))
        b = Boat(waku=waku)
        info = tds[2]
        divs = info.find_all("div")
        if divs:
            head = divs[0].get_text(" ", strip=True)
            m = re.search(r"(\d{4})", head)
            b.toban = m.group(1) if m else ""
            g = divs[0].find("span")
            b.grade = g.get_text(strip=True) if g else ""
        if len(divs) > 1:
            b.name = re.sub(r"\s+", "", divs[1].get_text())
        if len(divs) > 2:
            m = re.search(r"(\d+)歳/([\d.]+)kg", divs[2].get_text(" ", strip=True))
            if m:
                b.age, b.weight = int(m.group(1)), float(m.group(2))

        def lines(td):
            return [t for t in td.get_text("\n", strip=True).split("\n") if t]

        fls = lines(tds[3])  # F0 / L0 / 0.18
        if fls:
            b.f_count = int(_num(fls[0]) or 0)
        if len(fls) > 2:
            b.avg_st = _num(fls[2])
        nat = lines(tds[4])
        loc = lines(tds[5])
        mot = lines(tds[6])
        bt = lines(tds[7])
        b.nat_win = _num(nat[0]) if nat else None
        b.loc_win = _num(loc[0]) if loc else None
        b.nat_2r = _num(nat[1]) if len(nat) > 1 else None
        b.loc_2r = _num(loc[1]) if len(loc) > 1 else None
        trs = tb.find_all("tr")
        if len(trs) >= 4:  # 4行目が今節の着順
            b.meet_results = " ".join(
                t for t in (td.get_text(strip=True).translate(_ZEN) for td in trs[3].find_all("td")) if t.isdigit())
        b.motor_2r = _num(mot[1]) if len(mot) > 1 else None
        b.motor_no = int(_num(mot[0])) if mot and _num(mot[0]) is not None else None
        b.boat_no = int(_num(bt[0])) if bt and _num(bt[0]) is not None else None
        b.boat_2r = _num(bt[1]) if len(bt) > 1 else None
        boats.append(b)
    return sorted(boats, key=lambda x: x.waku)


def parse_beforeinfo(html: str) -> dict[int, dict]:
    """展示タイム・チルト・スタート展示（進入コース/ST）"""
    s = _soup(html)
    out: dict[int, dict] = {}
    for tb in s.select("div.table1 table.is-w748 tbody"):
        tds = tb.find("tr").find_all("td")
        if len(tds) < 6:
            continue
        waku = _num(tds[0].get_text())
        if waku is None:
            continue
        out[int(waku)] = {"ex_time": _num(tds[4].get_text()), "tilt": _num(tds[5].get_text())}
    for course, div in enumerate(s.select("div.table1_boatImage1"), start=1):
        n = div.select_one(".table1_boatImage1Number")
        t = div.select_one(".table1_boatImage1Time")
        if not n:
            continue
        waku = int(_num(n.get_text()))
        st_text = t.get_text(strip=True) if t else ""
        st = _num(st_text.replace("F", "").replace("L", ""))
        if st is not None and st_text.startswith("F"):
            st = -st
        out.setdefault(waku, {}).update({"course": course, "ex_st": st})
    return out


def parse_race_name(html: str) -> str:
    h = _soup(html).select_one("h3.title16_titleDetail__add2020")
    return h.get_text(" ", strip=True).split()[0] if h and h.get_text(strip=True) else ""


def parse_weather(html: str) -> dict:
    """直前情報の水面気象 -> {'wind': 風速m, 'wave': 波高cm}"""
    s = _soup(html)
    out = {}
    for key, cls in (("wind", "is-wind"), ("wave", "is-wave")):
        el = s.select_one(f"div.weather1_bodyUnit.{cls} .weather1_bodyUnitLabelData")
        if el:
            out[key] = _num(el.get_text())
    return out


def parse_beforeinfo_full(html: str) -> dict:
    """直前情報をすべて取り出す（過去データの収集と本番で共通に使う）。

    {"boats": {枠: {ex_weight, adj_weight, ex_time, tilt, propeller, parts, ex_course, ex_st}},
     "weather": {temp, weather, wind, wind_dir, water_temp, wave, course_dir}}
    wind_dir / course_dir は公式サイトの画像の番号（1〜16 の方位。風がない・不明は None）
    """
    s = _soup(html)
    boats: dict[int, dict] = {}
    for tb in s.select("div.table1 table.is-w748 tbody"):
        rows = tb.find_all("tr")
        tds = rows[0].find_all("td") if rows else []
        if len(tds) < 8:
            continue
        waku = _num(tds[0].get_text())
        if waku is None:
            continue
        parts = tds[7].select("li") or [tds[7]]
        adj = rows[2].find_all("td") if len(rows) > 2 else []
        boats[int(waku)] = {
            "ex_weight": _num(tds[3].get_text()),
            "adj_weight": _num(adj[0].get_text()) if adj else None,
            "ex_time": _num(tds[4].get_text()),
            "tilt": _num(tds[5].get_text()),
            "propeller": tds[6].get_text(strip=True),
            "parts": "/".join(t for t in (p.get_text(" ", strip=True) for p in parts) if t),
        }
    for course, div in enumerate(s.select("div.table1_boatImage1"), start=1):
        n = div.select_one(".table1_boatImage1Number")
        t = div.select_one(".table1_boatImage1Time")
        if not n or _num(n.get_text()) is None:
            continue
        st_text = t.get_text(strip=True) if t else ""
        st = _num(st_text.replace("F", "").replace("L", ""))
        if st is not None and st_text.startswith("F"):
            st = -st
        boats.setdefault(int(_num(n.get_text())), {}).update({"ex_course": course, "ex_st": st, "ex_st_raw": st_text})

    def unit(cls: str):
        return s.select_one(f"div.weather1_bodyUnit.{cls}")

    def label(cls: str):
        u = unit(cls)
        el = u.select_one(".weather1_bodyUnitLabelData") if u else None
        return el.get_text(strip=True) if el else None

    def image_no(cls: str, prefix: str):
        u = unit(cls)
        img = u.select_one(".weather1_bodyUnitImage") if u else None
        for c in (img.get("class") or []) if img else []:
            m = re.fullmatch(prefix + r"(\d+)", c)
            if m:
                return int(m.group(1))
        return None
    w = unit("is-weather")
    weather = {
        "temp": _num(label("is-direction") or ""),
        "weather": w.get_text(strip=True) if w else None,
        "wind": _num(label("is-wind") or ""),
        "wind_dir": image_no("is-windDirection", "is-wind"),
        "water_temp": _num(label("is-waterTemperature") or ""),
        "wave": _num(label("is-wave") or ""),
        "course_dir": image_no("is-direction", "is-direction"),
    }
    return {"boats": boats, "weather": weather}


def parse_odds3t(html: str) -> dict[str, float]:
    """3連単オッズ -> {'1-2-3': 12.3, ...}。未発売・欠場などは含まれない。

    表は 20行 × 6列（列=1着艇）で、行 r は 2着=残り艇の r//4 番目、3着=さらに残りの r%4 番目。
    """
    s = _soup(html)
    pts = [td.get_text(strip=True) for td in s.select("td.oddsPoint")]
    if len(pts) != 120:
        return {}
    odds = {}
    for r in range(20):
        for c in range(6):
            first = c + 1
            rest = [x for x in range(1, 7) if x != first]
            second = rest[r // 4]
            third = [x for x in rest if x != second][r % 4]
            v = _num(pts[r * 6 + c])
            if v and v > 0:
                odds[f"{first}-{second}-{third}"] = v
    return odds


def _odds_value(text: str) -> float | None:
    """'12.3' や複勝・拡連複の '1.5-2.3'（下限-上限）→ 数値。幅があるものは下限（控えめ）を使う"""
    m = re.match(r"^\s*([\d.]+)", text)
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


def parse_odds_tables(html: str) -> list[dict[str, float]]:
    """3連複・2連単/2連複・拡連複・単勝/複勝ページの表 → 表ごとの {組番: オッズ}

    列 = 1着（または1番目の艇）。行の中で n 番目のオッズは n 列目（艇 n）。
    オッズの直前の数字が残りの艇番。3連複は2番目の艇が rowspan で省略される行がある。
    """
    s = _soup(html)
    tables = []
    for tb in s.select("div.table1 table"):
        if not tb.select("td.oddsPoint"):
            continue
        odds: dict[str, float] = {}
        second_mem: dict[int, str] = {}
        for tr in tb.select("tbody tr"):
            labels: list[str] = []
            col = 0
            for td in tr.find_all("td"):
                cls = td.get("class") or []
                text = td.get_text(strip=True).translate(_ZEN)
                if "oddsPoint" in cls:
                    col += 1
                    v = _odds_value(text)
                    if len(labels) >= 2:  # 3連複: 2番目と3番目
                        second_mem[col] = labels[-2]
                        rest = [labels[-2], labels[-1]]
                    elif len(labels) == 1 and col in second_mem:
                        rest = [second_mem[col], labels[-1]]
                    elif len(labels) == 1:
                        rest = [labels[-1]]
                    else:
                        rest = []
                    key = "-".join([str(col)] + rest)
                    if v:
                        odds[key] = v
                    labels = []
                elif re.fullmatch(r"[1-6]", text):
                    labels.append(text)
            if not any("oddsPoint" in (td.get("class") or []) for td in tr.find_all("td")):
                continue
        tables.append(odds)
    return tables


def parse_win_place(html: str) -> list[dict[str, float]]:
    """単勝・複勝ページ → [単勝{艇:オッズ}, 複勝{艇:オッズ下限}]"""
    s = _soup(html)
    out = []
    for tb in s.select("div.table1 table"):
        pts = tb.select("td.oddsPoint")
        if not pts:
            continue
        d = {}
        for tr in tb.select("tbody tr"):
            tds = tr.find_all("td")
            p = tr.select_one("td.oddsPoint")
            if p and tds:
                boat = tds[0].get_text(strip=True).translate(_ZEN)
                v = _odds_value(p.get_text(strip=True))
                if re.fullmatch(r"[1-6]", boat) and v:
                    d[boat] = v
        out.append(d)
    return out


def parse_result(html: str) -> dict | None:
    """3連単の結果 -> {'combo': '5-1-4', 'payout': 5510}（100円あたり）。未確定なら None"""
    s = _soup(html)
    for tb in s.select("div.table1 table tbody"):
        first = tb.find("td")
        if not first or first.get_text(strip=True) != "3連単":
            continue
        nums = [x.get_text(strip=True).translate(_ZEN) for x in tb.select(".numberSet1_number")]
        pay = tb.select_one(".is-payout1")
        if len(nums) >= 3 and pay and pay.get_text(strip=True):
            return {"combo": "-".join(nums[:3]), "payout": int(re.sub(r"\D", "", pay.get_text()))}
        return None
    return None


# ---------------------------------------------------------------- high level

class BoatraceClient:
    def __init__(self, fetcher: Fetcher | None = None):
        self.f = fetcher or Fetcher()

    def venues(self, hd: str) -> list[str]:
        return parse_index(self.f.get("index", hd=hd))

    def deadlines(self, jcd: str, hd: str) -> dict[int, str]:
        return parse_deadlines(self.f.get("racelist", rno=1, jcd=jcd, hd=hd))

    def race(self, jcd: str, rno: int, hd: str) -> RaceInfo:
        html = self.f.get("racelist", rno=rno, jcd=jcd, hd=hd)
        info = RaceInfo(jcd=jcd, rno=rno, hd=hd, boats=parse_racelist(html), deadlines=parse_deadlines(html),
                        race_name=parse_race_name(html))
        bhtml = self.f.get("beforeinfo", rno=rno, jcd=jcd, hd=hd)
        before = parse_beforeinfo(bhtml)
        w = parse_weather(bhtml)
        info.wind, info.wave = w.get("wind"), w.get("wave")
        for b in info.boats:
            for k, v in before.get(b.waku, {}).items():
                setattr(b, k, v)
        full = parse_beforeinfo_full(bhtml)
        info.weather = full["weather"]
        for b in info.boats:
            b.before = full["boats"].get(b.waku, {})
        return info

    def odds3t(self, jcd: str, rno: int, hd: str) -> dict[str, float]:
        return parse_odds3t(self.f.get("odds3t", rno=rno, jcd=jcd, hd=hd))

    def result(self, jcd: str, rno: int, hd: str) -> dict | None:
        return parse_result(self.f.get("raceresult", rno=rno, jcd=jcd, hd=hd))
