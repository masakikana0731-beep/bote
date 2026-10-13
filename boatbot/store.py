"""SQLite に設定・スケジュール・推奨（買い目）・結果を保存する。"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .model import Settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK (id = 1), data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS schedule (
    hd TEXT, jcd TEXT, rno INTEGER, deadline TEXT, processed INTEGER DEFAULT 0,
    PRIMARY KEY (hd, jcd, rno));
CREATE TABLE IF NOT EXISTS recs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hd TEXT, jcd TEXT, rno INTEGER, deadline TEXT, created_at TEXT,
    status TEXT,              -- pending=未対応 / bought=購入済 / paper=仮想購入 / failed=自動購入失敗
                              -- unknown=購入されたか不明 / skipped=自分で見送り / pass=ツールが見送り
    tickets TEXT, total INTEGER, info TEXT,
    result_combo TEXT, payout INTEGER, returned INTEGER,
    UNIQUE (hd, jcd, rno));
CREATE TABLE IF NOT EXISTS buy_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, rec_id INTEGER, at TEXT, mode TEXT,
    ok INTEGER, uncertain INTEGER, total INTEGER, message TEXT);
"""

# 予算を消費しているとみなすステータス（未対応・不明も「買ったかもしれない」ので安全側で数える）
SPENDING = ("pending", "bought", "paper", "unknown")
# 実際にお金が動いた（かもしれない）ステータス
REAL = ("bought", "unknown")


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)

    def _q(self, sql: str, args=()):
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur.fetchall()

    # ---- settings
    def settings(self) -> Settings:
        rows = self._q("SELECT data FROM settings WHERE id=1")
        return Settings.from_dict(json.loads(rows[0]["data"])) if rows else Settings()

    def save_settings(self, s: Settings) -> None:
        self._q("INSERT OR REPLACE INTO settings (id, data) VALUES (1, ?)", (json.dumps(s.__dict__),))

    # ---- schedule
    def add_schedule(self, hd: str, jcd: str, deadlines: dict[int, str]) -> None:
        for rno, t in deadlines.items():
            self._q("INSERT OR IGNORE INTO schedule (hd, jcd, rno, deadline) VALUES (?,?,?,?)", (hd, jcd, rno, t))

    def schedule(self, hd: str) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM schedule WHERE hd=? ORDER BY deadline, jcd", (hd,))

    def has_schedule(self, hd: str) -> bool:
        return bool(self._q("SELECT 1 FROM schedule WHERE hd=? LIMIT 1", (hd,)))

    def mark_processed(self, hd: str, jcd: str, rno: int) -> None:
        self._q("UPDATE schedule SET processed=1 WHERE hd=? AND jcd=? AND rno=?", (hd, jcd, rno))

    # ---- recs
    def add_rec(self, hd, jcd, rno, deadline, created_at, status, tickets, total, info) -> int:
        self._q(
            "INSERT OR REPLACE INTO recs (hd, jcd, rno, deadline, created_at, status, tickets, total, info)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (hd, jcd, rno, deadline, created_at, status, json.dumps(tickets), total, json.dumps(info, ensure_ascii=False)))
        return self._q("SELECT id FROM recs WHERE hd=? AND jcd=? AND rno=?", (hd, jcd, rno))[0]["id"]

    def recs(self, hd: str | None = None) -> list[dict]:
        rows = (self._q("SELECT * FROM recs WHERE hd=? ORDER BY deadline DESC", (hd,)) if hd
                else self._q("SELECT * FROM recs ORDER BY hd DESC, deadline DESC"))
        return [self._rec(r) for r in rows]

    def rec(self, rid: int) -> dict | None:
        rows = self._q("SELECT * FROM recs WHERE id=?", (rid,))
        return self._rec(rows[0]) if rows else None

    @staticmethod
    def _rec(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["tickets"] = json.loads(d["tickets"] or "[]")
        d["info"] = json.loads(d["info"] or "{}")
        return d

    def set_status(self, rid: int, status: str) -> None:
        self._q("UPDATE recs SET status=? WHERE id=?", (status, rid))

    def set_result(self, rid: int, combo: str, payout: int, returned: int) -> None:
        self._q("UPDATE recs SET result_combo=?, payout=?, returned=? WHERE id=?", (combo, payout, returned, rid))

    def awaiting_result(self) -> list[dict]:
        rows = self._q("SELECT * FROM recs WHERE status != 'pass' AND result_combo IS NULL")
        return [self._rec(r) for r in rows]

    def spent(self, hd: str, exclude_id: int | None = None) -> tuple[int, int]:
        """(使用額, 参加レース数)"""
        r = self._q(f"SELECT COALESCE(SUM(total),0) s, COUNT(*) c FROM recs WHERE hd=? AND status IN {SPENDING}"
                    " AND id != ?", (hd, exclude_id or -1))[0]
        return r["s"], r["c"]

    def real_spent(self, hd: str) -> int:
        """実際にお金が動いた（かもしれない）額"""
        return self._q(f"SELECT COALESCE(SUM(total),0) s FROM recs WHERE hd=? AND status IN {REAL}", (hd,))[0]["s"]

    def log_buy(self, rec_id: int, at: str, mode: str, ok: bool, uncertain: bool, total: int, message: str) -> None:
        self._q("INSERT INTO buy_log (rec_id, at, mode, ok, uncertain, total, message) VALUES (?,?,?,?,?,?,?)",
                (rec_id, at, mode, int(ok), int(uncertain), total, message))

    def buy_log(self, limit: int = 30) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM buy_log ORDER BY id DESC LIMIT ?", (limit,))]

    def stats(self) -> list[dict]:
        """日別の収支。real=実際に購入（手動・本番自動）/ virtual=推奨を全部買っていた場合"""
        rows = self._q(f"""
            SELECT hd,
              SUM(CASE WHEN status IN {REAL} THEN total ELSE 0 END) real_bet,
              SUM(CASE WHEN status IN {REAL} THEN COALESCE(returned,0) ELSE 0 END) real_ret,
              SUM(CASE WHEN status!='pass' THEN total ELSE 0 END) virt_bet,
              SUM(CASE WHEN status!='pass' THEN COALESCE(returned,0) ELSE 0 END) virt_ret,
              SUM(CASE WHEN status!='pass' AND returned>0 THEN 1 ELSE 0 END) hits,
              SUM(CASE WHEN status!='pass' AND result_combo IS NOT NULL THEN 1 ELSE 0 END) settled
            FROM recs GROUP BY hd ORDER BY hd DESC""")
        return [dict(r) for r in rows]
