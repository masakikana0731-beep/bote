"""テレボート自動購入の準備用コマンド。

  python -m boatbot.teleboat_cli record   # 画面操作を録画（Playwright codegen）
  python -m boatbot.teleboat_cli test     # 投票ボタンの直前まで実行（お金は使わない）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .executor import AbortPurchase, TeleboatExecutor
from .engine import now_jst

URL = os.environ.get("TELEBOAT_URL", "https://spweb.brtb.jp/")
STATE = "data/teleboat_state.json"


def record() -> int:
    os.makedirs("data", exist_ok=True)
    cmd = [sys.executable, "-m", "playwright", "codegen", "--target", "python",
           "--device", "iPhone 13", "--save-storage", STATE, URL]
    print("ブラウザが開きます。ログイン→レース選択→買い目入力→確認画面まで操作してください。")
    print("右側のコードを teleboat_flow.py の TODO に貼り替えます。")
    return subprocess.call(cmd)


def test(jcd: str, rno: int, combo: str) -> int:
    order = {"hd": now_jst().strftime("%Y%m%d"), "jcd": jcd, "rno": rno, "venue": jcd,
             "deadline": "23:59", "total": 100, "tickets": [{"combo": combo, "amount": 100}]}
    from .scraper import VENUES
    order["venue"] = VENUES.get(jcd, jcd)

    def guard(total: int) -> None:
        if total != 100:
            raise AbortPurchase(f"テスト注文は100円のはずが {total}円")

    ex = TeleboatExecutor(flow_path=os.environ.get("TELEBOAT_FLOW", "teleboat_flow.py"),
                          headless=os.environ.get("TELEBOAT_HEADLESS", "0") != "0")
    res = ex.buy(order, guard, dry_run=True)
    print(res)
    ok = res.message.startswith("中止: テスト実行")
    print("✅ 投票直前まで正常に動きました" if ok else "❌ 投票直前まで到達できませんでした。data/shots の画像を確認してください")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record")
    t = sub.add_parser("test")
    t.add_argument("--jcd", default="01", help="場コード（その日開催している場を指定）")
    t.add_argument("--rno", type=int, default=12)
    t.add_argument("--combo", default="1-2-3")
    a = ap.parse_args()
    return record() if a.cmd == "record" else test(a.jcd, a.rno, a.combo)


if __name__ == "__main__":
    sys.exit(main())
