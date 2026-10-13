"""自動購入の実行部分。

- PaperExecutor   : 仮想購入（お金は動かない）。すぐ使える。
- TeleboatExecutor: Playwright でテレボートを操作する本番用。
                    画面操作そのものは teleboat_flow.py（利用者が録画して作る）に書く。

⚠️ テレボートの約定 第25条は、振興会が提供するもの以外のツールでの投票を禁止しており、
   違反と判断されると予告なく利用停止になることがある。本番モードは自己責任で。
"""
from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger("boatbot")


@dataclass
class BuyResult:
    ok: bool
    message: str
    uncertain: bool = False  # 買えたかどうか分からない（二重購入防止のため自動投資を止める）


class AbortPurchase(Exception):
    """最終確認で中止（お金は使っていない）"""


class Executor:
    name = "base"

    def buy(self, order: dict, guard: Callable[[int], None]) -> BuyResult:
        raise NotImplementedError


class PaperExecutor(Executor):
    name = "paper"

    def buy(self, order, guard):
        try:
            guard(order["total"])
        except AbortPurchase as e:
            return BuyResult(False, f"中止: {e}")
        return BuyResult(True, "仮想購入しました")


@dataclass
class Credentials:
    member_no: str = ""    # 加入者番号
    pin: str = ""          # 暗証番号
    password: str = ""     # 認証用パスワード
    vote_password: str = ""  # 投票用パスワード

    @classmethod
    def from_env(cls) -> "Credentials":
        return cls(os.environ.get("TELEBOAT_MEMBER_NO", ""), os.environ.get("TELEBOAT_PIN", ""),
                   os.environ.get("TELEBOAT_PASSWORD", ""), os.environ.get("TELEBOAT_VOTE_PASSWORD", ""))


def load_flow(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} がありません。teleboat_flow.example.py をコピーして作ってください")
    spec = importlib.util.spec_from_file_location("teleboat_flow", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "place_bets"):
        raise AttributeError(f"{path} に place_bets(page, order, creds, confirm) がありません")
    return mod


class TeleboatExecutor(Executor):
    """teleboat_flow.place_bets(page, order, creds, confirm) を呼ぶ。

    flow は「投票」ボタンを押す直前に必ず confirm(画面に表示された合計金額) を呼ぶこと。
    - confirm より前の失敗 → お金は使っていないので「失敗」
    - confirm より後の失敗 → 買えたか不明なので「不明」として自動投資を停止
    """
    name = "real"

    def __init__(self, flow_path: str | Path = "teleboat_flow.py",
                 state_path: str | Path = "data/teleboat_state.json",
                 shot_dir: str | Path = "data/shots", headless: bool = True):
        self.flow_path, self.state_path = Path(flow_path), Path(state_path)
        self.shot_dir, self.headless = Path(shot_dir), headless

    def buy(self, order, guard, dry_run: bool = False):
        if os.environ.get("BOATBOT_ALLOW_REAL") != "1" and not dry_run:
            return BuyResult(False, "本番モードは環境変数 BOATBOT_ALLOW_REAL=1 が必要です")
        try:
            flow = load_flow(self.flow_path)
            from playwright.sync_api import sync_playwright
        except Exception as e:
            return BuyResult(False, f"準備失敗: {e}")

        confirmed = {"done": False}

        def confirm(displayed_total: int) -> None:
            if displayed_total != order["total"]:
                raise AbortPurchase(f"画面の合計{displayed_total}円 ≠ 予定{order['total']}円")
            guard(order["total"])
            if dry_run:
                raise AbortPurchase("テスト実行のため投票直前で停止")
            confirmed["done"] = True

        self.shot_dir.mkdir(parents=True, exist_ok=True)
        shot = self.shot_dir / f"{order['hd']}_{order['jcd']}_{order['rno']}.png"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            ctx = browser.new_context(
                storage_state=str(self.state_path) if self.state_path.exists() else None,
                viewport={"width": 414, "height": 896}, locale="ja-JP")
            page = ctx.new_page()
            try:
                flow.place_bets(page, order, Credentials.from_env(), confirm)
                if not confirmed["done"]:
                    return BuyResult(False, "flow が confirm() を呼ばずに終了しました（未投票のはず）")
                return BuyResult(True, "テレボートで購入しました")
            except AbortPurchase as e:
                return BuyResult(False, f"中止: {e}")
            except Exception as e:
                log.exception("テレボート操作失敗")
                if confirmed["done"]:
                    return BuyResult(False, f"投票ボタン以降でエラー（購入されたか不明）: {e}", uncertain=True)
                return BuyResult(False, f"投票前にエラー: {e}")
            finally:
                try:
                    page.screenshot(path=str(shot), full_page=True)
                    ctx.storage_state(path=str(self.state_path))
                except Exception:
                    pass
                browser.close()
