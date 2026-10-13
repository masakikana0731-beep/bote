"""テレボートの画面操作（本番自動購入用）のひな形。

使い方:
  1. このファイルを teleboat_flow.py という名前でコピーする（teleboat_flow.py は git に含めない）
  2. `python -m boatbot.teleboat_cli record` でブラウザを開き、
     ログイン → 場・レース選択 → 3連単の買い目と金額入力 → 確認画面 まで実際に操作する
     （右側に Playwright のコードが自動で出るので、それを下の TODO に貼り替える）
     ※ 録画中は最後の「投票」ボタンは押さなくてよい
  3. `python -m boatbot.teleboat_cli test` で、投票ボタンの直前まで動くか確認する（お金は使わない）

守ること:
  - 最後の「投票」ボタンを押す直前に、必ず confirm(画面に表示された合計金額) を呼ぶ
    （予算チェックや金額の照合はそこで行われる。呼ばないと購入扱いにならない）
  - 失敗したら例外を投げる（握りつぶさない）
"""
import re


def place_bets(page, order, creds, confirm):
    """
    page   : Playwright の Page（スマホ画面サイズ）
    order  : {"jcd": "01", "venue": "桐生", "rno": 5, "total": 500,
              "tickets": [{"combo": "1-2-3", "amount": 200}, ...]}
    creds  : creds.member_no / creds.pin / creds.password / creds.vote_password（環境変数から）
    confirm: 投票ボタン直前に confirm(画面の合計金額:int) を呼ぶ
    """
    # --- 1. ログイン -------------------------------------------------------
    page.goto("https://spweb.brtb.jp/")  # TODO: 実際に使っているテレボートのURL
    # TODO: 録画したログイン操作に置き換える。例:
    # page.get_by_label("加入者番号").fill(creds.member_no)
    # page.get_by_label("暗証番号").fill(creds.pin)
    # page.get_by_label("認証用パスワード").fill(creds.password)
    # page.get_by_role("button", name="ログイン").click()

    # --- 2. 場・レースを選ぶ -----------------------------------------------
    # TODO: order["venue"]（例: "桐生"）と order["rno"] を使って選ぶ。例:
    # page.get_by_text(order["venue"]).click()
    # page.get_by_text(f"{order['rno']}R", exact=True).click()

    # --- 3. 3連単の買い目と金額を入力 ---------------------------------------
    for t in order["tickets"]:
        first, second, third = t["combo"].split("-")
        units = t["amount"] // 100  # テレボートの金額入力は100円単位のことが多い（要確認）
        # TODO: 1着=first, 2着=second, 3着=third を選び、units を入力してセットに追加する
        pass

    # --- 4. 確認画面で合計金額を読み取る ------------------------------------
    # TODO: 確認画面の合計金額の表示を取得する。例:
    # text = page.get_by_text(re.compile(r"合計")).inner_text()
    text = ""
    displayed_total = int(re.sub(r"\D", "", text) or 0)

    # --- 5. 最終チェック（必須） ------------------------------------------
    confirm(displayed_total)

    # --- 6. 投票 ----------------------------------------------------------
    # TODO: 投票用パスワード入力と投票ボタン。例:
    # page.get_by_label("投票用パスワード").fill(creds.vote_password)
    # page.get_by_role("button", name="投票").click()

    # --- 7. 完了を確認 -----------------------------------------------------
    # TODO: 完了画面の文言を待つ（見つからなければ例外になり「購入不明」として自動投資が止まる）。例:
    # page.get_by_text("投票を受け付けました").wait_for(timeout=15000)
    raise NotImplementedError("teleboat_flow.py の TODO を埋めてください")
