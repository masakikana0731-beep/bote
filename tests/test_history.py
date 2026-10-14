from datetime import date

from boatbot.history import parse_k

# 2019年の競走成績（K）ファイルの形式。レース名に全角スペースが入る（「予　選」）
K_TEXT = """01KBGN
   1R       予　選　　　                 H1800m  晴　  風  東　　 4m  波　  3cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ 逃げ
-------------------------------------------------------------------------------
  01  1 4787 椎　名　　　　豊 36   54  6.69   1    0.11     1.50.4
  02  5 4108 吉　村　　正　明 28   43  6.78   5    0.07     1.51.7
  03  3 4366 前　沢　　丈　史 54   50  6.76   3    0.16     1.52.6
  04  2 4415 下　出　　卓　矢 15   42  6.75   2    0.11     1.54.2
  05  6 4659 木　下　　翔　太 73   25  6.78   6    0.15     1.55.7
  06  4 4413 水　摩　　　　敦 65   20  6.79   4    0.13     1.56.1

        単勝     1          110
        ３連単   1-5-3     3030  人気     9

   2R       一　般　　　                 H1800m  曇り  風  南東　 4m  波　  4cm
  01  5 4108 吉　村　　正　明 28   43  6.78   5    0.07     1.51.7
01KEND
"""


def test_parse_k_race_name_with_fullwidth_spaces():
    boats, races = parse_k(K_TEXT, date(2019, 6, 1))
    assert [r["rno"] for r in races] == [1, 2]
    assert races[0]["wind_dir"] == "東" and races[0]["wind"] == 4 and races[0]["wave"] == 3
    assert races[0]["tri_combo"] == "1-5-3" and races[0]["tri_payout"] == 3030
    assert [b["rno"] for b in boats] == [1] * 6 + [2]   # 2R の艇が 1R に紛れ込まない
    assert boats[0]["toban"] == "4787" and boats[0]["finish"] == 1 and boats[0]["course"] == 1
