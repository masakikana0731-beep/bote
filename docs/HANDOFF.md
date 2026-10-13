# 引き継ぎメモ（ボート予想くん）

別セッション（自宅PCの Claude Code / Remote Control）で作業を続けるためのまとめ。
**新しいセッションは最初にこのファイルを読むこと。**

- リポジトリ: `masakikana0731-beep/bote`
- 作業ブランチ: `claude/boat-racing-auto-invest-74mg7w`（main への PR は未作成）
- 最終更新: 2026-10-13（クラウドセッションで作成）

---

## 1. ユーザーの要望と方針（これまでの合意事項）

- 目的は「稼ぐ」より **「楽しむ」**。**1日の最大投資額を必ず守る**
- スマホから操作。毎日のログイン程度の手作業はOK
- スライダで「何を優先するか」「ギャンブル性」などを調整したい
- 実行場所は **自宅PC / Raspberry Pi**（常時起動）。スマホからブラウザでアクセス
- 予想データは公式サイト（boatrace.jp）の出走表＋オッズ＋公式一括データ
- 自動購入は **ユーザーがリスクを理解したうえで実装を希望**
  - テレボート約定 第25条は、所定画面以外・提供外ツールでの投票を禁止しており、違反と判断されると予告なく利用停止の可能性がある（約定PDF 2013年3月版で確認。最新版かは未確認）
  - そのため本番モードは二重の許可制（画面＋環境変数 `BOATBOT_ALLOW_REAL=1`）、画面操作はユーザーが録画して `teleboat_flow.py` を作る方式。**まずは仮想モードで試す**方針
- ユーザー設定：あいまいな内容・未確認の事実はその旨を明示。会話は簡潔に、長文になるときは最後に要点を箇条書きで

## 2. できているもの（push 済み）

| 機能 | ファイル | 状態 |
|---|---|---|
| 公式サイトの出走表・直前情報・オッズ・結果の取得 | `boatbot/scraper.py` | 完成 |
| スマホ用Web画面（今日／設定／成績）、トークン認証、ntfy通知 | `boatbot/app.py`, `boatbot/static/` | 完成 |
| 定期実行（締切 N 分前に予想→通知／自動購入、結果取り込み、毎朝の過去データ更新） | `boatbot/engine.py` | 完成 |
| 予算管理（1日上限・1レース上限・最大R数・最終上限 `BOATBOT_HARD_DAILY_CAP`） | `boatbot/store.py`, `engine.py` | 完成 |
| 自動投資：オフ／仮想／本番、緊急停止、購入不明・3連続失敗で自動停止、購入ログ | `boatbot/executor.py`, `engine.py` | 仮想は完成、本番は `teleboat_flow.py` の録画待ち |
| 録画・投票直前までのテスト用コマンド | `boatbot/teleboat_cli.py`, `teleboat_flow.example.py` | 完成（本物のテレボートでは未検証） |
| 簡易予想モデル（重視スライダ） | `boatbot/model.py` | 完成 |
| 学習モデル（過去3年・LightGBM LambdaRank＋段階別温度 Plackett-Luce） | `boatbot/ml.py`, `models/` | 完成・本番組み込み済み |
| 公式一括データ（番組表B／競走成績K）の取得・パース（全7券種の払戻含む） | `boatbot/history.py` | 完成 |
| バックテスト（3連単） | `boatbot/backtest.py` | 完成 |
| 3連単の買い方最適化（3,000通り） | `boatbot/optimize.py` | 完成。結果は `docs/model.md`, `docs/results/` |
| **全券種・単位・配分を含む最適化** | `boatbot/optimize2.py` | **作成済み・本番の探索は未実行（データ取得待ち）** |

詳細な説明：`README.md`（使い方）、`docs/model.md`（特徴量・モデル・精度・最適化結果）

## 3. これまでの主な結果

- 学習モデル（テスト期間 13,588R）：1着的中 57.1%（コース勝率だけなら 54.1%）。ただし **オッズ（市場）の方がやや正確**（3連単 対数尤度 -3.89 vs -3.76）
- オッズ帯別に全目を買った場合の回収率：1〜5倍 86% … 1000倍以上 42%（**大穴ほど割に合わない**）
- 3連単の最適化（3,106R・前後半＋ブートストラップ）
  - 当たる楽しさ：的中25%・回収86%
  - バランス：的中19%・回収99%（90%区間 75〜125%）
  - 一発狙い：的中6%・回収137%（数回の大当たり依存。偶然の可能性が高い）
- これらは設定画面のプリセットボタンとして実装済み
- 全券種の試運転（283R）では「完全ランダムに買う」でも回収147%が出た → 少数レースの結果は運に大きく左右される

## 4. データの引き継ぎ（作業は自宅PCに一本化。クラウド側の取得は停止済み）

クラウドで取得したオッズを `data_snapshot/odds_20260715-20261012.tar.gz`（2MB）に入れてある。
- `data/odds/`：3連単の確定オッズ **3,240レース（全件）**
- `data/odds_multi/`：3連複・2連単・2連複・拡連複・単勝・複勝 **882/3,240レース**

自宅PCでの手順：
```bash
# 3連単オッズの取得(fetch-odds)が動いていたら止める（スナップショットに全件ある）
git pull
tar -xzf data_snapshot/odds_20260715-20261012.tar.gz      # data/odds と data/odds_multi に展開
python -m boatbot.backtest fetch-odds-multi --workers 4   # 残り約2,360レースだけ取得（取得済みは飛ばす）
```
- 対象：2026/7/15〜10/12、1日3場（`fetch-odds --seed 1` で選んだ場）
- 展開すると、選んだ場の一覧 `data/odds/*/_venues.json` もクラウドで使ったものになる

## 5. 次にやること（未完了タスク）

1. **全券種オッズの取得を完了**（上記）
2. **全券種での最適化**
   ```bash
   python -m boatbot.optimize2 prepare
   python -m boatbot.optimize2 search --n 20000 --jobs 4     # data/opt2_results.json
   ```
   - 探索する変数：券種（7種＋全券種から自動）、ギャンブル性、目安の幅、点数、期待値下限、人気の信じ度、配分（均等／払戻揃え／確率比例／ケリー）、1口の単位（100/200/500円）、1レース上限、堅いレース見送り、風速見送り
   - **評価は「前半(〜8/28)で選んだ設定を後半(8/29〜)でそのまま使った成績」を主に報告**（3,000通りから選ぶと見かけが良くなるため）
   - 比較対象：`random_baseline`（ランダム買い）、人気だけ（crowd=1.0）
   - 日単位ブートストラップで回収率の90%区間も出す（`optimize.py` で使った方法と同じ）
   - 結果を `docs/model.md` に追記し、ユーザーに報告（要点の箇条書き付き）
3. 結果しだいで **アプリに券種・単位の設定を追加**
   - `model.py` の `Settings` / `make_plan`、`engine.py`（券種ごとのオッズ取得：`scraper.parse_odds_tables` / `parse_win_place` は実装済み）、画面、`teleboat_flow`（券種ごとの入力）
4. 自宅PCでのセットアップ（下記）と仮想モードでの試運転

## 6. 自宅PCでのセットアップ

```bash
git clone https://github.com/masakikana0731-beep/bote.git && cd bote
git checkout claude/boat-racing-auto-invest-74mg7w
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
# 学習モデルは models/ に同梱。本番の予想に必要な過去データを一度だけ取得（数時間）
python -m boatbot.history download --start 20230701 --end <昨日のYYYYMMDD>
python -m boatbot.history build
./run.sh      # スマホから http://<PCのIP>:8000
```
- 外出先から使うなら Tailscale 等＋ `BOATBOT_TOKEN` を設定
- 通知は ntfy アプリ＋ `NTFY_TOPIC`
- 本番自動購入の準備は README「本番モードの準備」

## 7. 注意点・未確認事項

- テレボートのURL（既定 `https://spweb.brtb.jp/`）が正しいかは未確認
- `parse_race_name` が場によってレース種別でなく大会名を拾うことがある（特徴量の重要度は0.1%なので影響は小さい）
- 今節成績は、本番では当日の先のレースの結果も含む（学習データは朝時点）。影響は小さいと推測
- 学習モデルの再学習は手動（`python -m boatbot.ml train`）。期間は `ml.py` の `TRAIN_START / TRAIN_END / VALID_END`
- 確定オッズでのシミュレーションなので、締切数分前のオッズで買う実運用とはずれる
- `data/` は git 管理外（`.gitignore`）
- テスト：`python -m pytest`（26件）
