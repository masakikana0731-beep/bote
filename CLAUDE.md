# CLAUDE.md

競艇の予想・自動投資ツール「ボート予想くん」（Python / FastAPI / LightGBM）。

- **作業を始める前に `docs/HANDOFF.md` を読むこと**（これまでの経緯・合意事項・未完了タスク）
- 作業ブランチ: `claude/boat-racing-auto-invest-74mg7w`
- テスト: `python -m pytest`
- 使い方は `README.md`、モデルの説明は `docs/model.md`
- ユーザーへの返答は日本語・簡潔に。未確認の事実はその旨を明示し、長文の最後に要点を箇条書きで付ける
- 1日の投資上限などの安全装置を弱める変更はしない
