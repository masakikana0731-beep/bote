# アプリ本体（スマホ用の画面＋自動予想）を起動し、指定の日時になったら止める。
# Claude アプリの再起動で止まらないよう、WMI から独立したプロセスとして起動する想定:
#   Invoke-CimMethod Win32_Process Create -Arguments @{CommandLine = "powershell -NoProfile -ExecutionPolicy Bypass -File <このファイル> 2026-10-28T09:00"}
# 引数：止める日時（省略時は7日後）。ログは logs\app.log
$wd = Split-Path -Parent $PSScriptRoot
Set-Location $wd
$until = if ($args.Count -ge 1) { [datetime]::Parse($args[0]) } else { (Get-Date).AddDays(7) }
# collect.ps1 と同じ：Claude アプリの中で入れた Python を、アプリの外から見える場所で探す
$home_line = Get-Content (Join-Path $wd ".venv\pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
$pyhome = ($home_line -split '=', 2)[1].Trim()
if (-not (Test-Path (Join-Path $pyhome "python.exe")) -and $pyhome.StartsWith($env:APPDATA)) {
    $rel = $pyhome.Substring($env:APPDATA.Length).TrimStart([char]92)
    $cand = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Packages") -Directory -Filter "Claude_*" -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "LocalCache\Roaming\$rel" } | Where-Object { Test-Path (Join-Path $_ "python.exe") } |
        Select-Object -First 1
    if ($cand) { $pyhome = $cand }
}
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$wd;$wd\.venv\Lib\site-packages"
$log = Join-Path $wd "logs\app.log"
Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') 起動（$($until.ToString('yyyy-MM-dd HH:mm')) に停止予定）" -Encoding UTF8
$p = Start-Process -FilePath (Join-Path $pyhome "python.exe") `
    -ArgumentList "-m", "uvicorn", "boatbot.app:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory $wd -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $wd "logs\app_out.log") -RedirectStandardError (Join-Path $wd "logs\app_err.log")
Set-Content (Join-Path $wd "logs\app.pid") $p.Id
# Wait-Process の待ち時間は最大 32767 秒なので、5分ごとに確認する
while (-not $p.HasExited -and (Get-Date) -lt $until) { Start-Sleep -Seconds 300 }
if (-not $p.HasExited) {
    Stop-Process -Id $p.Id -Force
    Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') 予定どおり停止しました" -Encoding UTF8
} else {
    Add-Content $log "$(Get-Date -Format 'yyyy-MM-dd HH:mm') アプリが予定より早く終了しました（logs\app_err.log を確認）" -Encoding UTF8
}
