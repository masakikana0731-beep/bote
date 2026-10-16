# 収集を起動する。Claude アプリの再起動で止まらないよう、WMI から独立したプロセスとして起動する想定:
#   Invoke-CimMethod Win32_Process Create -Arguments @{CommandLine = "powershell -NoProfile -ExecutionPolicy Bypass -File <このファイル> watch"}
#   引数は boatbot.collect にそのまま渡す（例: before --start 20180101 --workers 6）
# WMI から起動すると venv の python.exe（ランチャー）が元の Python を見つけられないので、
# 元の Python を直接使い、venv のライブラリを PYTHONPATH で渡す。
$wd = Split-Path -Parent $PSScriptRoot
Set-Location $wd
$home_line = Get-Content (Join-Path $wd ".venv\pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' } | Select-Object -First 1
$pyhome = ($home_line -split '=', 2)[1].Trim()
# Claude アプリ（ストア版）の中で入れた Python は、アプリの外からは
# %LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\... に見える（AppData がアプリ専用の場所に振り替えられるため）
if (-not (Test-Path (Join-Path $pyhome "python.exe")) -and $pyhome.StartsWith($env:APPDATA)) {
    $rel = $pyhome.Substring($env:APPDATA.Length).TrimStart('')
    $cand = Get-ChildItem (Join-Path $env:LOCALAPPDATA "Packages") -Directory -Filter "Claude_*" -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "LocalCache\Roaming\$rel" } | Where-Object { Test-Path (Join-Path $_ "python.exe") } |
        Select-Object -First 1
    if ($cand) { $pyhome = $cand }
}
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$wd;$wd\.venv\Lib\site-packages"
$name = $args[0]
& (Join-Path $pyhome "python.exe") -m boatbot.collect @args *>> (Join-Path $wd "logs\collect_$name.log")
