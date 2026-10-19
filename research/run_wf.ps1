# ウォークフォワード検証を最初から最後まで実行する（Claude アプリの再起動で止まらないよう、WMI から起動する想定）
#   1. 直前情報の読み直し 2. 特徴量 3. 四半期ごとのモデル（2つずつ並行）4. ルール探索
$wd = Split-Path -Parent $PSScriptRoot
Set-Location $wd
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
$py = Join-Path $pyhome "python.exe"
$log = Join-Path $wd "logs\run_wf.log"
function Step($title, $cmd) {
    Add-Content $log "$(Get-Date -Format 'MM/dd HH:mm') $title" -Encoding UTF8
    cmd.exe /c "`"$py`" $cmd >> `"$log`" 2>&1"
}
Step "直前情報の読み直し" "-m boatbot.collect parse-before"
Step "特徴量" "research\wf_models.py prepare"
$folds = @(@("20240930","20241231"), @("20241231","20250331"), @("20250331","20250630"), @("20250630","20250930"),
           @("20250930","20251231"), @("20251231","20260331"), @("20260331","20260630"), @("20260630","20261015"))
for ($i = 0; $i -lt $folds.Count; $i += 2) {
    Add-Content $log "$(Get-Date -Format 'MM/dd HH:mm') モデル $($folds[$i][0]) / $($folds[$i+1][0])" -Encoding UTF8
    $ps = @()
    foreach ($f in $folds[$i..($i+1)]) {
        $ps += Start-Process -FilePath cmd.exe -ArgumentList "/c `"`"$py`" research\wf_models.py fold $($f[0]) $($f[1]) >> `"$wd\logs\wf_models.log`" 2>&1`"" -WindowStyle Hidden -PassThru
    }
    $ps | Wait-Process
}
Step "ルール探索" "research\wf_rules.py"
Add-Content $log "$(Get-Date -Format 'MM/dd HH:mm') 完了" -Encoding UTF8
