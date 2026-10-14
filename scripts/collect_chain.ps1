# 収集の続き：実行中の収集プロセスが終わるのを待ってから、次の範囲を取りに行く。
#   powershell -File scripts\collect_chain.ps1 -WaitPid <PID> -Cmd before -Start 20180101 -End 20261013 -Workers 6
param(
    [int]$WaitPid = 0,
    [string]$Cmd = "before",
    [string]$Start = "20180101",
    [string]$End = "",
    [int]$Workers = 2
)
$wd = Split-Path -Parent $PSScriptRoot
Set-Location $wd
$env:PYTHONUTF8 = "1"
if ($WaitPid -gt 0) {
    try { Wait-Process -Id $WaitPid -ErrorAction Stop } catch {}
}
$py = Join-Path $wd ".venv\Scripts\python.exe"
$pyArgs = @("-m", "boatbot.collect", $Cmd, "--start", $Start, "--workers", "$Workers")
if ($End) { $pyArgs += @("--end", $End) }
& $py @pyArgs *>> (Join-Path $wd "logs\collect_${Cmd}_chain.log")
