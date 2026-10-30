# 第二阶段接力：等 deep 队列彻底退出后，接着跑 --plan night
# （cross 跨数据集 4 个 run + seed 2 的 5 个 run）。
#
# 只启动一次这个脚本；它自己不会再拉起自己。
$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root
$log = Join-Path $root "build\phase2.log"

function Say($m) {
    $line = "$(Get-Date -Format 'HH:mm:ss') $m"
    Write-Output $line
    $line | Out-File -Append -Encoding utf8 $log
}

Say "phase2 启动，等待 deep 队列（night_queue）退出…"
$waited = 0
while ($true) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
             Where-Object { $_.CommandLine -like '*night_queue*' }
    if (-not $procs) { break }
    Start-Sleep -Seconds 60
    $waited += 1
    if ($waited % 10 -eq 0) { Say "仍在等待（已等 $waited 分钟）" }
}

Say "deep 队列已退出，开始 night 计划（cross + seed2）"
$py = Join-Path $root ".venv\Scripts\python.exe"
& $py -u -m tools.night_queue --plan night --hours 20 *> (Join-Path $root "build\queue_night.log")
Say "night 计划结束，退出码 $LASTEXITCODE"