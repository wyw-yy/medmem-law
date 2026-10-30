# Overnight serial chain v3.
# Waits until the currently running main grid stops writing its log, then runs
# deep -> cross -> seed2 -> seed3 in sequence. Every stage is resumable.
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$chainLog = Join-Path $root 'build\chain.log'
$mainLog = Join-Path $root 'build\queue_main64.log'

function Note($msg) {
    $line = '[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $msg
    Add-Content -Path $chainLog -Value $line -Encoding utf8
}

Note 'chain v3 started, waiting for main grid to go quiet'

$deadline = (Get-Date).AddHours(6)
while ((Get-Date) -lt $deadline) {
    $age = (Get-Date) - (Get-Item $mainLog).LastWriteTime
    if ($age.TotalMinutes -gt 5) { break }
    Start-Sleep -Seconds 60
}
Note ('main grid quiet for ' + [int]((Get-Date) - (Get-Item $mainLog).LastWriteTime).TotalMinutes + ' min, starting stages')

$stages = @('score1:1', 'deep:6', 'cross:4', 'seed2:5', 'seed3:5')
foreach ($s in $stages) {
    $name = $s.Split(':')[0]
    $hours = $s.Split(':')[1]
    $logPath = Join-Path $root ('build\queue_' + $name + '64.log')
    Note ('start ' + $name)
    if ($name -eq 'score1') {
        # 主网格的训练已经由 queue_main64 完成，这里只补评分（done_ep 已达标 → 跳过训练）
        & $py -u -m tools.night_queue --plan main --seed 1 --hours $hours *>> $logPath
    } elseif ($name -eq 'deep') {
        & $py -u -m tools.night_queue --plan deep --hours $hours *>> $logPath
    } elseif ($name -eq 'cross') {
        & $py -u -m tools.night_queue --plan cross --hours $hours *>> $logPath
    } else {
        $sd = $name.Replace('seed', '')
        & $py -u -m tools.night_queue --plan main --seed $sd --hours $hours *>> $logPath
    }
    Note ('end ' + $name + ' exit=' + $LASTEXITCODE)
}

Note 'all stages done'