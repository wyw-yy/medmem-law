# 夜间串行链 v2：等当前主网格队列退出后，依次跑 deep -> cross(seed1) -> main seed2 -> main seed3。
# 队列可续跑；中断后重跑本脚本即可。
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$chainLog = Join-Path $root 'build\chain.log'

function Note($msg) {
    $line = '[' + (Get-Date -Format 'HH:mm:ss') + '] ' + $msg
    Add-Content -Path $chainLog -Value $line -Encoding utf8
}

Note 'chain v2 启动，等待当前队列退出'

$deadline = (Get-Date).AddMinutes(400)
while ((Get-Date) -lt $deadline) {
    $q = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*night_queue*' }
    if (-not $q) { break }
    Start-Sleep -Seconds 60
}
Note '队列已退出，开始串行阶段'

$stages = @('score1:1', 'deep:6', 'cross:4', 'seed2:5', 'seed3:5')
foreach ($s in $stages) {
    $name = $s.Split(':')[0]
    $hours = $s.Split(':')[1]
    $logPath = Join-Path $root ('build\queue_' + $name + '64.log')
    Note ('开始 ' + $name)
    if ($name -eq 'score1') {
        & $py -u -m tools.night_queue --plan main --seed 1 --hours $hours *>> $logPath
    } elseif ($name -eq 'deep') {
        & $py -u -m tools.night_queue --plan deep --hours $hours *>> $logPath
    } elseif ($name -eq 'cross') {
        & $py -u -m tools.night_queue --plan cross --hours $hours *>> $logPath
    } else {
        $sd = $name.Replace('seed', '')
        & $py -u -m tools.night_queue --plan main --seed $sd --hours $hours *>> $logPath
    }
    Note ('结束 ' + $name + ' exit=' + $LASTEXITCODE)
}

Note '全部完成'