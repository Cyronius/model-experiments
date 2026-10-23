# Labels under dibs, yielding the gpu when another session asks; -Job oga (default) or tsr; 2026-10-17.
param([ValidateSet('oga', 'tsr')][string]$Job = 'oga')
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$dibs = 'C:\Users\josha\.claude\plugins\cache\dibs\dibs\0.3.0\bin\dibs.exe'
$env:DIBS_EXE = $dibs
$py = 'C:\dev\igpu-venv\Scripts\python.exe'
$logs = 'C:\dev\sprite-data\logs'
$env:YIELD_GPU = '1'
$what = if ($Job -eq 'tsr') { @("$PSScriptRoot\label_tsr.py", 'sprites2.jsonl') } else { @("$PSScriptRoot\harvest_oga.py", 'label', '--once', '--yield-gpu') }
while ($true) {
    $cmd = @('run', 'gpu', '--label', "`"sprite labelling $Job (yields on request)`"", '--eta', '25m', '--', $py, '-W', 'ignore') + $what
    $p = Start-Process $dibs -ArgumentList $cmd -Wait -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$logs\label-$Job-dibs.out" -RedirectStandardError "$logs\label-$Job-dibs.err"
    Get-Content "$logs\label-$Job-dibs.out" | Add-Content "$logs\label-$Job-dibs.log"
    "$(Get-Date -Format s) step exit $($p.ExitCode)" | Add-Content "$logs\label-$Job-dibs.log"
    if ($p.ExitCode -eq 3 -or ($Job -eq 'tsr' -and $p.ExitCode -eq 0)) { break }
    if ($p.ExitCode -eq 75) {
        Start-Sleep 30
        Start-Process $dibs -ArgumentList @('wait', 'gpu') -Wait -WindowStyle Hidden   # the waiting session takes the lock first
    } elseif ($p.ExitCode -ne 0) { break }
}
