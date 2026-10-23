# Trains in slices under dibs: yields the gpu on request, resumes from the checkpoint; args are train.py's; 2026-10-18.
param([Parameter(Mandatory)][string]$Out, [Parameter(ValueFromRemainingArguments)]$Rest)
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$dibs = 'C:\Users\josha\.claude\plugins\cache\dibs\dibs\0.3.0\bin\dibs.exe'
$env:DIBS_EXE = $dibs
$py = 'C:\dev\igpu-venv\Scripts\python.exe'
$name = Split-Path $Out -Leaf
New-Item -ItemType Directory -Force $Out | Out-Null
$log = "$Out\wrapper.log"
while ($true) {
    $cmd = @('run', 'gpu', '--label', "`"sprite training $name (yields on request)`"", '--eta', '25m', '--', $py, '-W', 'ignore',
             "$PSScriptRoot\train.py", '--out', $Out, '--yield-gpu', '--slice-steps', '1500') + $Rest
    $p = Start-Process $dibs -ArgumentList $cmd -Wait -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$Out\run.out" -RedirectStandardError "$Out\run.err"
    Get-Content "$Out\run.out" | Add-Content "$Out\train.log"
    "$(Get-Date -Format s) exit $($p.ExitCode)" | Add-Content $log
    if ($p.ExitCode -eq 3) { break }
    if ($p.ExitCode -eq 75) {
        Start-Sleep 30
        Start-Process $dibs -ArgumentList @('wait', 'gpu') -Wait -WindowStyle Hidden   # the waiting session takes the lock first
    } elseif ($p.ExitCode -ne 0) { break }
}
