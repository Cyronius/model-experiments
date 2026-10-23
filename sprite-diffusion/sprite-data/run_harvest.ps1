# Every step resumes from its cache, so re-running this script after -Stop continues where it left off.
param([switch]$Stop, [switch]$Crawl)
$ErrorActionPreference = 'Stop'
if ($Stop) {
    Stop-ScheduledTask -TaskName SpriteHarvest -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'sprite-data' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId): $($_.CommandLine)" }
    return
}
$here = $PSScriptRoot
$sys = 'C:\Python311\python.exe'
$gpu = 'C:\dev\igpu-venv\Scripts\python.exe'
$logs = 'C:\dev\sprite-data\logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$ts = Get-Date -Format 'yyyyMMdd-HHmm'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
$quota = 'snes:300,nes:250,game_boy_advance:200,sega_genesis:100,game_boy_gbc:80,turbografx_16:40,master_system:30,game_gear:30'

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # keep the machine awake while this runs

function Launch($name, $exe, [string[]]$argv) {
    $p = Start-Process $exe -ArgumentList $argv -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$logs\$name-$ts.log" -RedirectStandardError "$logs\$name-$ts.err"
    $null = $p.Handle   # without a cached handle, ExitCode reads back as null after exit
    "$(Get-Date -Format s) start $name (pid $($p.Id))" | Add-Content "$logs\run-$ts.log"
    $p
}
function Finish($name, $p) {
    $p.WaitForExit()
    "$(Get-Date -Format s) end $name (exit $($p.ExitCode))" | Add-Content "$logs\run-$ts.log"
    if ($p.ExitCode -ne 0) { throw "$name failed, see $logs\$name-$ts.err" }
}
function Step($name, $exe, [string[]]$argv) { Finish $name (Launch $name $exe $argv) }

if (-not $Crawl) {   # TSR answers bots with a Cloudflare challenge since 2026-10-08
    Step 'label' $gpu @("$here\label_tsr.py", 'sprites2.jsonl')
    "$(Get-Date -Format s) all done" | Add-Content "$logs\run-$ts.log"
    return
}
# The crawl is network-bound and the labelling GPU-bound, so they overlap.
$crawl = Launch 'crawl' $sys @("$here\crawl_tsr.py", '--unseeded', '--quota', $quota)
Step 'label-existing' $gpu @("$here\label_tsr.py", 'sprites2.jsonl')
Finish 'crawl' $crawl
Step 'extract' $sys @("$here\selftrain_extract.py", '--out', 'sprites2.jsonl')
Step 'clean' $gpu @("$here\selftrain_clean.py", 'sprites2.jsonl')
Step 'label' $gpu @("$here\label_tsr.py", 'sprites2.jsonl')
"$(Get-Date -Format s) all done" | Add-Content "$logs\run-$ts.log"
