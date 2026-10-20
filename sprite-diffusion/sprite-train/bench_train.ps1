# Times train.py itself at patch 2 (256 tokens) and patch 4 (64 tokens) on the iGPU; 2026-10-17.
$py = 'C:\dev\igpu-venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
$out = "$env:TEMP\sprite-bench"
foreach ($cfg in @(@{p=2; h=384; d=10}, @{p=4; h=384; d=10})) {
    $dir = "$out\p$($cfg.p)-h$($cfg.h)-$(Get-Date -Format yyyyMMddHHmmss)"   # fresh dir, or train.py would resume
    "== patch $($cfg.p) hidden $($cfg.h) depth $($cfg.d)"
    & $py -W ignore "$PSScriptRoot\train.py" --out $dir --patch $cfg.p --hidden $cfg.h --depth $cfg.d --steps 200 --ckpt-every 1000000 2>&1 |
        Select-String 'params|step '
}
