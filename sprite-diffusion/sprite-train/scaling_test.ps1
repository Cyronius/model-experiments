# The 25/50/100% data-scaling test, one sliced run after another (each resumes after any interruption); 2026-10-18.
param([int]$Steps = 11700, [int]$Patch = 4)
foreach ($frac in 0.25, 0.5, 1.0) {
    & "$PSScriptRoot\train_dibs.ps1" -Out "C:\dev\sprite-data\runs\scale-p$Patch-f$frac" --frac $frac --patch $Patch --steps $Steps --ckpt-every 1500
}
