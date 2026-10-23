. C:\dev\mlir-aie\iron_env.ps1
if (-not $env:OFLM_ROOT) { $env:OFLM_ROOT = 'C:\code\OpenFlowLM-Next2' }   # any OpenFlowLM checkout: dit_gemm, harness, build tools
$env:PYTHONIOENCODING = 'utf-8'
foreach ($round in 1, 2) {
  foreach ($p in 4, 2) {
    foreach ($s in 8, 16) {
      Write-Host "== round $round patch $p steps $s"
      python $PSScriptRoot\npu_sprite.py --weights "C:\dev\sprite-npu\weights-p$p.npz" --sets "C:\dev\sprite-npu\sets-p$p" sheet --cat food --steps $s --seed 1 --out "C:\dev\sprite-npu\sheet-p$p-s$s-r$round.png"
    }
  }
}
