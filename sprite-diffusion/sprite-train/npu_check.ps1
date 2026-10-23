param([int[]]$Patches = @(4, 2))
. C:\dev\mlir-aie\iron_env.ps1
if (-not $env:OFLM_ROOT) { $env:OFLM_ROOT = 'C:\code\OpenFlowLM-Next2' }   # any OpenFlowLM checkout: dit_gemm, harness, build tools
$env:PYTHONIOENCODING = 'utf-8'
foreach ($p in $Patches) {
  Write-Host "== patch $p"
  python $PSScriptRoot\npu_sprite.py --weights "C:\dev\sprite-npu\weights-p$p.npz" --sets "C:\dev\sprite-npu\sets-p$p" check --golden "C:\dev\sprite-npu\goldens\p$p.npz"
}
