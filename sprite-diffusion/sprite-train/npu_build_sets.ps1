param([int[]]$Patches = @(2, 4))
$ErrorActionPreference = 'Stop'
. C:\dev\mlir-aie\iron_env.ps1
if (-not $env:OFLM_ROOT) { $env:OFLM_ROOT = 'C:\code\OpenFlowLM-Next2' }   # any OpenFlowLM checkout: dit_gemm, harness, build tools
foreach ($p in $Patches) {
  python $PSScriptRoot\build_npu_sets.py --patch $p --out "C:\dev\sprite-npu\sets-p$p"
  if ($LASTEXITCODE) { throw "sets p$p failed" }
}
