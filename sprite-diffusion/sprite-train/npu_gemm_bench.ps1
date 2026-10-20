# dit_gemm at the sprite DiT's block shapes (64-sprite sheet, CFG doubles the batch), patch 2 vs patch 4; 2026-10-19.
param([string]$Phase = 'all')
$ErrorActionPreference = 'Stop'
. C:\dev\mlir-aie\iron_env.ps1
if (-not $env:OFLM_ROOT) { $env:OFLM_ROOT = 'C:\code\OpenFlowLM-Next2' }   # any OpenFlowLM checkout: dit_gemm, harness, build tools
$repo = $env:OFLM_ROOT
$dg = "$repo\open_kernels\designs\dit_gemm"
$shapes = @(
  @{name='p2_lin1'; M=32768; K=512;  N=4096},
  @{name='p2_lin2'; M=32768; K=1536; N=1024},
  @{name='p4_lin1'; M=8192;  K=512;  N=4096},
  @{name='p4_lin2'; M=8192;  K=1536; N=1024}
)
foreach ($s in $shapes) {
  $b = "C:\dev\sprite-npu\$($s.name)\build"; $t = "C:\dev\sprite-npu\$($s.name)\test"
  if ($Phase -in 'all','build') {
    if (-not (Test-Path "$b\final.xclbin")) {
      $env:DG_M = $s.M; $env:DG_K = $s.K; $env:DG_N = $s.N; Remove-Item Env:DG_LAYOUT -ErrorAction SilentlyContinue
      python "$repo\open_kernels\build_design.py" "$dg\dit_gemm.py" $b
      if ($LASTEXITCODE) { throw "build $($s.name) failed" }
    }
    python "$dg\make_test.py" --M $s.M --K $s.K --N $s.N --out $t --build $b --runs 8
  }
  if ($Phase -in 'all','run') {
    Write-Host "== $($s.name) $($s.M)x$($s.K)x$($s.N)"
    python "$repo\open_kernels\harness\bench.py" "$t\run_t.cfg" --driver "$repo\open_kernels\harness\out\run_kernel.exe" --warm 2
    python "$dg\compare.py" $t t
  }
}
