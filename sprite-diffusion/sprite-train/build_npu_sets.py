# Builds the sprite DiT's three NPU kernel sets (gemm, fa, ew) for one patch size and sheet size; ironenv; 2026-10-19.
#   . C:\dev\mlir-aie\iron_env.ps1; $env:OFLM_ROOT = <OpenFlowLM checkout>; python build_npu_sets.py --patch 2 --out C:\dev\sprite-npu\sets-p2
import argparse, json, os, sys
from pathlib import Path

# dit_gemm, build_design.py and the exporter's helpers come from an OpenFlowLM checkout; dit_fa / dit_ew are this experiment's forks.
ROOT = Path(os.environ['OFLM_ROOT'])
KERNELS = Path(__file__).resolve().parents[1] / 'npu-kernels'
sys.path.insert(0, str(ROOT / 'open_kernels'))
import export_dit_kernels as edk  # noqa: E402
edk.FA_DESIGN = KERNELS / 'dit_fa' / 'dit_fa.py'
edk.EW_DESIGN = KERNELS / 'dit_ew' / 'dit_ew.py'

EL = 384


def streams(patch, sprites):
    T = (32 // patch) ** 2
    M = 2 * sprites * T            # CFG: conditional rows, then unconditional
    half = M // 2
    lda = 64 if patch == 2 else 128
    gemm = {
        'inp': dict(M=M, K=512, N=1024, layout={'lda': lda, 'a_size': M * lda + 512}),
        'lin1': dict(M=M, K=512, N=4096),
        'lin2': dict(M=M, K=1536, N=1024),
        'outp': dict(M=M, K=512, N=1024),
    }
    fa = {'attn': dict(L=M, heads=4, seq=T, qkv_ld=4096, k_col=512, v_col=1024, o_ld=1536, o_col=0)}
    v = lambda T_, ld, off=0, E=1: {'off': off, 'ld': ld, 'T': T_, 'E': E}  # noqa: E731
    ew = {
        'ln': {'op': 'ln_mod', 'a': v(half, 1024), 'y': v(half, 512), 'p_off': EL, 'n_par': 2,
               'idx': {'shift': 0, 'scale': 1}, 'W': EL,
               'sizes': {'X': half * 1024, 'B': EL, 'P': 4 * EL, 'Y': half * 512, 'Z': EL}},
        'res': {'op': 'res_ln_mod', 'a': v(half, 1024), 'b': v(half, 1024), 'y': v(half, 512), 'z': v(half, 1024),
                'p_off': EL, 'n_par': 4, 'idx': {'gate': 0, 'shift': 1, 'scale': 2, 'bias': 3}, 'W': EL,
                'sizes': {'X': half * 1024, 'B': half * 1024, 'P': 6 * EL, 'Y': half * 512, 'Z': half * 1024}},
        'qk': {'op': 'qk', 'a': v(M, 4096), 'b': v(M, 4096, 512), 'y': v(M, 4096), 'z': v(M, 4096, 512),
               'p_off': EL, 'n_par': 7, 'rope': 'sprite', 'grid_w': 32 // patch, 'heads': 3,
               'b_q_heads': 0, 'b_k_heads': 3,
               'sizes': {'X': M * 4096, 'B': M * 4096, 'P': 9 * EL, 'Y': M * 4096, 'Z': M * 4096}},
        'swiglu': {'op': 'swiglu', 'a': v(M, 4096, 1536, 3), 'b': v(M, 4096, 2688, 3), 'y': v(M, 1536, 384, 3),
                   'sizes': {'X': M * 4096, 'B': M * 4096, 'P': EL, 'Y': M * 1536, 'Z': EL}},
    }
    return dict(M=M, T=T, lda=lda, gemm=gemm, fa=fa, ew=ew)


def jobs(sp, out):
    j = {}
    for n, s in sp['gemm'].items():
        j[f'gemm/{n}'] = ({k: s[k] for k in ('M', 'K', 'N')} | ({'layout': s['layout']} if 'layout' in s else {}),
                          {'DG_M': s['M'], 'DG_K': s['K'], 'DG_N': s['N'], 'DG_LAYOUT': json.dumps(s.get('layout', {}))},
                          edk.DESIGN, out / 'gemm')
    for n, s in sp['fa'].items():
        j[f'fa/{n}'] = (s | {'tau': edk.FA_TAU},
                        {'DF_L': s['L'], 'DF_HEADS': s['heads'], 'DF_SEQ': s['seq'], 'DF_TAU': edk.FA_TAU,
                         'DF_QKV_LD': s['qkv_ld'], 'DF_K_COL': s['k_col'], 'DF_V_COL': s['v_col'],
                         'DF_O_LD': s['o_ld'], 'DF_O_COL': s['o_col']},
                        edk.FA_DESIGN, out / 'fa')
    for n, s in sp['ew'].items():
        j[f'ew/{n}'] = (s | {'EL': EL, 'NPAR': 7}, {'DE_SPEC': json.dumps(s), 'DE_EL': EL, 'DE_NPAR': 7},
                        edk.EW_DESIGN, out / 'ew')
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch', type=int, required=True)
    ap.add_argument('--sprites', type=int, default=64)
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    sp = streams(a.patch, a.sprites)
    for k in ('gemm', 'fa', 'ew'):
        (out / k).mkdir(parents=True, exist_ok=True)
    built = {}
    for name, job in jobs(sp, out).items():
        kset, sname = name.split('/')
        built.setdefault(kset, {})[sname] = edk.build_stream(sname, *job, a.force)
    for kset, dirs in built.items():
        edk.assemble(out / kset, dirs, out / kset / 'set.json', {'streams': list(dirs)})
    (out / 'sprite_sets.json').write_text(json.dumps(sp | {'patch': a.patch, 'sprites': a.sprites}, indent=1))
    print('sets complete:', out)


if __name__ == '__main__':
    main()
