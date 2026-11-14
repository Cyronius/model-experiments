"""Compile an experimental fused2 encoder variant into build_f2_<tag>."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cap',type=int,choices=(8,32,128),default=32)
    p.add_argument('--layers',type=int,choices=(2,3),default=3)
    p.add_argument('--stub',default='',help="'all', or comma-separated kernels (attend,linear,...) that return immediately")
    p.add_argument('--bdepth',type=int,choices=(1,2),default=1,help='Broadcast FIFO depth')
    p.add_argument('--defs',default='',help='Comma-separated kernel macros, e.g. F2_PROF_NOEPI')
    p.add_argument('--opt',default='',choices=('','0','1','2','3'),help='aiecc -O level for core code')
    a=p.parse_args()
    os.environ['ENCODER_CAP']=str(a.cap)
    os.environ['ENCODER_LAYERS']=str(a.layers)
    os.environ['F2_STUB']=a.stub
    os.environ['F2_BDEPTH']=str(a.bdepth)
    os.environ['F2_OPT']=a.opt
    os.environ['F2_DEFS']=a.defs
    if os.name == 'nt':
        tools=ROOT.parent/'npu-width-probe/tools'
        packager=tools/'xclbinutil-build/Release/xclbinutil.exe'
        os.environ.setdefault('AIE_XCLBINUTIL',str(packager))
        os.environ['PATH']=os.pathsep.join([str(packager.parent),str(tools/'deps/x64-windows/bin'),os.environ['PATH']])
    else:
        packager=os.environ.get('AIE_XCLBINUTIL') or shutil.which('xclbinutil')
        if not packager:
            raise RuntimeError('Set AIE_XCLBINUTIL or put xclbinutil on PATH')
        os.environ['AIE_XCLBINUTIL']=str(packager)
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name('npu2',n_cols=8))
    from fused2_design import DESIGN,SPECIALIZE,SOURCES,PACKETS
    # Keep failed compiler products for diagnosing the 16 KiB core budget.
    import aie.utils.compile.jit.compilabledesign as compiler_module
    compiler_module._cleanup_failed_compilation=lambda path: None
    tag=f'L{a.layers}_cap{a.cap}_d{a.bdepth}'+(f'_O{a.opt}' if a.opt else '')+(''.join('_'+m.replace('F2_PROF_','').lower() for m in a.defs.split(',') if m))+(f"_stub-{a.stub.replace(',','+')}" if a.stub else '')
    out=ROOT/f'build_f2_{tag}'
    out.mkdir(exist_ok=True)
    work=(out/'final.prj').resolve()
    if not work.is_relative_to(ROOT.resolve()) or work.name!='final.prj':
        raise ValueError('Build workdir must stay inside experiment')
    if work.exists():
        shutil.rmtree(work)
    start=time.perf_counter()
    DESIGN.specialize(**SPECIALIZE).compile(xclbin_path=str(out/'final.xclbin'),inst_path=str(out/'insts.bin'))
    size_tool=Path(os.environ['PEANO_INSTALL_DIR'])/('bin/llvm-size.exe' if os.name == 'nt' else 'bin/llvm-size')
    sections=subprocess.check_output([str(size_tool),'-A',str(work/'elfs_main_core_0_2/elfs_main_core_0_2.elf')],text=True)
    text_bytes=int(re.search(r'^\.text\s+(\d+)',sections,re.MULTILINE)[1])
    sources=SOURCES
    report={'tag':tag,'stub':a.stub,'broadcast_depth':a.bdepth,'packets':PACKETS,'aiecc_opt':a.opt,'cap':a.cap,'texts_per_dispatch':128//a.cap,'cores':32,'layers':a.layers,
            'scope':f'One dispatch, {a.layers} full encoder layers; host embedding lookup/input norm/output projection/pooling',
            'program_text_bytes':text_bytes,'stack_bytes_per_core':0x1400,
            'mlir_aie':importlib.metadata.version('mlir_aie'),
            'llvm_aie':importlib.metadata.version('llvm-aie'),
            'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
            'artifact_sha256':{n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('final.xclbin','insts.bin')},
            'build_seconds':time.perf_counter()-start}
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
