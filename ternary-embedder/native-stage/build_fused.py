"""Compile the complete two-layer native-384 array encoder."""
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
    a=p.parse_args()
    os.environ['ENCODER_CAP']=str(a.cap)
    tools=ROOT.parent/'npu-width-probe/tools'
    packager=tools/'xclbinutil-build/Release/xclbinutil.exe'
    os.environ['AIE_XCLBINUTIL']=str(packager)
    os.environ['PATH']=os.pathsep.join([str(packager.parent),str(tools/'deps/x64-windows/bin'),os.environ['PATH']])
    import aie.iron as iron
    from aie.iron.device import from_name
    iron.set_current_device(from_name('npu2',n_cols=8))
    from fused_design import DESIGN,SPECIALIZE
    # Keep failed compiler products for diagnosing the 16 KiB core budget.
    import aie.utils.compile.jit.compilabledesign as compiler_module
    compiler_module._cleanup_failed_compilation=lambda path: None
    out=ROOT/f'build_fused_cap{a.cap}'
    out.mkdir(exist_ok=True)
    work=(out/'final.prj').resolve()
    if not work.is_relative_to(ROOT.resolve()) or work.name!='final.prj':
        raise ValueError('Build workdir must stay inside experiment')
    if work.exists():
        shutil.rmtree(work)
    start=time.perf_counter()
    DESIGN.specialize(**SPECIALIZE).compile(xclbin_path=str(out/'final.xclbin'),inst_path=str(out/'insts.bin'))
    size_tool=Path(os.environ['PEANO_INSTALL_DIR'])/'bin/llvm-size.exe'
    sections=subprocess.check_output([str(size_tool),'-A',str(work/'elfs_main_core_0_2/elfs_main_core_0_2.elf')],text=True)
    text_bytes=int(re.search(r'^\.text\s+(\d+)',sections,re.MULTILINE)[1])
    sources=['fused.h','fused_design.py','vecmath.h','ironutil.py']+sorted(f.name for f in ROOT.glob('ep_*.cc'))
    report={'cap':a.cap,'texts_per_dispatch':128//a.cap,'cores':32,'layers':2,
            'scope':'One dispatch, two full encoder layers; host embedding lookup/input norm/output projection/pooling',
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
