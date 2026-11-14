"""Build the Linux persistent XRT C ABI using an installed XRT SDK.

`python build_host.py td_npu2` builds the asynchronous superset used by fused2."""
import sys
import json
import os
from pathlib import Path
import shlex
import subprocess
import hashlib

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

ROOT=Path(__file__).resolve().parent


def main():
    if os.name!='posix':
        raise RuntimeError('This build helper targets Linux')
    xrt=Path(os.environ.get('XILINX_XRT','/opt/xilinx/xrt'))
    if not (xrt/'include/xrt/xrt_device.h').exists():
        raise RuntimeError('Set XILINX_XRT to the installed XRT SDK')
    lib=next((p for p in (xrt/'lib',xrt/'lib64') if (p/'libxrt_coreutil.so').exists()),None)
    if lib is None:raise RuntimeError('Cannot find libxrt_coreutil.so in XRT SDK')
    out=ROOT/'host/out';out.mkdir(parents=True,exist_ok=True)
    name=sys.argv[1] if len(sys.argv)>1 else 'td_npu'
    source=ROOT/f'host/{name}.cpp';target=out/f'lib{name}.so'
    command=shlex.split(os.environ.get('CXX','g++'))+[
        '-std=c++17','-shared','-fPIC','-O2',str(source),'-I'+str(xrt/'include'),
        '-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lxrt_coreutil','-o',str(target)]
    subprocess.run(command,check=True)
    (out/('build.json' if name=='td_npu' else f'{name}.build.json')).write_text(json.dumps({'command':command,'source_sha256':digest(source),
                                            'library_sha256':digest(target)},indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
