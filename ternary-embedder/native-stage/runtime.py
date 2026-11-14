"""Load the persistent XRT host ABI on Linux or Windows."""
import ctypes as ct
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_LIBRARY = (Path('C:/code/.oflm-retired-worktrees/files/openflowlm-tiny/open_kernels/designs/tiny_planner/host/out/td_npu.dll')
                   if os.name == 'nt' else ROOT/'host/out/libtd_npu.so')
LIBRARY = Path(os.environ.get('TERNARY_NPU_LIBRARY', str(DEFAULT_LIBRARY)))


def load_runtime(library=None):
    directories = []
    if os.name == 'nt':
        directories = [os.add_dll_directory(p) for p in ('C:/dev/xrtNPUfromDLL','C:/Windows/System32/AMD')]
    dll = ct.CDLL(str(library or LIBRARY))
    dll.tdn_open.argtypes, dll.tdn_open.restype = [ct.c_char_p,ct.c_char_p],ct.c_void_p
    dll.tdn_buf.argtypes, dll.tdn_buf.restype = [ct.c_void_p,ct.c_size_t],ct.c_void_p
    dll.tdn_sync.argtypes = [ct.c_void_p,ct.c_int,ct.c_int,ct.c_size_t,ct.c_size_t]
    dll.tdn_sync.restype = None
    dll.tdn_run.argtypes, dll.tdn_run.restype = [ct.c_void_p],ct.c_double
    dll.tdn_close.argtypes, dll.tdn_close.restype = [ct.c_void_p],None
    if hasattr(dll, 'tdn_bind'):
        dll.tdn_bind.argtypes, dll.tdn_bind.restype = [ct.c_void_p,ct.c_int,ct.POINTER(ct.c_int)],ct.c_int
        dll.tdn_start.argtypes, dll.tdn_start.restype = [ct.c_void_p,ct.c_int],ct.c_int
        dll.tdn_wait.argtypes, dll.tdn_wait.restype = [ct.c_void_p,ct.c_int],ct.c_double
    if hasattr(dll, 'tdn_last_error'):
        dll.tdn_last_error.argtypes=[]
        dll.tdn_last_error.restype=ct.c_char_p
        def check(result, function, arguments):
            error=dll.tdn_last_error()
            if error:
                raise RuntimeError(f'{function.__name__}: {error.decode(errors="replace")}')
            return result
        for name in ('tdn_open','tdn_buf','tdn_sync','tdn_run')+(('tdn_bind','tdn_start','tdn_wait') if hasattr(dll,'tdn_bind') else ()):
            getattr(dll,name).errcheck=check
    return dll, directories
