"""Short titles for prompts. See README.md."""

import ctypes
import os
import sys
from pathlib import Path

__all__ = ["Titler", "TitlerError"]

_HERE = Path(__file__).parent
_LIB_NAMES = {"win32": "titler_c.dll", "darwin": "libtitler_c.dylib"}


class TitlerError(RuntimeError):
    pass


def _load_library() -> ctypes.CDLL:
    name = _LIB_NAMES.get(sys.platform, "libtitler_c.so")
    candidates = [os.environ.get("TITLER_LIB"), _HERE / name]
    for c in candidates:
        if c and Path(c).exists():
            lib = ctypes.CDLL(str(c))
            break
    else:
        raise TitlerError(f"cannot find {name}; build it with `cargo build --release -p titler-capi` and set TITLER_LIB")
    lib.titler_new_from_file.argtypes = [ctypes.c_char_p]
    lib.titler_new_from_file.restype = ctypes.c_void_p
    lib.titler_new_from_memory.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.titler_new_from_memory.restype = ctypes.c_void_p
    lib.titler_free.argtypes = [ctypes.c_void_p]
    lib.titler_free.restype = None
    lib.titler_title_n.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    lib.titler_title_n.restype = ctypes.c_int64
    lib.titler_last_error.argtypes = []
    lib.titler_last_error.restype = ctypes.c_char_p
    lib.titler_version.argtypes = []
    lib.titler_version.restype = ctypes.c_char_p
    return lib


_lib = None


def _library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


def _error() -> str:
    return (_library().titler_last_error() or b"unknown error").decode("utf-8", "replace")


class Titler:
    """A loaded model. Safe to share between threads."""

    def __init__(self, model=None):
        lib = _library()
        if model is None:
            model = _HERE / "titler.gguf"
            if not model.exists():
                raise TitlerError("this install has no bundled model; pass the path of a .gguf file")
        if isinstance(model, (bytes, bytearray, memoryview)):
            data = bytes(model)
            self._handle = lib.titler_new_from_memory(data, len(data))
        else:
            self._handle = lib.titler_new_from_file(os.fspath(model).encode("utf-8"))
        if not self._handle:
            raise TitlerError(_error())

    def title(self, prompt: str, beams: int = 4) -> str:
        if not self._handle:
            raise TitlerError("this Titler is closed")
        lib = _library()
        data = prompt.encode("utf-8", "replace")
        buf = ctypes.create_string_buffer(512)
        n = lib.titler_title_n(self._handle, data, len(data), beams, buf, len(buf))
        if n < 0:
            raise TitlerError(_error())
        return buf.value.decode("utf-8")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _library().titler_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def version() -> str:
    return _library().titler_version().decode()
