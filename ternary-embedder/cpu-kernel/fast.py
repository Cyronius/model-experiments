"""Replace a ternary Student's linears with the packed VNNI kernels (mode 'i8' or 'p2')."""
import ctypes, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; P = HERE.parent
sys.path[:0] = [str(P/'tern-embed')]
from model import tern_codes
lib = ctypes.CDLL(str(HERE/'libtern_cpu.so'))
vp = ctypes.c_void_p; ci = ctypes.c_int

def ptr(t): return vp(t.data_ptr())

def pack2(u):  # u: (N,K) uint8 in 0..2 -> (N,K/4); byte j of each 128-block holds weights j, j+32, j+64, j+96
    N, K = u.shape; b = u.reshape(N, K//128, 4, 32).astype(np.uint8)
    return (b[:, :, 0] | b[:, :, 1] << 2 | b[:, :, 2] << 4 | b[:, :, 3] << 6).reshape(N, K//4)

def tile_i8(u):  # (N,K) -> [ntile][k4][32][4]
    N, K = u.shape
    return np.ascontiguousarray(u.reshape(N//32, 32, K//4, 4).transpose(0, 2, 1, 3))

def tile_p2(u):  # (N,K) -> [ntile][k/16][32][4 bytes]
    N, K = u.shape; b = u.reshape(N//32, 32, K//16, 4, 4).astype(np.uint8)
    by = sum(b[..., s, :] << (2*s) for s in range(4)).astype(np.uint8)  # (ntile,32,K/16,4 bytes)
    return np.ascontiguousarray(by.transpose(0, 2, 1, 3))

class FastLinear(torch.nn.Module):
    def __init__(self, layer, mode):
        super().__init__()
        codes, s = tern_codes(layer.weight.detach())
        u = (codes + 1).to(torch.uint8).numpy(); self.N, self.K = u.shape
        self.gs = 256 if self.K > 256 and self.K % 256 == 0 else self.K
        self.mode = mode; self.v2 = mode.endswith('2')
        self.fn = {'i8': lib.lin_i8, 'p2': lib.lin_p2, 'i8v2': lib.lin2_i8, 'p2v2': lib.lin2_p2}[mode]
        self.W = torch.from_numpy({'i8': lambda: u, 'p2': lambda: pack2(u), 'i8v2': lambda: tile_i8(u), 'p2v2': lambda: tile_p2(u)}[mode]()).contiguous()
        self.ws = s.reshape(-1).float().contiguous(); self.bias = layer.bias.detach().float().contiguous()
    def forward(self, x):
        shp = x.shape; M0 = x.numel() // self.K; x = x.reshape(M0, self.K).float().contiguous(); G = self.K // self.gs
        M = (M0 + 7)//8*8 if self.v2 else M0
        if M != M0: x = torch.cat([x, torch.zeros(M-M0, self.K)])
        q = torch.empty(M, self.K, dtype=torch.int8); sc = torch.empty(M, G); asum = torch.empty(M, G, dtype=torch.int32)
        lib.quant_rows(ptr(x), ci(M), ci(self.K), ci(self.gs), ptr(q), ptr(sc), ptr(asum))
        out = torch.empty(M, self.N)
        self.fn(ptr(q), ptr(sc), ptr(asum), ci(M), ci(self.K), ci(self.gs), ptr(self.W), ptr(self.ws), ptr(self.bias), ci(self.N), ptr(out))
        return out[:M0].reshape(*shp[:-1], self.N)

def fast_student(path, mode):
    from model import Student
    m = Student.load(path).eval()
    for layer in m.layers:
        for n in ('q', 'k', 'v', 'o', 'fc1', 'fc2'): setattr(layer, n, FastLinear(getattr(layer, n), mode))
    return m
