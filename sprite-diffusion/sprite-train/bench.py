# Training throughput of klein-style DiT blocks (head dim 128, 64 tokens, batch 256, bf16
# autocast, AdamW) at three sizes. Usage: bench.py cuda   (C:\dev\igpu-venv, ROCm iGPU)
#                                          bench.py cpu
import time, torch, torch.nn as nn, torch.nn.functional as F, sys
dev = sys.argv[1]
class Blk(nn.Module):
    def __init__(s, h):
        super().__init__(); s.n1=nn.LayerNorm(h); s.qkv=nn.Linear(h,3*h); s.o=nn.Linear(h,h)
        s.n2=nn.LayerNorm(h); s.f1=nn.Linear(h,6*h); s.f2=nn.Linear(3*h,h); s.H=h//128
    def forward(s,x):
        B,T,C=x.shape; q,k,v=s.qkv(s.n1(x)).view(B,T,3,s.H,128).permute(2,0,3,1,4)
        x=x+s.o(F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(B,T,C))
        a,b=s.f1(s.n2(x)).chunk(2,-1); return x+s.f2(F.silu(a)*b)
for h,L in [(256,8),(384,10),(512,12)]:
    m=nn.Sequential(*[Blk(h) for _ in range(L)]).to(dev); p=sum(t.numel() for t in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),1e-4); B,T=256,64
    x=torch.randn(B,T,h,device=dev)
    def step():
        with torch.autocast(dev if dev!='cuda' else 'cuda',dtype=torch.bfloat16):
            l=m(x).float().pow(2).mean()
        l.backward(); opt.step(); opt.zero_grad()
    for _ in range(3): step()
    if dev=='cuda': torch.cuda.synchronize()
    t=time.time(); n=10
    for _ in range(n): step()
    if dev=='cuda': torch.cuda.synchronize()
    dt=(time.time()-t)/n
    print(f'{dev} h={h} L={L} params={p/1e6:.1f}M  {dt*1000:.0f} ms/step(B=256)  {B/dt:.0f} samples/s  {6*p*T*B/dt/1e12:.2f} TFLOPS', flush=True)
