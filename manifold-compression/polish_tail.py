# Short end-to-end polish of a projected student's last few layers and final norm on KL to the teacher; the rest stays frozen (2026-10-19).
import argparse
import time

import torch
import torch.nn.functional as F

from common import KLAccumulator, Runner, add_session_args, kl_per_token, load_model, save_json, session_from

ap = argparse.ArgumentParser()
add_session_args(ap, bs=8)
ap.add_argument("--student", required=True)
ap.add_argument("--tail", type=int, default=4, help="last layers trained, plus the final norm")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
args = ap.parse_args()

S = session_from(args)
r, dev = S.runner, S.device
student, _ = load_model(args.student)
sr = Runner(student)
L = sr.L
start = 2 * (L - args.tail)
IDS = torch.stack([torch.tensor(x[1]) for x in S.data["fit"]])
n = len(IDS)
S.eval_batches()
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


@torch.no_grad()
def student_kl():
    acc = KLAccumulator()
    for b in S.eval_batches():
        emb, ctx = sr.prepare(b["ids"])
        logits, _ = sr.run(emb, 0, ctx, positions=b["pos"], keep_first=False)
        acc.add(kl_per_token(b["ref_lp"].to(dev), logits), b["doms"], b["focus"])
    return acc.report()


with torch.no_grad():
    z = []
    for i in range(0, n, 8):
        emb, ctx = sr.prepare(IDS[i:i + 8])
        z.append(sr.run(emb, 0, ctx, record={start}, positions=torch.tensor([0]), keep_first=False)[1][start].half())
    z = torch.cat(z)
before = student_kl()
log(f"before polish: KL {before['all']['mean']:.4f}")

params = [p for li in range(L - args.tail, L) for p in sr.lm.layers[li].parameters()] + [sr.lm.norm.weight]
for p in params:
    p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
pos = torch.arange(1, S.T, 2)
for step in range(args.steps):
    idx = torch.randperm(n)[:args.batch]
    with torch.no_grad():
        emb, ctx_t = r.prepare(IDS[idx])
        ref = F.log_softmax(r.run(emb, 0, ctx_t, positions=pos, keep_first=False)[0].float(), -1)
    _, ctx = sr.prepare(IDS[idx])
    h = z[idx].float()
    for s in range(start, 2 * L):
        i, half = divmod(s, 2)
        h = h + (sr.mixer(i, h, ctx) if half == 0 else sr.mlp(i, h))
    lp = F.log_softmax(student.lm_head(sr.lm.norm(h[:, pos])).float(), -1)
    loss = (ref.exp() * (ref - lp)).sum(-1).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 100 == 0:
        log(f"step {step}: train KL {loss.item():.4f}")
for p in params:
    p.requires_grad_(False)

after = student_kl()
log("POLISHED vs 2B: KL " + f"{after['all']['mean']:.4f}  " + "  ".join(f"{d} {after[d]['mean']:.4f}" for d in after if d != "all"))
save_json({"args": vars(args), "before": before, "after": after}, S.results / f"polish{args.tag}.json")
student.to(torch.bfloat16).save_pretrained(args.out)
S.tok.save_pretrained(args.out)
log(f"saved {args.out}")
