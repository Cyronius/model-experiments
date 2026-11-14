"""Heads from 384-d student features to the 4096-d teacher space (Stage 2)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIXES = (256, 512, 1024, 4096)


def prefix_loss_np(pred, tgt):
    """Mean over prefixes of 1 - cosine(prefix(pred), prefix(target)); numpy, per-row mean."""
    out = []
    for k in PREFIXES:
        a = pred[:, :k] / np.linalg.norm(pred[:, :k], axis=1, keepdims=True)
        b = tgt[:, :k] / np.linalg.norm(tgt[:, :k], axis=1, keepdims=True)
        out.append(1 - (a * b).sum(1))
    return float(np.mean(out))


def prefix_loss(pred, tgt):
    loss = 0
    for k in PREFIXES:
        loss = loss + (1 - F.cosine_similarity(pred[:, :k], tgt[:, :k], dim=1)).mean()
    return loss / len(PREFIXES)


class Standardize:
    def __init__(self, X):
        self.mu = X.mean(0).astype(np.float32)
        self.sd = (X.std(0) + 1e-6).astype(np.float32)

    def __call__(self, X):
        return (X - self.mu) / self.sd


class Ridge:
    """Closed-form linear map with bias on standardized features."""
    kind = 'ridge'

    def fit(self, X, Y, lam):
        self.std = Standardize(X)
        Z = self.std(X).astype(np.float64)
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        A = Z.T @ Z
        reg = lam * len(Z) * np.eye(A.shape[0]); reg[-1, -1] = 0
        self.W = np.linalg.solve(A + reg, Z.T @ Y.astype(np.float64)).astype(np.float32)
        return self

    @staticmethod
    def gram(X, Y, std):
        Z = std(X).astype(np.float64)
        Z = np.hstack([Z, np.ones((len(Z), 1))])
        return Z.T @ Z, Z.T @ Y.astype(np.float64), len(Z)

    def raw(self, X):
        return self.std(X) @ self.W[:-1] + self.W[-1]

    def state(self):
        return {'kind': 'ridge', 'mu': self.std.mu, 'sd': self.std.sd, 'W': self.W[:-1], 'b': self.W[-1]}


class MLP(nn.Module):
    """384 -> 2048 -> 4096 with GELU, plus a linear skip 384 -> 4096."""

    def __init__(self, d=384, h=2048, out=4096):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, out)
        self.skip = nn.Linear(d, out)

    def forward(self, z):
        return self.fc2(F.gelu(self.fc1(z), approximate='tanh')) + self.skip(z)


class MLPHead:
    kind = 'mlp'

    def __init__(self, std, net):
        self.std, self.net = std, net

    @torch.inference_mode()
    def raw(self, X):
        out = []
        for i in range(0, len(X), 8192):
            out.append(self.net(torch.from_numpy(self.std(X[i:i + 8192]).astype(np.float32))).numpy())
        return np.concatenate(out)

    def state(self):
        sd = {k: v.detach().numpy() for k, v in self.net.state_dict().items()}
        return {'kind': 'mlp', 'mu': self.std.mu, 'sd': self.std.sd, **sd}


def train_mlp(X, Y, Xv, Yv, epochs=6, lr=1e-3, batch=1024, wd=0.01, margin=None, seed=0, log=print):
    """margin: dict(Fpos, Fneg, Q, tpos, tneg) for the H3 term (teacher dot-product gaps)."""
    torch.manual_seed(seed)
    std = Standardize(X)
    Z = torch.from_numpy(std(X).astype(np.float32))
    T = torch.from_numpy(np.asarray(Y))
    Zv = std(Xv).astype(np.float32)
    net = MLP()
    # initialise the skip path at the ridge solution scale: plain default init is fine for AdamW
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * (len(Z) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    if margin is not None:
        Mp = torch.from_numpy(std(margin['Fpos']).astype(np.float32))
        Mn = torch.from_numpy(std(margin['Fneg']).astype(np.float32))
        MQ = torch.from_numpy(margin['Q'].astype(np.float32))
        Mgap = torch.from_numpy((margin['tpos'] - margin['tneg']).astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    hist = []
    head = MLPHead(std, net)
    for ep in range(epochs):
        perm = torch.randperm(len(Z), generator=g)
        net.train()
        tot = 0
        for s in range(len(Z) // batch):
            idx = perm[s * batch:(s + 1) * batch]
            loss = prefix_loss(net(Z[idx]), T[idx].float())
            if margin is not None:
                j = torch.randint(0, len(Mp), (256,), generator=g)
                pp = F.normalize(net(Mp[j]), dim=1); pn = F.normalize(net(Mn[j]), dim=1)
                gap = (MQ[j] * (pp - pn)).sum(1)
                loss = loss + 1.0 * ((gap - Mgap[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        net.eval()
        vl = prefix_loss_np(head.raw(Xv), np.asarray(Yv, np.float32))
        hist.append({'epoch': ep + 1, 'train_loss': tot / (len(Z) // batch), 'val_loss': vl})
        log(hist[-1])
    return head, hist
