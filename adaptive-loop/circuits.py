"""Random Boolean circuits as a set of nodes wired by labels.

Each circuit has n_nodes nodes: a few inputs (level 0) and gates at levels 1..D. A gate at level l
takes input a from level l-1 (so its level is exact) and input b from any lower level. Nodes are
shuffled and each gets a random label from n_labels, so a model must find a gate's inputs by
matching labels, not by position.

Per position: op (IN0, IN1, AND, OR, XOR, NAND, NOR, NOT), own label, label of input a, label of
input b (NONE for inputs, and b = NONE for NOT). Targets: the node's value and its level.
"""
import numpy as np

OPS = ["IN0", "IN1", "AND", "OR", "XOR", "NAND", "NOR", "NOT"]
N_OPS = len(OPS)
GATE_FNS = {
    2: lambda a, b: a & b,
    3: lambda a, b: a | b,
    4: lambda a, b: a ^ b,
    5: lambda a, b: 1 - (a & b),
    6: lambda a, b: 1 - (a | b),
    7: lambda a, b: 1 - a,
}


def generate(batch, n_nodes, depth_lo, depth_hi, n_labels, rng):
    """Returns dict of int64 arrays, each [batch, n_nodes]: op, lab, la, lb, val, level."""
    NONE = n_labels
    op = np.zeros((batch, n_nodes), np.int64)
    a = np.zeros((batch, n_nodes), np.int64)
    b = np.zeros((batch, n_nodes), np.int64)
    level = np.zeros((batch, n_nodes), np.int64)
    for i in range(batch):
        D = int(rng.integers(depth_lo, depth_hi + 1))
        n_in = int(rng.integers(3, 7))
        assert n_in + D <= n_nodes
        lv = np.concatenate([np.zeros(n_in, np.int64), np.arange(1, D + 1),
                             rng.integers(1, D + 1, n_nodes - n_in - D)])
        lv.sort()
        level[i] = lv
        start = np.searchsorted(lv, lv - 1, "left")   # first node of level l-1
        end = np.searchsorted(lv, lv - 1, "right")    # one past last node of level l-1
        g = lv > 0
        u = rng.random((2, n_nodes))
        a[i] = np.where(g, start + (u[0] * (end - start)).astype(np.int64), 0)
        b[i] = np.where(g, (u[1] * end).astype(np.int64), 0)
        op[i] = np.where(g, rng.integers(2, N_OPS, n_nodes), rng.integers(0, 2, n_nodes))
    # Values, in node order (parents always come earlier), vectorized over the batch.
    val = np.zeros((batch, n_nodes), np.int64)
    rows = np.arange(batch)
    for j in range(n_nodes):
        va, vb = val[rows, a[:, j]], val[rows, b[:, j]]
        o = op[:, j]
        v = np.where(o == 1, 1, 0)
        for k, fn in GATE_FNS.items():
            v = np.where(o == k, fn(va, vb), v)
        val[:, j] = v
    # Shuffle positions and assign labels.
    perm = np.argsort(rng.random((batch, n_nodes)), 1)
    labels = np.argsort(rng.random((batch, n_labels)), 1)[:, :n_nodes]  # label of node j
    gate = level > 0
    la = np.where(gate, labels[rows[:, None], a], NONE)
    lb = np.where(gate & (op != 7), labels[rows[:, None], b], NONE)
    out = {"op": op, "lab": labels, "la": la, "lb": lb, "val": val, "level": level}
    return {k: v[rows[:, None], perm] for k, v in out.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    import time
    t = time.time()
    c = generate(256, 48, 1, 12, 64, rng)
    print(f"{time.time() - t:.3f}s for 256 circuits")
    # Check: recompute values by label lookup, iterating to a fixed point.
    for i in range(4):
        by_lab = {}
        ops, lab, la, lb, val = (c[k][i] for k in ("op", "lab", "la", "lb", "val"))
        for _ in range(48):
            for j in range(48):
                if ops[j] < 2:
                    by_lab[lab[j]] = ops[j]
                elif la[j] in by_lab and (ops[j] == 7 or lb[j] in by_lab):
                    by_lab[lab[j]] = int(GATE_FNS[ops[j]](by_lab[la[j]], by_lab.get(lb[j], 0)))
        assert all(by_lab[lab[j]] == val[j] for j in range(48)), i
    print("values check out; levels:", np.bincount(c["level"].ravel()), "mean val", c["val"].mean())
