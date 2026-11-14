"""Stage 0 step 4: uint16 memmap of the training rows' token ids (cut 504 = full,
since training rows fit the contract), with starts; same for validation rows.
Usage: python -I stage0/train_memmap.py"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

jw = C.WORK / "jaswanth"
ids = np.memmap(jw / "ids.u16", dtype=np.uint16, mode="r")
st = np.load(jw / "starts.npy")
for name in ("train", "val"):
    rows = np.load(C.WORK / "stage0" / f"{name}_rows.npy")
    out = C.WORK / "stage0" / name
    out.mkdir(parents=True, exist_ok=True)
    lens = st[rows + 1] - st[rows]
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    mm = np.memmap(out / "ids.u16", dtype=np.uint16, mode="w+", shape=(int(starts[-1]),))
    for r, a in zip(rows, starts[:-1]):
        mm[a:a + st[r + 1] - st[r]] = ids[st[r]:st[r + 1]]
    mm.flush()
    np.save(out / "starts.npy", starts)
    np.save(out / "rows.npy", rows)
    print(name, len(rows), int(starts[-1]))
