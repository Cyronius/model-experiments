"""Fetch a bounded, pinned sample of Horizon Parquet row groups (no paid APIs)."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SOURCES = {
    "math": ("IFM/Math-Reasoning", "e249295774c3a66943e1483dbf53e0c811bc41dd", "math-thinking-qwen"),
    "code": ("IFM/Code-Reasoning", "e971bd7801485b24bb217dcde9d46f4bad42ea00", "codeio"),
    "instruction": ("IFM/SFT-Reasoning", "4d4dbbe65e478a2ca179c4cce43b5e1f051aacd2", "3efforts-pretrain"),
}


def fetch(out, groups=4, seed=251025, replay_manifest=None):
    out.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=False)
    replay=json.loads(replay_manifest.read_text()) if replay_manifest else None
    if replay:
        seed=replay['seed'];groups=replay['groups_per_source']
    manifest = {"seed": seed, "groups_per_source": groups, "sources": {}, "files": []}
    for kind, (repo, revision, config) in SOURCES.items():
        prefix = f"datasets/{repo}@{revision}/{config}"
        if replay:
            expected=[f for f in replay['files'] if f['file'].startswith(kind+'-')]
            selected=[f['source_file'] for f in expected]
            if any(not p.startswith(prefix+'/') for p in selected):raise ValueError('Different pinned source')
        else:
            files = sorted(x["name"] for x in fs.ls(prefix, detail=True) if x["name"].endswith(".parquet"))
            selected = random.Random(f"{seed}:{kind}").sample(files, min(groups, len(files)))
        manifest["sources"][kind] = {"repo": repo, "revision": revision, "config": config}
        for i, path in enumerate(selected):
            dest = out / f"{kind}-{i:02d}.jsonl"
            meta = dest.with_suffix(".meta.json")
            if not (dest.exists() and meta.exists()):
                with fs.open(path, "rb", block_size=1024*1024, cache_type="readahead") as remote:
                    parquet = pq.ParquetFile(remote)
                    group = expected[i]['row_group'] if replay else random.Random(f"{seed}:{path}").randrange(parquet.num_row_groups)
                    count = parquet.metadata.row_group(group).num_rows
                    compressed = sum(parquet.metadata.row_group(group).column(c).total_compressed_size
                                     for c in range(parquet.metadata.num_columns))
                    if compressed > 256*1024*1024 or count > 100000:
                        raise ValueError("Refusing unexpectedly large row group")
                    print(kind, i, "group", group, "rows", count, "compressed_MB", round(compressed/1e6, 2), flush=True)
                    rows = parquet.read_row_group(group).to_pylist()
                offset = sum(parquet.metadata.row_group(g).num_rows for g in range(group))
                with dest.with_suffix(".tmp").open("w") as f:
                    for j, row in enumerate(rows):
                        f.write(json.dumps({"kind": kind, "source_file": path, "row": offset+j,
                                            "source_group": group, **row}, ensure_ascii=False)+"\n")
                dest.with_suffix(".tmp").replace(dest)
                meta.write_text(json.dumps({"file": dest.name, "source_file": path, "row_group": group,
                    "rows": len(rows), "compressed_bytes": compressed,
                    "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}, indent=2)+"\n")
            record=json.loads(meta.read_text())
            if record['source_file']!=path or record['sha256']!=hashlib.sha256(dest.read_bytes()).hexdigest():
                raise ValueError('Cached row group differs')
            if replay and record!=expected[i]:raise ValueError('Replayed row group differs')
            manifest["files"].append(record)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("target/manifold-healing/horizon-raw"))
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--replay-manifest",type=Path,help='Reproduce recorded source files and row groups exactly')
    args = ap.parse_args()
    fetch(args.out, args.groups,replay_manifest=args.replay_manifest)
