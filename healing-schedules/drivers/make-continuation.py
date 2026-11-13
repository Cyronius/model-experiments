"""Package a finished arm's final model as a start artifact for run.py --damaged, so the arm can
be continued (fresh optimizer; use --data-offset <rows the arm drew> to continue its data stream).
Usage: make-continuation.py <run-dir> <original-damaged-artifact> <output-dir>"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pruning import sha256  # noqa: E402

run, original, out = map(Path, sys.argv[1:4])
if out.exists():
    raise FileExistsError(out)
summary = json.loads((run / 'summary.json').read_text())
if summary['status'] != 'budget' or not summary['canary']['end']['ok']:
    raise ValueError(f'{run} did not finish cleanly')
provenance = json.loads((run / 'provenance.json').read_text())
if provenance['damaged_manifest_sha256'] != sha256(original / 'pruning.json'):
    raise ValueError('Run was not made from this damaged artifact')
out.mkdir(parents=True)
for path in sorted((run / 'final').iterdir()):
    shutil.copy2(path, out / path.name)
manifest = json.loads((original / 'pruning.json').read_text())
curve = json.loads((run / 'curve.json').read_text())
manifest['artifact_sha256'] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
manifest['continued_from'] = dict(run=str(run), strategy=provenance['arguments']['strategy'],
                                  original_artifact=str(original), original_manifest_sha256=sha256(original / 'pruning.json'),
                                  rows_drawn=summary['tokens'] // 2048,  # mix-2048 rows; one row per sequence
                                  tokens=summary['tokens'], training_flops=summary['training_flops'],
                                  original_start_validation_ce=curve[0]['validation']['ce'],
                                  final_validation_ce=curve[-1]['validation']['ce'],
                                  note='Weights only: the optimizer state was not saved, so continuation restarts AdamW and warmup')
(out / 'pruning.json').write_text(json.dumps(manifest, indent=1) + '\n')
print(json.dumps(manifest['continued_from']))
