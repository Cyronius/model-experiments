"""Stage 5 step 2-3: assemble the release folder.

    python assemble.py MODEL.tde DEST [--name NAME]

DEST receives: model.tde, tokenizer.json, libtde.so, tde/ (package), src/ + build.sh (library
source), azure_probe.py, drift_canary.py + canary.json + canary-vectors.npy, config.json.
README.md is written separately (it carries measured numbers).
"""
import json, shutil, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
QS = HERE.parent
sys.path.insert(0, str(QS)); sys.path.insert(0, str(QS / 'runtime'))
from common import PROJECT, WORK, sha256
import shared
from tde import read_model


def main():
    model, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    rt = QS / 'runtime'
    shutil.copy2(model, dest / 'model.tde')
    shutil.copy2(PROJECT / 'tokenizer/tokenizer.json', dest / 'tokenizer.json')
    shutil.copy2(rt / 'build/libtde.so', dest / 'libtde.so')
    if (dest / 'tde').exists():
        shutil.rmtree(dest / 'tde')
    shutil.copytree(rt / 'tde', dest / 'tde', ignore=shutil.ignore_patterns('__pycache__'))
    if (dest / 'src').exists():
        shutil.rmtree(dest / 'src')
    shutil.copytree(rt / 'src', dest / 'src')
    shutil.copy2(rt / 'build.sh', dest / 'build.sh')
    shutil.copy2(rt / 'azure_probe.py', dest / 'azure_probe.py')
    shutil.copy2(HERE / 'drift_canary.py', dest / 'drift_canary.py')
    # drift canary: 50 fixed SciFact documents (public), cut to the contract, with their hosted vectors
    d = shared.dev_task('SciFact')
    idx = sorted(np.random.default_rng(20261112).choice(len(d['texts']), 50, replace=False).tolist())
    texts = [shared.cut_to_contract(d['texts'][i], 504) for i in idx]
    vecs = shared.embed(texts, plan='A')  # cache hits (Stage 1 dev vectors)
    (dest / 'canary.json').write_text(json.dumps({'date': '2026-11-12', 'source': 'BEIR SciFact corpus, 50 documents, seed 20261112, cut to the 504 contract',
                                                  'endpoint': shared.URL, 'doc_ids': [d['cids'][i] for i in idx], 'texts': texts}, indent=1) + '\n')
    np.save(dest / 'canary-vectors.npy', vecs)
    cfg, meta, _ = read_model(dest / 'model.tde')
    s2 = json.loads((QS / 'stage2/result.json').read_text())
    s3 = json.loads((QS / 'stage3/result.json').read_text()) if (QS / 'stage3/result.json').exists() else {}
    config = {
        'name': dest.name, 'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'teacher': 'Qwen/Qwen3-Embedding-8B', 'provider': 'DeepInfra (direct), endpoint ' + shared.URL,
        'served_version_label': 'excl-v1', 'date_checked': '2026-11-12',
        'stored_dimension': 1024, 'output_dimension': cfg['out'],
        'document_contract': 'first 504 WordPiece content tokens (BERT uncased vocab), cut never inside a word; '
                             'send Embedder.contract_text(text) to the hosted model when refining',
        'reading_mode': meta.get('reading_mode'),
        'query_protocol': 'hosted model, input string "Instruct: {instruction}\\nQuery:{query}", custom_instruction empty, '
                          'full 4096-d vector requested (no dimensions field), prefix-cut and re-normalised locally',
        'weight_format': meta.get('weight_format'), 'embedding_format': meta.get('embedding_format'),
        'head': s2.get('selected_arm'), 'calibration': dict(s2.get('calibration', {}), variant=s2['gates']['G2']['variant']),
        'refinement_signal': s3.get('recommended_signal'),
        'source_checkpoint': meta.get('source_checkpoint'), 'source_checkpoint_sha256': meta.get('source_checkpoint_sha256'),
        'head_file_sha256': meta.get('head_file_sha256'),
        'files_sha256': {f: sha256(dest / f) for f in ('model.tde', 'tokenizer.json', 'libtde.so')},
        'language': 'English only',
    }
    (dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config, indent=1))


if __name__ == '__main__':
    main()
