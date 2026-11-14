"""Integrate reviewed program checks, then freeze reproducible experiment inputs."""
import argparse
import ast
import copy
import json
from pathlib import Path
import shutil

from retention_prepare import OUT, ROOT, SOURCE, PARENT, canonical, save, sha


def integrate(draft):
    if (OUT / 'frozen.json').exists():
        raise ValueError('Already frozen')
    approvals_path = OUT / 'program-domain-approvals.json'
    approvals = json.loads(approvals_path.read_text())
    if approvals['reviewer'] != 'claude':
        raise ValueError('Missing peer domain review')
    audit = json.loads((draft / 'program-domain-review.json').read_text())
    if approvals.get('revised_review_sha256') != sha(draft / 'program-domain-review.json'):
        raise ValueError('Revised program review is not bound to this draft')
    decisions = {r['group']: r for r in approvals['entries']}
    data = json.loads((draft / 'data.json').read_text())
    for split in ('dev', 'test'):
        for case in data[split]:
            if case['category'] != 'program':
                continue
            decision = decisions[case['group']]
            if decision['source_id'] != case['source_id'] or decision['supplied_tests'] != case['check']['tests']:
                raise ValueError('Program review identity mismatch')
            if decision['status'] not in ('extras_accepted', 'supplied_only'):
                raise ValueError('Unreviewed or rejected program')
            case['supplied_check'] = copy.deepcopy(case['check'])
            extra = [[r['inputs'], r['expected']] for r in decision['accepted_extras']]
            if any(r not in case['differential_diagnostics'] for r in extra):
                raise ValueError('Approved diagnostic differs from prepared input/output')
            case['check']['tests'] += extra
            case['differential_domain_review'] = {'reviewer': 'claude', 'status': decision['status'], 'note': decision['note']}
            case['grading_scope'] = 'Supplied I/O plus manually domain-checked, independently recomputed extra I/O; still not a full-spec proof'
    protocol = data['protocol']
    protocol['program_extras'] = 'All-tests grade includes Claude-approved domain-valid extras; supplied-only grade also reported'
    protocol['evaluation_order'] = 'Fixed endpoints dev/test first; CUDA 0b stock,A0,A1,A2,S0,S1,S2; historical veto regressions last. Missing required evidence blocks promotion.'
    protocol['precision'] = 'FP32 stored weights, vocabulary heads, teacher and evaluation; BF16 autocast student body during training; TF32 disabled'
    protocol['status'] = 'prepared_and_domain_reviewed_pending_final_freeze'
    archive = OUT / 'draft-before-program-review'
    archive.mkdir(exist_ok=False)
    names = ('data.json', 'protocol.json', 'data-manifest.json', 'generation-gates.json', 'program-domain-review.json')
    for name in names:
        shutil.copy2(OUT / name, archive / name)
    save(data, OUT / 'data.json')
    save(protocol, OUT / 'protocol.json')
    shutil.copy2(draft / 'program-domain-review.json', OUT / 'program-domain-review.json')
    manifest = json.loads((draft / 'data-manifest.json').read_text())
    manifest.update(data_sha256=sha(OUT / 'data.json'), program_audit_sha256=sha(OUT / 'program-domain-review.json'),
                    domain_approvals_sha256=sha(approvals_path))
    save(manifest, OUT / 'data-manifest.json')
    gates = json.loads((draft / 'generation-gates.json').read_text())
    gates['data_sha256'] = sha(OUT / 'data.json')
    save(gates, OUT / 'generation-gates.json')
    overlap = json.loads((OUT / 'new-training-overlap.json').read_text())
    if overlap['training_data_sha256'] != sha(draft / 'data.json') or overlap['hits']:
        raise ValueError('Missing clean revised-training 0b overlap audit')
    old_train = json.loads((draft / 'data.json').read_text())['train']
    if old_train != data['train']:
        raise ValueError('Training changed since overlap check')
    overlap.update(training_data_sha256=sha(OUT / 'data.json'),
                   prior_draft_sha256=sha(draft / 'data.json'), train_array_unchanged=True)
    save(overlap, OUT / 'new-training-overlap.json')


def code_closure():
    queue = ['retention_run', 'retention_worker', 'run_retention_pod', 'retention_freeze', 'tf_metrics',
             'test_retention_run', 'test_retention_prepare', 'test_retention_healing', 'test_retention_pod']
    visited = set()
    while queue:
        name = queue.pop()
        path = ROOT / (name + '.py')
        if path in visited or not path.exists():
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                queue.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                queue.append(node.module.split('.')[0])
    return visited


def candidate():
    files = code_closure() | {ROOT / 'retention_requirements.txt'}
    files |= {OUT / name for name in ('data.json', 'protocol.json', 'data-manifest.json', 'generation-gates.json',
              'program-domain-review.json', 'program-domain-approvals.json', 'science-source-review.json', 'new-training-overlap.json')}
    files |= {ROOT / 'retention_science_eval.json', ROOT / 'results/local-projection-20261025/shared-broad-recipe.pt'}
    files |= {PARENT / 'selected' / name for name in ('adapter.json', 'adapter.safetensors')}
    files |= {ROOT / 'results/tf-metrics-20261027' / name for name in ('manifest.jsonl', 'prepare-report.json')}
    manifest = json.loads((SOURCE / 'download_manifest.json').read_text())
    files |= {SOURCE / name for name in [*manifest['files'], 'model.safetensors.index.json', 'download_manifest.json']}
    for path in files:
        if not path.is_file():
            raise ValueError('Missing input: ' + str(path))
    frozen = {'status': 'pending_peer_review', 'reviewer': None, 'session': 'Manifold',
              'data_sha256': sha(OUT / 'data.json'),
              'files': {str(p.relative_to(ROOT.parent)): sha(p) for p in sorted(files)},
              'authorization': 'One A10080 Secure <=1.79 USD/h, 4h creation-to-deletion, 60GB ephemeral, shared cap20USD; Codex sole launcher',
              'no_auto_promotion': True}
    save(frozen, OUT / 'frozen-candidate.json')
    print(json.dumps({'files': len(files), 'data_sha256': frozen['data_sha256'],
                      'candidate_sha256': sha(OUT / 'frozen-candidate.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrate', type=Path)
    args = parser.parse_args()
    integrate(args.integrate) if args.integrate else candidate()
