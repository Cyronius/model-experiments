"""Independently verify a deterministic filtered export against every parent record."""
import argparse
import hashlib
import json
from pathlib import Path
from check_quality import input_paths, records, sha256
from quality_gates import inspect_pair


def payload(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate(parent, output):
    summary = json.loads((output/'summary.json').read_text())
    report = json.loads((output/'quality-report.json').read_text())
    if sha256(parent/'summary.json') != report['parent_summary_sha256'] or sha256(output/'quality-report.json') != summary['quality_report_sha256']:
        raise ValueError('parent/report fingerprint differs')
    for name, expected in report['code_sha256'].items():
        if sha256(Path(__file__).parent/name) != expected:
            raise ValueError('quality implementation fingerprint differs')
    expected = {}
    for path, _ in input_paths(output):
        for row in records(path):
            if row['id'] in expected:
                raise ValueError('duplicate retained ID')
            expected[row['id']] = payload(row)
    retained = len(expected)
    rejected = {}
    for row in records(output/'rejected.jsonl'):
        if row['id'] in rejected or row['reason'] != 'deterministic_quality':
            raise ValueError('invalid/duplicate rejection')
        rejected[row['id']] = set(row['quality_rules'])
    scanned = 0
    for path, _ in input_paths(parent):
        for row in records(path):
            scanned += 1
            hard = {i['rule'] for i in inspect_pair(row) if i['action']=='exclude'}
            if hard:
                if row['id'] in expected or rejected.pop(row['id'],None) != hard:
                    raise ValueError('rejection differs from deterministic rules')
            elif expected.pop(row['id'],None) != payload(row):
                raise ValueError('retained payload missing or altered')
            if scanned % 100000 == 0:
                print(f'verified parent records: {scanned:,}',flush=True)
    if expected or rejected or scanned != report['scanned'] or retained != report['kept'] or retained != summary['pairs']:
        raise ValueError('filter accounting differs')
    result = {'passed':True,'parent_records':scanned,'retained_records':retained,
              'excluded_records':scanned-retained,'retained_payloads_unchanged':True,
              'scope':'exact subset, all deterministic decisions, original metadata/splits/groups unchanged',
              'summary_sha256':sha256(output/'summary.json'),'code_sha256':sha256(Path(__file__))}
    (output/'quality-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.parent,args.out),indent=2))
