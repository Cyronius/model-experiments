"""Copy and freeze independent float/ternary branches and genuine Q4/Q8 exports.

Run under Dibs CPU. This prepares experimental artifacts, not a deployment.
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import torch

from export_pair_precision import export
from pair_training_state import atomic_json, digest

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    for precision in ('float', 'ternary'):
        parser.add_argument('--' + precision + '-summary', type=Path, required=True)
        parser.add_argument('--' + precision + '-arm', required=True)
    for precision in ('q4', 'q8'):
        parser.add_argument('--' + precision + '-summary', type=Path)
        parser.add_argument('--' + precision + '-arm')
    parser.add_argument('--q4-group', type=int, choices=[16,32,64], default=16)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Refuse to replace an existing artifact family')
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  frozen_before_new_precision_scores=True, experimental=True,
                  development_scores_used_for_family_selection=True,
                  scope='Independent float and ternary candidates; Q4/Q8 derive only from float. No deployment or packed-kernel speed claim.', models={})
    for precision in ('float', 'ternary'):
        summary = getattr(args, precision + '_summary').resolve()
        arm = getattr(args, precision + '_arm')
        model = json.loads(summary.read_text())['models'][arm]
        selection = model['selection']
        assert selection['precision'] == precision
        source = Path(selection['path'])
        assert digest(source) == selection['sha256']
        target = args.out / ('model-' + precision + '.pt')
        shutil.copy2(source, target)
        assert digest(target) == selection['sha256']
        checkpoint = torch.load(target, map_location='cpu', weights_only=False)
        assert checkpoint['quant'] == (precision == 'ternary')
        query_projection=checkpoint.get('meta',{}).get('query_projection')
        extra=query_projection.numel() if query_projection is not None else 0
        frozen['models'][precision] = dict(selection, path=str(target.resolve()), source_summary=str(summary), source_arm=arm,
                                           source_summary_sha256=digest(summary), bytes=target.stat().st_size,
                                           parameters=sum(t.numel() for t in checkpoint['state'].values())+extra,
                                           additional_query_parameters=extra,query_head=bool(extra))
    for bits in (4, 8):
        name = f'q{bits}'
        summary=getattr(args,name+'_summary');arm=getattr(args,name+'_arm')
        if bool(summary)!=bool(arm):raise ValueError('Both low-bit summary and arm are required for reuse')
        if summary:
            summary=summary.resolve();old=json.loads(summary.read_text())['models'][arm]['selection'];source=Path(old['path'])
            assert old['precision']==name and digest(source)==old['sha256']
            blob=torch.load(source,map_location='cpu',weights_only=False)
            assert blob['metadata']['source_sha256']==frozen['models']['float']['sha256']
            target=args.out/('model-'+name+'.pt');shutil.copy2(source,target)
            frozen['models'][name]=dict(old,path=str(target.resolve()),bytes=target.stat().st_size,
                                        source_summary=str(summary),source_arm=arm,source_summary_sha256=digest(summary))
            continue
        group=args.q4_group if bits==4 else 64
        report = export(args.out / 'model-float.pt', args.out / ('model-' + name + '.pt'), bits,group_size=group)
        frozen['models'][name] = dict(path=str(Path(report['path']).resolve()), sha256=report['sha256'], bytes=report['bytes'],
                                      precision=name, layers=frozen['models']['float']['layers'],
                                      source_float_sha256=report['source_sha256'],
                                      eligible_for_promotion=False,
                                      format=f'Packed W4/W8 dense matrices, int8 activations, BF16 scales, group{group}; embeddings/output/bias/norm remain float32.',
                                      performance='Dense reference loader for quality; native packed-kernel speed unmeasured.')
    atomic_json(args.out / 'frozen.json', frozen)
    print(json.dumps({name: dict(bytes=value['bytes'], sha256=value['sha256']) for name, value in frozen['models'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
