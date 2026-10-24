"""Apply corrected target/noise gates to an interrupted stage and record the migration."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from build_local_million import ROOT, SAFE_OPERATIONS, target_quality
from harvest_sources import sha256
from noise_rules import Noiser
from checker import Lexicon


def curate(output):
    config_path=output/'build_config.json'
    previous=json.loads(config_path.read_text(encoding='utf-8'))
    db=sqlite3.connect(output/'pairs.sqlite3')
    db.execute('CREATE INDEX IF NOT EXISTS pairs_origin ON pairs(origin_id)')
    db.commit()
    before=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
    lex,noiser=Lexicon(),Noiser()
    removals=Counter()
    targets=db.execute('SELECT origin_id,MIN(ste),MIN(mode) FROM pairs GROUP BY origin_id').fetchall()
    for index,(origin,text,mode) in enumerate(targets,1):
        reason=target_quality(text,mode,lex,previous['max_unknown'])
        if reason:
            removed=db.execute('DELETE FROM pairs WHERE origin_id=?',(origin,)).rowcount
            db.execute('DELETE FROM target_documents WHERE origin_id=?',(origin,))
            db.execute('UPDATE processed SET reason=? WHERE origin_id=?',('curated_'+reason,origin))
            removals[reason]+=removed
        else:
            valid={(operation,noisy) for operation,noisy in noiser.candidates(text,mode) if operation in SAFE_OPERATIONS}
            for pair_id,src,operation in db.execute('SELECT id,src,noise FROM pairs WHERE origin_id=?',(origin,)).fetchall():
                if (operation,src) not in valid:
                    db.execute('DELETE FROM pairs WHERE id=?',(pair_id,))
                    removals['corrected_noise']+=1
        if index%1000==0:
            db.commit()
        if index%20000==0:
            print(f'curated {index:,}/{len(targets):,} targets',flush=True)
    db.execute('DELETE FROM source_labels')
    from build_pairs import digest
    from text_utils import normalized
    for mode,src,tgt in db.execute('SELECT mode,src,ste FROM pairs').fetchall():
        db.execute('INSERT OR IGNORE INTO source_labels VALUES (?,?,?)',(mode,digest(normalized(src)),digest(normalized(tgt))))
    db.commit()
    after=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
    current=dict(previous)
    current['code_sha256']={p.name:sha256(p) for p in [ROOT/'data/build_local_million.py',ROOT/'data/noise_rules.py',ROOT/'data/text_utils.py',ROOT/'checker/checker.py']}
    history_path=output/'curation_history.json'
    history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    history.append({'pairs_before':before,'pairs_after':after,'removed':dict(removals),
                    'previous_config':previous,'current_config':current,'curator_sha256':sha256(Path(__file__))})
    history_path.write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
    config_path.write_text(json.dumps(current,indent=2)+'\n',encoding='utf-8')
    db.close()
    return {k:history[-1][k] for k in ('pairs_before','pairs_after','removed')}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(curate(args.out),indent=2))
