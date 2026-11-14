"""Write a reviewable data/job report and grounded-question generation queue."""
import collections
import hashlib
import heapq
import json
from pathlib import Path
from broad_data_fetch import atomic,digest
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent


def main():
    ready=ROOT/'work/broad-data/ready';audit=json.loads((ready/'audit.json').read_text());plan=json.loads((ROOT/'broad-data/runpod-job.json').read_text())
    package=json.loads((ROOT/'work/broad-data-run/package.json').read_text());preflight=json.loads((ROOT/'broad-data/preflight.json').read_text())
    assert package['plan_sha256']==digest(ROOT/'broad-data/runpod-job.json')==preflight['plan_sha256']
    # Queue public, already-audited TRAIN passages only. No generator/API is invoked.
    selected=[];seen=set()
    eligible=('msmarco','stack-exchange-preferences','gooaq','natural-questions','squad','trivia','hotpot')
    for line in (ready/'train.jsonl').open():
        r=json.loads(line)
        if not any(name in r['source'] for name in eligible):continue
        key=r['positive_visible_key']
        if key in seen:continue
        priority=int(hashlib.blake2b(b'lcqa-queue20261025'+bytes.fromhex(key),digest_size=8).hexdigest(),16)
        if len(selected)<5000:
            heapq.heappush(selected,(-priority,key,r['positive'],r['source'],r['origin_id']));seen.add(key)
        elif priority < -selected[0][0]:
            old=heapq.heapreplace(selected,(-priority,key,r['positive'],r['source'],r['origin_id']));seen.remove(old[1]);seen.add(key)
    queue=ROOT/'work/broad-data/lcqa-generation-requests.jsonl'
    with queue.open('w') as output:
        for _,key,passage,source,origin in sorted(selected,reverse=True):
            prompt=('Create one retrieval-training example from the PASSAGE. Write a concise standalone search question directly answered by it. '
                    'Return JSON with exactly question and answer_quote. The quote must occur verbatim in the passage. '
                    'Do not refer to this passage, earlier messages or external sources. Use a short factual question, not a request to write code or an essay.\nPASSAGE:\n'+passage)
            output.write(json.dumps(dict(id=key,split='train',source=source,origin_id=origin,passage=passage,prompt=prompt),ensure_ascii=False)+'\n')
    atomic(ROOT/'broad-data/lcqa-generation-queue.json',dict(rows=len(selected),path=str(queue.relative_to(PROJECT)),sha256=digest(queue),
                status='Prepared only; generation not run and not part of the proposed four-arm training charge.',
                validator='broad_lcqa_pilot.generated_pair requires exact evidence quote and complete inputs within128tokens.',
                provenance='Only training passages from the globally audited corpus; no held passages or test judgments.'))
    lines=['# Broader retrieval data and proposed controlled training','',
           f"Prepared **{audit['train_rows']:,} unique training pairs**, **{audit['validation_rows']:,} validation rows**, and **{audit['all_new_held_rows']:,} complete new held rows**. The model stays at three layers,384dimensions and17,241,984parameters. No new paid training has been authorized or launched.",'',
           '| Source | Retained training pairs |','|---|---:|']
    for source,s in sorted(audit['by_source'].items(),key=lambda item:-item[1].get('train',0)):
        if s.get('train'):lines.append(f"| {source} | {s['train']:,} |")
    lines += ['', 'Counts are globally deduplicated model-visible pairs, rather than raw downloaded rows or repeated epochs. The original clean corpus supplies retention examples. Broad QA adds new sources and recovers long positives only with original answer evidence. SQuAD uses the annotated answer occurrence; NQ/TriviaQA use original TRAIN answer aliases matched inside the original positive passage. Alias matching is weaker than independently verified relevance.','',
              'All new accepted inputs fit128tokens without blind truncation. Known benchmark texts and protected internal validation texts are excluded using exact tokenizer identities, normalized text and punctuation-normalized identities. Original source groups stay together; components containing known prior fine-tuning exposures stay in train. An independent final-file pass verified zero train/validation input overlap. These exclusions protect this continuation; they cannot undo benchmark-text exposure already present in its starting checkpoints. Historical pretraining and semantic near-duplicate exposure remain unknown.','',
              'K2-Horizon contributes a strict structured QA pilot, including reusable curated examples. The unstructured instruction-following sample was rejected because prompt/answer boundaries could not be established. The public LCQA examples validate the extraction path but remain a small separate pilot. A5,000-passage generation queue and evidence validator are ready; no generation service was called.','',
              'The final cohort caps synthetic PAQ at1M eligible pairs. Stack Exchange samples communities rather than letting the hundreds of StackOverflow shards dominate; only accepted or positively voted answers are retained, with original URLs and author attribution. Supplied mined negatives and lower-voted answers are not labeled irrelevant.','',
              'Yahoo Answers is excluded from this first mix after spot checks found nonresponsive joke answers. Generic Horizon refusals are filtered. New candidates receive pinned, checksum-verified encoding repair and safe conversion of serialized string lists into text; answer evidence must survive cleanup. Legacy retention strings are preserved unchanged. Source-level rejection and cleanup counts are recorded in the audit. These checks reduce known noise but do not certify every answer as relevant.','',
              'Two fresh reserved label sets are Touche2020 and TREC-COVID. Only their corpus/query texts were fetched for exclusions; qrels remain unopened. FiQA,SCIDOCS and the previous12-task subset are already observed development data. These checks do not establish full-MTEB superiority.','',
              'The proposed comparison has four arms: ternary broad/repeated-old and float broad/repeated-old. Each starts from the exact final checkpoint of the previous campaign and sees one complete epoch with the same total pair count, seed71, batch1024, microbatch128 and LR1e-5. The repeat control preserves the old mixture proportions to rounding. Source composition and token lengths differ; this compares whole data recipes, not isolated novelty at identical source weights.','',
              'Best-checkpoint selection uses held-source macro nDCG with a2-point per-source retention floor, including unchanged step0. All four arms freeze before external development scoring. Reserved judgments open only for one frozen candidate that exceeds both references on every predeclared development retrieval task, STSBenchmark and the fixed12-task mean. No training or selection follows reserved scoring.','',
              f"CPU-only train/reload preflights passed for both precisions. The SHA-verified upload package is **{package['bytes']/2**30:.2f} GiB**. Smokes verify plumbing and recovery; they are not new quality results.",'',
              'Proposed resource: one RunPod Secure RTX4090 24GB, at least32GB host RAM,80GB temporary disk. Listed GPU rate is$0.89/hour; launch ceiling$0.90/hour. Expected4–5.5hours, about$3.61–$4.96 including temporary disk. **New total cap$6, maximum6hours**, with work stopped earlier and an independent deletion watchdog. No retained paid volumes. This is separate from the closed$9.58/$10 campaign.','',
              'Full model/optimizer/RNG/source-sampler checkpoints save every100steps or120seconds. Local SHA-verified backup attempts run every60seconds. Two latest complete resume states are retained. Every allocated pod must be deleted and GET404 verified before closing the job.','',
              'Review: [job plan](runpod-job.json), [data audit](data-audit.json), [preflight](preflight.json), [source locks](sources.lock.json), [extra source locks](extra-sources.lock.json), [PAQ cohorts](paq-cohorts.lock.json), [new evaluation texts](evaluation.lock.json), [quarantined samples](quarantine.json), [LCQA queue](lcqa-generation-queue.json).','',
              'Source terms are recorded with pinned revisions. H4 Stack Exchange attribution is preserved in work/broad-data/ready/attribution.jsonl. Several derivative cards omit license metadata; this preparation is not a blanket license grant or a commercial-release clearance.','']
    text='\n'.join(lines)
    for a,b in [('three layers,384','three layers, 384'),('and17,','and 17,'),('17,241,984parameters','17,241,984 parameters'),('fit128','fit 128'),('A5,000','A 5,000'),('at1M','at 1M'),('FiQA,SCIDOCS','FiQA, SCIDOCS'),('previous12','previous 12'),('seed71','seed 71'),('batch1024','batch 1024'),('microbatch128','microbatch 128'),('LR1e','LR 1e'),('a2-point','a 2-point'),('step0','step 0'),('fixed12','fixed 12'),('RTX4090','RTX 4090'),('least32','least 32'),('RAM,80','RAM, 80'),('is$','is $'),('ceiling$','ceiling $'),('Expected4','Expected 4'),('cap$','cap $'),('maximum6','maximum 6'),('closed$','closed $'),('every100steps','every 100 steps'),('or120seconds','or 120 seconds'),('every60seconds','every 60 seconds')]:text=text.replace(a,b)
    for a,b in [('384dimensions','384 dimensions'),('128tokens','128 tokens'),('1M eligible','1M eligible'),('5.5hours','5.5 hours'),('6hours','6 hours'),('24GB','24 GB'),('32GB','32 GB'),('80GB','80 GB'),('about$','about $')]:text=text.replace(a,b)
    (ROOT/'broad-data/README.md').write_text(text)
    atomic(ROOT/'broad-data/preparation-complete.json',dict(data_audit_sha256=digest(ready/'audit.json'),job_plan_sha256=digest(ROOT/'broad-data/runpod-job.json'),package_sha256=package['sha256'],new_spending_authorized=False,paid_pod_active=False))
    print('Wrote corpus/job review and grounded-generation queue.',flush=True)


if __name__=='__main__':main()
