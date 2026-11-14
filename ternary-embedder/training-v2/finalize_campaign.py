"""Write a compact closeout from frozen artifact, quality, billing, and timing evidence."""
import json,os,re
from pathlib import Path
from pair_training_state import digest,atomic_json
ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent
def main():
 family=PROJECT/'models/campaign-20261025';frozen=json.loads((family/'frozen.json').read_text());quality=json.loads((ROOT/'delivery/family-results/summary.json').read_text());cost=json.loads((ROOT/'delivery/cost-closeout.json').read_text())
 assert cost['active_owned_pods']==0 and cost['estimated_spent_usd']<=10
 assert len(cost['jobs'])==len({j['pod'] for j in cost['jobs']})==9
 for job in cost['jobs']:
  assert job['deleted'] and job['deletion_verified_http_status']==404
  assert digest(Path(job['state_path']))==job['state_sha256']
 assert set(frozen['models'])==set(quality['models'])=={'float','ternary','q4','q8'}
 for name,selection in frozen['models'].items():assert digest(Path(selection['path']))==selection['sha256']==quality['models'][name]['selection']['sha256']
 native=json.loads((ROOT/'delivery/native/npu-quality.json').read_text());assert native['checkpoint_sha256']==frozen['models']['ternary']['sha256']
 speed_path=ROOT/'delivery/cpu-speed.json';speed=json.loads(speed_path.read_text()) if speed_path.exists() else None
 audit_path=ROOT/'delivery/latency-dibs.log';audit=audit_path.read_text() if audit_path.exists() else ''
 clean=bool(speed and 'dibs: CLEAN\n' in audit and 'dibs: CLEAN after' in audit and 'NOISY' not in audit)
 confirmation=json.loads((ROOT/'delivery/reserved-confirmation.json').read_text())
 assert confirmation['activation']['candidate']['sha256']==frozen['models']['q8']['sha256']
 assert confirmation['reserved_judgments_opened'] and not confirmation['further_selection_performed']
 text_audit=json.loads((ROOT/'data-first/audit.json').read_text())
 verified_texts={}
 for task in ('FiQA','SCIDOCS'):
  for filename in ('corpus.jsonl','queries.jsonl'):
   relative=f'comparisons/qwen06b/data/{task.lower()}/{filename}'
   expected=text_audit['reference_text_files'][relative]
   assert digest(PROJECT/relative)==expected
   assert all(m['tasks'][task]['source_sha256'][filename]==expected for m in confirmation['models'].values())
   verified_texts[relative]=expected
 from summarize_pair_results import interval
 prior_rows=json.loads((ROOT/'round2/query-rows.json').read_text())
 q8_rows=json.loads((ROOT/'delivery/family-results/q8-rows.json').read_text())
 development_contrasts={name:interval(prior_rows[name],q8_rows) for name in ('bge_micro','minilm')}
 report=dict(scope='Fixed 12-task development subset plus two reserved retrieval confirmations; not full MTEB or independent seed confirmation.',models=quality['models'],cost=cost,native_ternary=native,final_latency_clean=clean,reserved_judgments_opened=True,reserved_confirmation=confirmation,q8_development_contrasts=development_contrasts,original_training_exclusion_texts_verified=verified_texts)
 if clean:report['cpu_speed']=speed;report['native_speed']=json.loads((ROOT/'delivery/native-speed.json').read_text())
 atomic_json(ROOT/'delivery/summary.json',report)
 lines=['# Experimental embedding family','',f"Estimated campaign cost **${cost['estimated_spent_usd']:.2f} of $10**. All nine owned RunPod allocations were deleted; their controllers recorded GET404. No retained paid volumes.",'',
 'Three Transformer layers, width384,12heads, FFN1536, mean pooling,384-dimensional unit vectors, maximum128tokens. Each backbone has17,241,984parameters. The ternary Transformer arithmetic is retained. Float is an independently trained branch; Q4/Q8 derive from the exact selected float checkpoint.','',
 '| Artifact | SciFact | NFCorpus | Retrieval mean | STSBenchmark |12-task mean | File MiB |','|---|---:|---:|---:|---:|---:|---:|']
 for name in ['ternary','float','q4','q8']:
  m=quality['models'][name];s=frozen['models'][name]
  lines.append(f"| [model-{name}.pt](model-{name}.pt) | {m['tasks']['SciFact']['ndcg10']:.3f} | {m['tasks']['NFCorpus']['ndcg10']:.3f} | {m['core_retrieval']:.3f} | {m['stsbenchmark']:.3f} | {m['mteb_12_mean']:.3f} | {Path(s['path']).stat().st_size/2**20:.2f} |")
 if not all(confirmation['beats_mean'].values()):
  ahead=[{'minilm':'MiniLM','bge_micro':'BGE-micro'}[name] for name,won in confirmation['beats_mean'].items() if not won]
  outcome='**Quality improved, but the goal of beating both references broadly remains unmet.** Q8 leads the development point estimates, and trails '+ ' and '.join(ahead)+' on the reserved two-task mean. The complete confirmation and intervals are below.'
 elif not all(confirmation['beats_each_task'].values()):
  outcome='**Q8 improves the reserved two-task mean over both references, with mixed individual-task results.** This is a limited confirmation, not broad embedding-model superiority; see the intervals below.'
 else:
  outcome='**Q8 exceeds both references on both reserved retrieval point estimates.** This limited confirmation does not replace full-MTEB evaluation or training-seed replication; see the intervals below.'
 lines[4:4]=[outcome,'']
 lines += ['| Original two-layer ternary |54.853|29.403|42.128|81.583|70.156| — |','| bge-micro-v2 |61.330|31.098|46.214|82.497|70.904| — |','| MiniLM-L6-v2 |62.386|31.453|46.920|82.032|71.481| — |','| Qwen0.6B, prompted |67.732|35.503|51.618|91.199| unmeasured | — |','',
 'Retrieval is the equal-task nDCG@10 mean over full SciFact/NFCorpus corpora. All quality uses a128-token cap. Model selection used observed development results across many experiments. The12-task mean is not full MTEB. Q8 exceeds both references on each core task, STSBenchmark, and the 12-task mean, as point estimates. Full-pipeline seed replication is outstanding. The original project default is not overwritten.','',
 'Q4 uses real four-bit dense codes with group16 BF16 scales and int8 activations; Q8 uses eight-bit dense codes with group64 scales and int8 activations. Embeddings, output projection, biases, and norms remain float32. These packed files have been quality-tested through the unpacked reference loader; optimized packed CPU/NPU kernels for Q4/Q8 are not provided. Ternary checkpoints store floating master/reconstruction tensors, while the native engine packs their three-valued dense weights at load time.','',
 f"The selected ternary model was checked on the real NPU against CPU vectors at both32/128-token buckets. Native full-corpus retrieval mean is **{native['mean_core_retrieval']:.3f}**, versus CPU **{quality['models']['ternary']['core_retrieval']:.3f}**.",'']
 lines += ['Development Q8-minus-reference retrieval differences (paired query intervals, without correction for repeated model selection):','']
 for name,c in development_contrasts.items():
  lo,hi=c['paired_query_bootstrap_95_ci'];lines.append(f"- {name}: **{c['mean_delta_ndcg10']:+.3f}** points, 95% interval **[{lo:+.3f}, {hi:+.3f}]**.")
 lines += ['', '## Reserved retrieval confirmation','', 'After Q8 passed the development gates, its exact file hash was frozen before opening FiQA and SCIDOCS test judgments. Only that candidate and the two pinned reference models were tested. No further training or selection used these judgments; these datasets are now opened and cannot serve as future reserved confirmation. Public historical pretraining exposure is unknown.','', '| Model | FiQA nDCG@10 | SCIDOCS nDCG@10 | Equal-task mean |','|---|---:|---:|---:|']
 for name in ('q8','bge_micro','minilm'):
  m=confirmation['models'][name];lines.append(f"| {name} | {m['tasks']['FiQA']['ndcg10']:.3f} | {m['tasks']['SCIDOCS']['ndcg10']:.3f} | {m['mean_retrieval']:.3f} |")
 lines += ['', 'Paired query-bootstrap differences, Q8 minus reference (5,000 resamples; seed 20261025):','']
 for name,c in confirmation['q8_minus_references'].items():
  lo,hi=c['paired_query_bootstrap_95_ci'];lines.append(f"- {name}: **{c['mean_delta_ndcg10']:+.3f}** points, 95% interval **[{lo:+.3f}, {hi:+.3f}]**. Higher on both individual tasks: {confirmation['beats_each_task'][name]}.")
 lines += ['', 'Intervals quantify query sampling within these two fixed datasets. They do not measure general domain superiority, training-seed variation, or development-selection uncertainty. See the [protocol](../../training-v2/delivery/RESERVED_PROTOCOL.md) and [activation receipt](../../training-v2/delivery/reserved/activation.json).','']
 if clean:
  lines += ['## Loaded-model latency','', 'Same two pinned physical CPU cores, batch1, tokenization included, fixed padding, minimum0.75seconds of warmup. Initial and final Dibs checks were CLEAN. Qwen uses BF16; the small models use float32 CPU inference. Ternary CPU timing uses cached effective weights and dense kernels; NPU timing uses actual native ternary kernels.','', '| CPU model |32-token cap, ms |128-token cap, ms |','|---|---:|---:|']
  for name in ['float','ternary','bge_micro','minilm','qwen_0_6b_document','qwen_0_6b_query']:
   t=speed['timings'][name];lines.append(f"| {name} | {t['32']['median_ms']:.3f} | {t['128']['median_ms']:.3f} |")
  n=json.loads((ROOT/'delivery/native-speed.json').read_text());lines += ['', 'Native NPU dispatches one actual text, with unused compiled slots padded. These are single-text latencies, not batch throughput divided by four.']
  for row in n['runs']:lines.append(f"At the{row['cap']}-token cap: **{row['median_text_to_vector_ms']:.3f}ms**.")
 else:lines += ['Final checkpoint timing is pending the shared-machine measurement slot. Earlier matched architecture timings are in training-v2/science-mix/RESULTS.md; they are not assigned as exact measurements of these final weights.','']
 lines += ['', '## Use in this repository','', 'Use the existing project environments and tokenizer. From the repository root:','', '```python','import sys','sys.path.insert(0, "ternary-embedder/training-v2")','from evaluate_round2_mteb import Encoder','encoder = Encoder("ternary-embedder/models/campaign-20261025/model-ternary.pt")','documents = encoder.encode_documents(["A passage to index."])','queries = encoder.encode_queries(["A search query."])','# For packed variants: Encoder(path, precision="q4") or precision="q8".','```','',
 'The winning models use the common query/document head. The separately tested optional query-head variants selected the unchanged parent and are not part of this family. Native inference accepts model-ternary.pt through FusedEncoder(cap, checkpoint=...). The prebuilt three-layer32/128 graphs and local NPU runtime are required.','',
 '## Findings','', 'Pretrained-weight retention produced the strongest balanced gains. Fresh science helped float retrieval more clearly than ternary. The tested clone-and-append layer, unjudged BM25 negatives, additional continuation, learned scales, and output/query projection changes did not close the remaining reference gap. Human entailment/contradiction ranking raised broader quality but hurt retrieval; logical contradiction is not a human judgment of passage irrelevance. More epochs alone did not solve the problem.','',
 'The current direction is better relevance supervision with stronger protection of pretrained behavior. This is an inference from this campaign, not a proven optimal recipe. Q8 leads the development point estimates, while the confirmation table above determines how well that carries to the two reserved domains. Ternary still trails the references on development retrieval. The small Q8 gain over float may be quantization sensitivity on these samples. See [experiment interpretation](../../training-v2/delivery/EXPERIMENT_NOTES.md).','',
 'Evidence: [frozen file hashes](frozen.json), [delivery summary](../../training-v2/delivery/summary.json), [billing closeout](../../training-v2/delivery/cost-closeout.json), [all development comparisons](../../training-v2/campaign/COMPARISON.md), [native quality](../../training-v2/delivery/native/npu-quality.json), [timing audit](../../training-v2/delivery/latency-dibs.log).','',
 'Full optimizer, source-sampler, classifier where used, and RNG backups remain under training-v2/work/*-run/backups with SHA-verified receipts. The scale continuation recovered a saved step68 checkpoint and completed its original training budget. No credentials or private SSH keys are included in this model directory.','']
 text='\n'.join(lines)
 for a,b in [('width384','width 384'),('12heads','12 heads'),('maximum128','maximum 128'),('has17,','has 17,'),('a128','a 128'),('both32','both 32'),('group16','group 16'),('group64','group 64'),('the32','the 32'),('the128','the 128'),('three-layer32','three-layer 32'),('saved step68','saved step 68')]:text=text.replace(a,b)
 for a,b in [('384,12','384, 12'),('FFN1536','FFN 1536'),('pooling,384','pooling, 384'),('128tokens','128 tokens'),('17,241,984parameters','17,241,984 parameters'),('The12-task','The 12-task'),('the12-task','the 12-task'),('|12-task','| 12-task'),('batch1','batch 1'),('minimum0.75seconds','minimum 0.75 seconds'),('both32/128','both 32/128')]:text=text.replace(a,b)
 text=re.sub(r'(?<=\d)ms\b',' ms',text)
 (family/'README.md').write_text(text)
 def delivery_link(match):
  target=match.group(1)
  return ']('+os.path.relpath(family/target,ROOT/'delivery')+')' if not target.startswith(('https:','http:','#')) else match.group(0)
 (ROOT/'delivery/RESULTS.md').write_text(re.sub(r'\]\(([^)]+)\)',delivery_link,text))
 print('Wrote delivery summary and model README; final timing CLEAN:',clean,flush=True)
if __name__=='__main__':main()
