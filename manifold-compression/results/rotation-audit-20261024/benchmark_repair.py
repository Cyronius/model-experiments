import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.utils.checkpoint import checkpoint

torch.set_num_threads(12);torch.set_num_interop_threads(1)
student_path='models/manifold-compression/distill-08b/continuation-500m/ckpt_500M'
student,info=AutoModelForCausalLM.from_pretrained(student_path,dtype=torch.float32,local_files_only=True,attn_implementation='eager',output_loading_info=True)
assert not info['missing_keys'] and not info['mismatched_keys'],info
teacher=AutoModelForCausalLM.from_pretrained('models/manifold-audit/Qwen3.5-2B',dtype=torch.float32,local_files_only=True,attn_implementation='eager').eval().requires_grad_(False)
student.train().requires_grad_(True)
tok=AutoTokenizer.from_pretrained(student_path,local_files_only=True)
text='Explain how to check whether a Python function correctly computes the mean of a list. Include empty lists, negative numbers, and fractions. '
tokens=tok(text,add_special_tokens=False).input_ids
ids=torch.tensor([(tokens*100)[:512]])
opt=torch.optim.AdamW(student.parameters(),lr=1e-6)
results={'device':'cpu','threads':12,'sequence_length':512,'batch_rows':1,'synthetic_text':text,'purpose':'Timing only; updated weights are discarded. No quality claim.', 'precision_runs':{}}
for amp in [False,True]:
 timings=[]
 for step in range(3):
  start=time.perf_counter();opt.zero_grad(set_to_none=True)
  with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp): ht=teacher.model(input_ids=ids,use_cache=False).last_hidden_state[:,:-1]
  with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp): hs=student.model(input_ids=ids,use_cache=False).last_hidden_state[:,:-1]
  def kl(x,y):
   with torch.no_grad():lt=teacher.lm_head(y).float().log_softmax(-1)
   ls=student.lm_head(x).float().log_softmax(-1)
   return (lt.exp()*(lt-ls)).sum()
  loss=0
  for j in range(0,511,32):
   with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):loss=loss+checkpoint(kl,hs[:,j:j+32],ht[:,j:j+32],use_reentrant=False)
  (loss/511).backward();torch.nn.utils.clip_grad_norm_(student.parameters(),1.0);opt.step()
  elapsed=time.perf_counter()-start;timings.append(elapsed)
  print('bf16' if amp else 'fp32',step,elapsed,'seconds',512/elapsed,'tokens/s',flush=True)
 results['precision_runs']['bf16_autocast' if amp else 'fp32']={'step_seconds':timings,'tokens_per_second_after_warmup':1024/sum(timings[1:])}
Path('manifold-compression/results/rotation-audit-20261024/cpu-repair-throughput.json').write_text(json.dumps(results,indent=2)+'\n')
