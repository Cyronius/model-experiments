"""Diagnostic triggered by increased FP32 held-out KL after Qwen healing."""
import json,gc
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from audit_rotation import compare,save
from healing_pilot import load_checked,encoded_rows,source_student,healed_path

root=Path(__file__).parent
data=json.loads((root/'data.json').read_text())
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained('models/manifold-audit/Qwen3.5-2B',local_files_only=True)
cache=torch.load('target/manifold-audit/healing-cache/qwen35.pt',mmap=True,weights_only=True)
head_bf16=cache['head'].to(torch.bfloat16)
train=encoded_rows(data,tok,'train')
val=encoded_rows(data,tok,'validation')
rows=[('train',r) for r in (train[0],train[1],train[-2],train[-1])]+[('validation',r) for r in val[:2]]
result={'purpose':'Separate precision sensitivity from fitting/generalization; diagnostic subset only','models':{}}
with torch.no_grad():
 for label,path in [('before',source_student('qwen35')),('after',healed_path('qwen35','corrected'))]:
  model=load_checked(path)
  output=[]
  for split,row in rows:
   ids=torch.tensor([row['input_ids']])
   select=torch.arange(0,len(row['positions']),8)
   positions=torch.tensor(row['positions'])[select]
   hs=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,positions]
   exact=model.lm_head(hs).float()
   teacher=F.linear(cache['rows'][row['id']][select],cache['head']).float()
   with torch.autocast('cpu',dtype=torch.bfloat16):
    hs=model.model(input_ids=ids,use_cache=False).last_hidden_state[0,positions]
    amp=model.lm_head(hs).float()
    teacher_amp=F.linear(cache['rows'][row['id']][select],head_bf16).float()
   output.append({'split':split,'id':row['id'],'fp32_vs_autocast':compare([exact],[amp]),
                  'fp32_teacher_vs_student':compare([teacher],[exact]),
                  'training_precision_teacher_vs_student':compare([teacher_amp],[amp])})
  result['models'][label]=output
  del model
  gc.collect()
  save(result,root/'precision-diagnostic.json')
for label,rows in result['models'].items():
 for r in rows:
  print(label,r['split'],r['id'],{k:round(r[k]['mean_kl'],6) for k in ('fp32_vs_autocast','fp32_teacher_vs_student','training_precision_teacher_vs_student')})
