import json,torch
from pathlib import Path
from scale_probe import load_source,compare
from scale_data import load_tokenizer
from transformers import GenerationConfig
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
model,_=load_source('models/qwen2','cuda:0')
tok=load_tokenizer('models/qwen2');tok.padding_side='left';tok.pad_token=tok.eos_token
cases=json.loads(Path('data.json').read_text())['generation_eval'][4:8]
texts=[tok.apply_chat_template(c['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False) for c in cases]
config=GenerationConfig(do_sample=False,use_cache=True,max_new_tokens=32,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,repetition_penalty=1.)
result={}
with torch.no_grad():
 batch=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt').to('cuda')
 kwargs=model.prepare_inputs_for_generation(batch.input_ids,attention_mask=batch.attention_mask,use_cache=True,logits_to_keep=1)
 batched=model(**kwargs).logits.float().cpu()
 out=model.generate(**batch,generation_config=config)[:,batch.input_ids.shape[1]:]
 result['batch_answers']={c['id']:tok.decode(s,skip_special_tokens=True) for c,s in zip(cases,out)}
 singles=[]
 for c,text in zip(cases,texts):
  row=tok(text,add_special_tokens=False,return_tensors='pt').to('cuda')
  kw=model.prepare_inputs_for_generation(row.input_ids,attention_mask=row.attention_mask,use_cache=True,logits_to_keep=1)
  singles.append(model(**kw).logits.float().cpu())
  gen=model.generate(**row,generation_config=config)[:,row.input_ids.shape[1]:]
  result.setdefault('single_answers',{})[c['id']]=tok.decode(gen[0],skip_special_tokens=True)
 result['prefill_batch_vs_single']=compare([torch.cat(singles,dim=0)],[batched])
Path('results/qwen2-batching-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
