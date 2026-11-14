import json, sys, time, torch
sys.path.insert(0, '/home/cyrus/code/model-experiments/manifold-compression')
from transformers import AutoTokenizer, GenerationConfig
from retention_run import configure
from local_projection_probe import SOURCE
arm, threads, batch, new = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
configure(threads)
t = time.monotonic()
if arm == 'stock':
    from local_projection_probe import load_source; model = load_source()
else:
    from allocation_pilot import load_p1; model = load_p1()
load = time.monotonic() - t
tok = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True); tok.padding_side = 'left'
d = json.load(open('/home/cyrus/code/model-experiments/manifold-compression/results/retention-objective-20261027/data.json'))
cases = [c for c in d['dev'] if c['category'] in ('program', 'science', 'retention')][:batch]
prompts = [tok.apply_chat_template(c['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False) for c in cases]
inputs = tok(prompts, add_special_tokens=False, padding=True, return_tensors='pt')
cfg = GenerationConfig(do_sample=False, use_cache=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, max_new_tokens=new, min_new_tokens=new)
with torch.no_grad():
    t = time.monotonic(); out = model.generate(**inputs, generation_config=cfg); gen = time.monotonic() - t
print(json.dumps({'arm': arm, 'threads': threads, 'batch': batch, 'prompt_len': inputs.input_ids.shape[1], 'new': out.shape[1] - inputs.input_ids.shape[1], 'load_s': round(load, 1), 'gen_s': round(gen, 1), 's_per_step': round(gen / new, 3)}))
