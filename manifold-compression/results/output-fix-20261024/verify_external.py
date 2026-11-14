"""Verify the saved bundles through HF's external and repository loaders."""
import gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from projection_models import load_causal_model
from audit_rotation import compare, save

torch.set_num_threads(2)
for name in ('qwen35', 'smollm2'):
    path = f'models/manifold-compression/output-fix-{name}-1792'
    ids = AutoTokenizer.from_pretrained(path, local_files_only=True)(
        'Why is the sky blue?', return_tensors='pt').input_ids
    external, info = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True, dtype=torch.float32,
        attn_implementation='eager', output_loading_info=True)
    assert all(not info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    with torch.no_grad():
        expected = external(ids, use_cache=False).logits
        del external
        gc.collect()
        local = load_causal_model(path, local_files_only=True, dtype=torch.float32, attn_implementation='eager')
        metrics = compare([expected], [local(ids, use_cache=False).logits])
    assert metrics['numerical_pass']
    result = {'source': path, 'loading': info, 'external_vs_local': metrics}
    save(result, Path(__file__).parent / f'{name}-external-load.json')
    print(name, metrics, flush=True)
    del local, expected
    gc.collect()
