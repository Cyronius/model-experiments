"""Exploratory Mistral readout ablation; primary scale-probe arms stay frozen.

Same BF16 body, basis, scales and parameter count as the 75%-retained arm.
Replace its final head/matrix composition by c W D P and an identity matrix.
The identity is retained to keep the parameter accounting matched. This is
only valid without duplicating vocabulary weights for an untied source head.
"""
import json
from pathlib import Path

import torch

from audit_rotation import norm_gain, cache_check
from projection_models import project_residual_width, install_output_matrix
from scale_data import load_tokenizer
from scale_probe import load_source, validation_rows, evaluate_logits, generate, save


@torch.no_grad()
def main(root=Path('/workspace/scale')):
    name = 'mistral24'
    primary = root / 'results' / name
    out = root / 'results' / 'mistral24-exact-readout'
    out.mkdir(exist_ok=True)
    data = json.loads((root / 'data.json').read_text())
    run = json.loads((primary / 'result.json').read_text())
    arm = run['plan']['arms'][0]
    k = arm['width']
    source = root / 'models' / name
    tok = load_tokenizer(source)
    model, _ = load_source(source, 'cuda:0')
    assert model.model.embed_tokens.weight is not model.lm_head.weight
    p = torch.load(primary / 'basis.pt', weights_only=True)[:, :k].contiguous().cuda()
    scales = json.loads((primary / 'rms-scales.json').read_text())[str(k)]
    gain = norm_gain(model.model.norm).float()
    exact = model.lm_head.weight.new_empty((model.config.vocab_size, k))
    for start in range(0, model.config.vocab_size, 1024):
        w = model.lm_head.weight[start:start + 1024].float()
        exact[start:start + len(w)] = ((w * gain) @ p.float()) * scales[-1]
    project_residual_width(model, p, scales, chunk_size=1024)
    model.lm_head.weight.copy_(exact)
    install_output_matrix(model, torch.eye(k, device='cuda', dtype=torch.bfloat16))
    del exact, p, gain
    torch.cuda.empty_cache()
    count = sum(p.numel() for p in model.parameters())
    assert count == arm['parameters']
    rows = validation_rows(data, tok)
    result = {'status': 'running', 'parameters': count, 'primary_data_sha256': run['data_sha256'],
              'scope': 'Post-hoc exact-readout ablation at 75% retention; 64 frozen short cases, no long-form cases, no healing.',
              'cache': cache_check(model, torch.tensor([rows[0]['input_ids'][:128]], device='cuda'))}
    if result['cache']['max_token_kl'] > .05:
        raise RuntimeError('Exact-readout BF16 cache check failed')
    result['validation'] = evaluate_logits(model, rows, primary / 'teacher_logits')
    save(result, out / 'result.json')
    cases = [c for c in data['generation_eval'] if c['check'] is not None]
    assert len(cases) == 64
    result['generation'] = generate(model, tok, cases, out / 'generation.json', 4)
    result['status'] = 'complete'
    save(result, out / 'result.json')


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main()
