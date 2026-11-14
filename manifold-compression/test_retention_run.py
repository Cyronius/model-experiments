import copy
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from retention_run import batch_terms, exact_terms, row_terms, runtime_smoke, update


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(11, 5)

    def forward(self, input_ids, use_cache=False, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()
        self.lm_head = nn.Linear(5, 11, bias=False)


def row(replay):
    return dict(input_ids=[1, 2, 3, 4], positions=[1, 2], targets=[3, 4], replay=replay)


def test_exact_forward_kl_value_and_student_gradient():
    torch.manual_seed(21)
    s = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    t = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([1, 3, 4, 0, 9])
    ce, kl = exact_terms(s, labels, t)
    expected = F.kl_div(s.log_softmax(-1), t.detach().softmax(-1), reduction='sum')
    torch.testing.assert_close(kl, expected)
    (ce + kl).backward()
    expected_grad = s.detach().softmax(-1) * 2 - t.detach().softmax(-1) - F.one_hot(labels, 11)
    torch.testing.assert_close(s.grad, expected_grad)
    assert t.grad is None


def test_chunks_and_replay_mask_preserve_loss_and_gradient():
    torch.manual_seed(22)
    m, teacher = Tiny().double(), Tiny().double().requires_grad_(False)
    a = row_terms(m, row(True), teacher, chunk=1, autocast=False)
    a.sum().backward()
    grads = [p.grad.clone() for p in m.parameters()]
    m.zero_grad(set_to_none=True)
    b = row_terms(m, row(True), teacher, chunk=64, autocast=False)
    b.sum().backward()
    torch.testing.assert_close(a, b)
    for p, expected in zip(m.parameters(), grads):
        torch.testing.assert_close(p.grad, expected)
    ce_only = row_terms(m, row(False), teacher, autocast=False)
    assert ce_only[1] == 0
    torch.testing.assert_close(ce_only[0], a[0])
    assert all(p.grad is None for p in teacher.parameters())


def test_update_normalizes_by_all_answer_tokens():
    torch.manual_seed(23)
    m, teacher = Tiny(), Tiny().requires_grad_(False)
    expected = copy.deepcopy(m)
    rows = [row(True), row(False), dict(input_ids=[2, 3], positions=[0], targets=[3], replay=False)]
    opt = torch.optim.SGD(m.parameters(), lr=.01)
    ref = torch.optim.SGD(expected.parameters(), lr=.01)
    loss = sum(row_terms(expected, r, teacher, autocast=False).sum() for r in rows) / 5
    loss.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
    ref.step()
    report = update(m, opt, rows, teacher, .01, autocast=False)
    for p, q in zip(m.parameters(), expected.parameters()):
        torch.testing.assert_close(p, q)
    assert report['answer_tokens'] == 5 and report['replay_tokens'] == 2
    assert abs(report['objective_per_answer_token'] - loss.item()) < 1e-6


def test_hybrid_qwen_backward_uses_actual_bf16_body_fp32_head():
    result = runtime_smoke('cpu')
    assert result['after_ce_kl'][0] < result['before_ce_kl'][0]
    assert result['before_ce_kl'][1] < .01


def test_batched_replay_alignment_and_teacher_only_sees_replay_rows():
    torch.manual_seed(24)
    m, teacher = Tiny().double(), Tiny().double().requires_grad_(False)
    rows = [row(False), dict(input_ids=[8, 2], positions=[0], targets=[2], replay=True), row(True)]
    seen = []
    hook = teacher.model.register_forward_pre_hook(lambda mod, args, kw: seen.append(kw['input_ids'].shape[0]), with_kwargs=True)
    terms = batch_terms(m, rows, teacher, chunk=2, autocast=False)
    hook.remove()
    assert seen == [2]
    expected = sum(row_terms(m, r, teacher, chunk=2, autocast=False) for r in rows)
    torch.testing.assert_close(terms, expected, atol=1e-6, rtol=1e-6)


def hybrid_batch_equivalence(device='cpu'):
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    config = Qwen3_5TextConfig(vocab_size=257, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=128, linear_value_head_dim=128,
        layer_types=['linear_attention', 'full_attention'], tie_word_embeddings=False,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 1., 'mrope_section': [8, 8, 16]})
    config._attn_implementation = 'eager'
    torch.manual_seed(25)
    m = Qwen3_5ForCausalLM(config).float().to(device).eval()
    teacher = copy.deepcopy(m).requires_grad_(False)
    from correctness_heal import install_lora
    install_lora(m, 4, 8)
    # Exercise gradients of BOTH factors, as in a continued trained adapter.
    with torch.no_grad():
        for name, parameter in m.named_parameters():
            if parameter.requires_grad and name.endswith('.b'):
                parameter.normal_(std=.001)
    rows = [dict(input_ids=list(range(1, n + 1)), positions=list(range(n - 5, n - 1)),
                 targets=list(range(n - 3, n + 1)), replay=(n != 17)) for n in (17, 9, 25)]
    report = {'device': device, 'modes': {}, 'trainable': 'LoRA A/B, trained nonzero B'}
    for autocast in (False, True):
        m.zero_grad(set_to_none=True)
        separate = sum(row_terms(m, r, teacher, autocast=autocast) for r in rows)
        separate.sum().backward()
        expected = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
        m.zero_grad(set_to_none=True)
        together = batch_terms(m, rows, teacher, autocast=autocast)
        together.sum().backward()
        torch.testing.assert_close(together, separate, atol=.02 if autocast else 2e-5, rtol=1e-4)
        norms = []
        for n, p in m.named_parameters():
            if n in expected:
                if not autocast:
                    torch.testing.assert_close(p.grad, expected[n], atol=2e-5, rtol=2e-4,
                                               msg=lambda msg: n + ': ' + msg)
                elif expected[n].norm() > 1e-7:
                    # BF16 reductions change rounding with batch shape. Measure
                    # whole-gradient error; relative error at near-zero entries
                    # has no useful interpretation.
                    relative = (p.grad - expected[n]).norm() / expected[n].norm()
                    cosine = F.cosine_similarity(p.grad.flatten(), expected[n].flatten(), dim=0)
                    assert relative < .02 and cosine > .999, (n, relative.item(), cosine.item())
                    norms.append((relative.item(), cosine.item()))
                else:
                    torch.testing.assert_close(p.grad, expected[n], atol=1e-7, rtol=0)
        report['modes']['bf16_body' if autocast else 'fp32'] = {
            'loss_max_abs_difference': (together - separate).abs().max().item(),
            'grad_max_relative_l2': max((r[0] for r in norms), default=0),
            'grad_min_cosine': min((r[1] for r in norms), default=1), 'passed': True}
    return report


def test_hybrid_qwen_right_padding_matches_independent_rows_and_gradients():
    hybrid_batch_equivalence('cpu')
