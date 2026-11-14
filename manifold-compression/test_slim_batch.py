import collections
import json

import numpy as np
import pytest
import torch

import moe_carve as mc
import slim_batch as sb


def tiny_qwen(seed=0, layers=4):
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(
        vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=layers, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=8, linear_value_head_dim=8, initializer_range=0.2,
        layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] * (layers // 4),
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000., 'partial_rotary_factor': 0.5,
                         'mrope_section': [2, 1, 1], 'mrope_interleaved': True})
    return AutoModelForCausalLM.from_config(config).float().eval().requires_grad_(False)


def ids(seed, length=24, vocab=96):
    return torch.randint(0, vocab, (length,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_stored_fp32_state_reproduces_the_stock_kernels():
    model = tiny_qwen()
    x = ids(1)
    stock = sb.hidden(model, [x, ids(2)])
    with sb.StoredState(model, ['fp32', 'fp32'], 0):
        stored = sb.hidden(model, [x, ids(2)])
    torch.testing.assert_close(stored, stock, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sb.hidden(model, [x, ids(2)]), stock)


def test_stored_state_decode_continues_the_prefill():
    model = tiny_qwen()
    x = torch.tensor([ids(3)])
    with torch.no_grad(), sb.StoredState(model, ['int16_row'], 0):
        whole = model(input_ids=x, use_cache=False).logits
        out = model(input_ids=x[:, :-3], use_cache=True)
        steps = [out.logits]
        for t in range(x.shape[1] - 3, x.shape[1]):
            out = model(input_ids=x[:, t:t + 1], past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, 1), whole, atol=1e-4, rtol=1e-4)


def test_store_formats_round_trip_within_their_precision():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 4, 8, 16, generator=g) * torch.logspace(-3, 2, 16)
    assert torch.equal(sb.store('fp32', x, g), x)
    assert torch.equal(sb.store('bf16', x, g), x.bfloat16().float())
    for name, top in (('int16_row', 32767), ('int8_row', 127)):
        y = sb.store(name, x, g)
        assert ((y - x).abs() <= x.abs().amax(-1, keepdim=True) / top / 2 + 1e-12).all()
    y = sb.store('fp16_sr', x, g)
    assert torch.equal(y, y.half().float())
    assert ((y - x).abs() <= (x.half().float() - x).abs() * 2 + x.abs() * 2 ** -10).all()
    # Unbiased in fp16's normal range (below it, fp16 itself rounds to nearest).
    mean = torch.stack([sb.store('fp16_sr', x, g) for _ in range(2000)]).mean(0)
    normal = x.abs() > 2 ** -14
    assert ((mean - x).abs() / x.abs())[normal].max() < 2e-4
    jitter = sb.store('fp32_jitter', x, g)
    assert ((jitter - x).abs() <= x.abs() * 2 ** -19).all()


def test_stored_state_gives_each_batch_element_its_own_format():
    model = tiny_qwen()
    x = ids(4)
    with sb.StoredState(model, ['fp32', 'int8_row', 'fp32'], 0):
        h = sb.hidden(model, [x, x, x])
    torch.testing.assert_close(h[0], h[2])
    assert (h[0] - h[1]).abs().max() > 1e-5


def test_deltanet_cut_drops_per_element_and_swaps_weights():
    model = tiny_qwen()
    cut = sb.DeltaNetCut(model)
    x = ids(5)
    base = sb.hidden(model, [x])[0]
    cut.keep = {1: torch.tensor([1., 0.])}
    h = sb.hidden(model, [x, x])
    torch.testing.assert_close(h[0], base)
    assert (h[1] - base).abs().max() > 1e-4
    cut.keep = {}
    block = cut.blocks[1]
    original = block.out_proj.weight.data.clone()
    cut.swap({1: {'out_proj': torch.zeros_like(original)}})
    assert block.out_proj.weight.abs().sum() == 0
    torch.testing.assert_close(sb.hidden(model, [x, x])[1], h[1])
    cut.swap({})
    assert torch.equal(block.out_proj.weight.data, original)


def test_whitened_low_rank_is_exact_at_full_rank_and_beats_plain_svd():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(12, 8, generator=g)
    x = torch.randn(500, 8, generator=g) * torch.tensor([10., 5, 1, 1, .1, .1, .1, .1])
    cov = (x.T @ x).double()
    full, low = sb.lowrank(w, cov, [8, 3], 1e-9)
    torch.testing.assert_close(full, w, atol=1e-4, rtol=1e-4)
    u, s, vh = torch.linalg.svd(w)
    plain = (u[:, :3] * s[:3]) @ vh[:3]
    err = lambda m: ((x @ (w - m).T) ** 2).sum()
    assert err(low) < err(plain)


def test_input_moments_and_lowrank_weights_count_their_saving():
    model = tiny_qwen()
    cut = sb.DeltaNetCut(model)
    docs = [{'input_ids': ids(s)} for s in range(4)]
    cov = sb.input_moments(model, cut, docs)
    assert set(cov) == {(i, j) for i in cut.blocks for j in (0, 1)}
    (weights, saved), = sb.lowrank_weights(cut, cov, [0.5], 1e-6)
    expected = 0
    for i, m in cut.blocks.items():
        for name in sb.PROJECTIONS:
            w = getattr(m, name).weight
            r = sb.rank_for(w, 0.5)
            assert torch.linalg.matrix_rank(weights[i][name].float()) <= r
            expected += w.numel() - r * sum(w.shape)
    assert saved == expected > 0


def test_row_cache_streams_lru_membership():
    mc.DEVICE = 'cpu'
    cache = sb.RowCache(2, 10)
    masks = cache.stream([1, 2, 1, 3, 4])
    member = [set(m.nonzero().flatten().tolist()) for m in masks]
    assert member == [{1}, {1, 2}, {1, 2}, {1, 3}, {3, 4}]


def test_cache_features_match_a_direct_computation():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(5, 10, generator=g) * 3
    masks = torch.rand(5, 10, generator=g) > 0.5
    masks[:, 0] = True
    targets = torch.randint(0, 10, (5,), generator=g)
    f = sb.cache_features(logits, masks, targets)
    p = logits.softmax(-1)
    torch.testing.assert_close(f['p_out'], 1 - (p * masks).sum(-1), atol=1e-6, rtol=1e-5)
    for t in range(5):
        q = logits[t][masks[t]].softmax(-1)
        torch.testing.assert_close(f['entropy'][t], -(q * q.log()).sum(), atol=1e-5, rtol=1e-5)
        assert bool(f['pick_out'][t]) != bool(masks[t, logits[t].argmax()])
        assert bool(f['true_out'][t]) != bool(masks[t, targets[t]])
        assert masks[t, f['cached_pick'][t]]


def test_detector_threshold_keeps_the_requested_recall():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(4000, 3, generator=g)
    y = (x[:, 0] + 0.5 * torch.randn(4000, generator=g)) > 1
    det = sb.fit_detector(x, y, 0.95, 0)
    fired = sb.detect(det, x)
    assert (fired & y).sum() / y.sum() >= 0.95
    assert fired.float().mean() < 0.5


def byte_level_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    text = ['the cat sat on the mat', 'def forward(self, x): return x + 1', 'naïve café 漢字 汉字',
            'theta θ and pi π', 'cats and mats and hats'] * 20
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=['<|endoftext|>'],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(text, trainer)
    return tok


def test_trimmed_vocabulary_keeps_merge_ancestry_and_round_trips():
    tok = byte_level_tokenizer()
    tj = json.loads(tok.to_str())
    english = 'the cat sat on the mat and def return'
    counts = collections.Counter(tok.encode(english).ids)
    size = 300
    kept, source = sb.trim_vocab(tj, counts, size)
    assert len(kept) == len(set(kept)) <= size
    assert source['alphabet'] == 256 and source['special'] == 1
    inv = {i: t for t, i in tj['model']['vocab'].items()}
    keep = {inv[i] for i in kept if i in inv}
    for m in tj['model']['merges']:
        a, b = m.split(' ', 1) if isinstance(m, str) else m
        if a + b in keep:
            assert a in keep and b in keep
    short = sb.TrimmedTokenizer(tj, kept)
    for text in (english, 'naïve 漢字 θ', 'cats and hats<|endoftext|>x'):
        out = short.encode(text)
        assert set(out) <= set(kept)
        assert short.decode(out) == text
        assert len(out) >= len(tok.encode(text).ids)
    assert short.encode(english) == tok.encode(english).ids
    assert short.encode('<|endoftext|>') == [tok.token_to_id('<|endoftext|>')]


def test_english_filter():
    assert sb.english(' the_cat(x) -> θ — “ok” …')
    assert not sb.english('café') and not sb.english('漢字')


def test_final_number_prefers_the_boxed_answer_after_reasoning():
    assert sb.final_number('<think>maybe 3 or 4</think> so \\boxed{1,234}.') == 1234
    assert sb.final_number('The answer is 18.') == 18
    assert sb.final_number('no digits') is None


def test_parameter_counts_follow_the_plan():
    model = tiny_qwen()
    c = model.config
    plan = {'router': 0, 'active_neurons': c.num_hidden_layers * c.intermediate_size, 'vocab': c.vocab_size,
            'head_rows_per_token': c.vocab_size, 'deltanet_saved': 0}
    counts = sb.parameter_counts(c, plan)
    total = sum(p.numel() for p in model.parameters())
    dense = counts['steps']['dense']
    embed = c.vocab_size * c.hidden_size
    assert dense['stored_total'] == total
    assert dense['stored_non_embedding'] == total - 2 * embed
    assert dense['active_total'] == total - embed
    for step in ('moe', 'vocab', 'cache', 'deltanet'):
        assert counts['steps'][step] == dense
    half = dict(plan, active_neurons=plan['active_neurons'] // 2, vocab=48, head_rows_per_token=12,
                deltanet_saved=100)
    steps = sb.parameter_counts(c, half)['steps']
    mlp = c.num_hidden_layers * 3 * c.intermediate_size * c.hidden_size
    assert steps['moe']['active_non_embedding'] == dense['active_non_embedding'] - mlp // 2
    assert steps['vocab']['stored_total'] == dense['stored_total'] - 2 * (c.vocab_size - 48) * c.hidden_size
    assert steps['cache']['active_total'] == steps['moe']['active_non_embedding'] + 12 * c.hidden_size
    assert steps['deltanet']['stored_non_embedding'] == steps['cache']['stored_non_embedding'] - 100
