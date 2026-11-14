import copy
import json
import types

import numpy as np
import torch
import torch.nn.functional as F

import heal_sessions as hs
from test_slim_heal import lora_nonzero, student
from test_slim_batch import ids

KEPT = set(range(80))
EOT = 79


def write_shard(folder, name, rows, teacher):
    path = folder / name
    path.mkdir(parents=True)
    (path / 'samples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    np.savez(path / 'teacher.npz', **teacher)


def one_hot_teacher(answer):
    """Top-20 whose first entry is the written token with probability 1."""
    t = np.full((len(answer), 20), -1, np.int32)
    lp = np.full((len(answer), 20), -np.inf, np.float16)
    t[:, 0], lp[:, 0] = answer, 0
    return t, lp


def fake_pool(tmp_path, n=6, seed=0):
    g = np.random.default_rng(seed)
    rows, teacher = [], {}
    for k in range(n):
        prompt = g.integers(0, 78, 5).tolist()
        answer = g.integers(0, 78, 4 + k % 3).tolist()
        rows.append({'id': f'p{k // 2}', 'sample': k % 2, 'prompt_ids': prompt, 'answer_ids': answer,
                     'correct': True, 'finished': True})
        teacher[f'ids_{k}'], teacher[f'logprobs_{k}'] = one_hot_teacher(answer)
    write_shard(tmp_path, 'shard_000', rows, teacher)
    return rows


SELECT = {'alpha': 0.0, 'low': 0.0, 'high': 100.0, 'seed': 0}


def test_length_rule_keeps_short_answers_more_often_and_drops_the_extremes():
    assert hs.keep_probability(50, 100, 1.0, 1 / 3, 2) == 1.0
    assert hs.keep_probability(200, 100, 1.0, 1 / 3, 2) == 0.5
    assert hs.keep_probability(201, 100, 1.0, 1 / 3, 2) == 0.0
    assert hs.keep_probability(30, 100, 1.0, 1 / 3, 2) == 0.0
    rows = [{'id': 'a', 'sample': s, 'answer_ids': [1] * n, 'correct': c, 'finished': True}
            for s, (n, c) in enumerate([(10, True), (20, True), (30, False), (11, True)])]
    assert hs.select(rows, {'alpha': 0.0, 'low': 0.0, 'high': 100.0, 'seed': 0}) == [0, 1, 3]
    rows[1]['finished'] = False
    assert hs.select(rows, {'alpha': 0.0, 'low': 0.0, 'high': 100.0, 'seed': 0}) == [0, 3]


def test_packing_lines_the_teacher_up_with_the_tokens_it_predicts(tmp_path):
    rows = fake_pool(tmp_path)
    pool = hs.Pool(tmp_path, SELECT, EOT)
    (x, mask, tids, tlp), cursor = pool.batch([0, 0], 2, 24)
    for b in range(2):
        where = mask[b].nonzero()[0]
        # Every masked position's top teacher token is the next token, and only answer tokens are masked.
        assert (tids[b, where, 0] == x[b, where + 1]).all()
        assert (tlp[b, where, 0] == 0).all()
    assert mask.sum() == sum(len(rows[k]['answer_ids']) for k in range(cursor[1]))
    assert pool.batch(cursor, 50, 24) is None


def test_stored_loss_is_the_kl_to_the_renormalized_top20_and_ce(tmp_path):
    g = torch.Generator().manual_seed(3)
    head, h = torch.randn(80, 8, generator=g), torch.randn(5, 8, generator=g)
    index = torch.arange(96)
    index[80:] = -1
    tids = torch.randint(0, 96, (5, 20), generator=g)
    tlp = torch.randn(5, 20, generator=g).log_softmax(-1)
    target = torch.randint(0, 80, (5,), generator=g)
    kl, ce = hs.stored_loss(h, target, tids, tlp, head, index)
    logp = (h @ head.T).log_softmax(-1)
    want = 0
    for p in range(5):
        ok = tids[p] < 80
        pt = tlp[p][ok].log_softmax(-1)
        want += (pt.exp() * (pt - logp[p, tids[p][ok]])).sum()
    torch.testing.assert_close(kl, want)
    torch.testing.assert_close(ce, F.cross_entropy(h @ head.T, target, reduction='sum'))


def session(model, slim, tmp_path, state, steps, checkpoint=None, every=None, stop=None):
    kept = torch.arange(80)
    index = torch.full((96,), -1)
    index[kept] = kept
    vocab = types.SimpleNamespace(index=index, head=model.lm_head.weight[kept].detach())
    cfg = {'loss_chunk': 7, 'ce_weight': 0.5, 'router_weight': 1.0, 'lr': 1e-2, 'router_lr': 1e-2,
           'betas': [0.9, 0.99], 'clip': 1.0, 'batch_sequences': 2, 'micro_batch': 1, 'seq_len': 24,
           'warmup_steps': 2, 'log_every': 100}
    if every:
        cfg['checkpoint_every'] = every
    pool = hs.Pool(tmp_path, SELECT, EOT)
    return hs.train(model, slim, vocab, pool, state, cfg, float('inf'), steps=steps, checkpoint=checkpoint, stop=stop)


def test_a_session_resumed_from_its_saved_state_continues_exactly(tmp_path):
    fake_pool(tmp_path, n=12)
    model, slim = student()
    lora_nonzero(slim)
    start = dict(slim.state(), step=0, cursor=[0, 0], optimizer=None, router_optimizer=None)
    whole, losses = session(model, slim, tmp_path, start, 2)
    assert len(losses) == 2 and all(np.isfinite(l['kl']) and l['positions'] > 0 for l in losses)
    model, slim = student()
    lora_nonzero(slim)
    half, _ = session(model, slim, tmp_path, start, 1)
    path = tmp_path / 'state.pt'
    torch.save(half, path)
    model, slim = student()
    saved = torch.load(path, weights_only=False)
    slim.load({'lora': saved['lora'], 'routers': saved['routers']})
    resumed, _ = session(model, slim, tmp_path, saved, 1)
    assert resumed['step'] == whole['step'] == 2 and resumed['cursor'] == whole['cursor']
    for n, (a, b) in whole['lora'].items():
        torch.testing.assert_close(resumed['lora'][n][0], a)
        torch.testing.assert_close(resumed['lora'][n][1], b)


def test_a_checkpoint_resumes_as_if_the_session_had_not_been_cut(tmp_path):
    # A crash after a checkpoint loses only the steps after it: resuming the checkpointed state
    # ends where the uncut session ends.
    fake_pool(tmp_path, n=12)
    model, slim = student()
    lora_nonzero(slim)
    start = dict(slim.state(), step=0, cursor=[0, 0], optimizer=None, router_optimizer=None)
    saved = []
    # Deep copies, as torch.save would freeze them: on the CPU the state shares the live tensors.
    whole, _ = session(model, slim, tmp_path, start, 2, checkpoint=lambda st, losses: saved.append(copy.deepcopy(st)),
                       every=1)
    assert [st['step'] for st in saved] == [1, 2]
    model, slim = student()
    slim.load({'lora': saved[0]['lora'], 'routers': saved[0]['routers']})
    resumed, _ = session(model, slim, tmp_path, saved[0], 1)
    assert resumed['step'] == whole['step'] == 2 and resumed['cursor'] == whole['cursor']
    for n, (a, b) in whole['lora'].items():
        torch.testing.assert_close(resumed['lora'][n][0], a)
        torch.testing.assert_close(resumed['lora'][n][1], b)


def test_a_session_stops_when_asked_to_yield_the_gpu(tmp_path):
    # Another session queueing for the GPU (or the end of a segment) stops training before the next step.
    fake_pool(tmp_path, n=12)
    model, slim = student()
    start = dict(slim.state(), step=0, cursor=[0, 0], optimizer=None, router_optimizer=None)
    asked = []
    state, losses = session(model, slim, tmp_path, start, 5, stop=lambda: asked.append(1) or len(asked) > 2)
    assert state['step'] == 2 and len(losses) == 2


def test_each_thinking_mode_has_its_own_median():
    # Two short thinking-off answers and two long thinking-on ones: with one median for all four,
    # the high and low cuts would drop every answer.
    rows = [{'id': 'a', 'sample': s, 'thinking': s >= 2, 'answer_ids': [1] * n, 'correct': True, 'finished': True}
            for s, n in enumerate([10, 12, 100, 110])]
    assert hs.select(rows, {'alpha': 0.0, 'low': 1 / 3, 'high': 2.0, 'seed': 0}) == [0, 1, 2, 3]


def fake_vllm(monkeypatch, plan):
    """A stand-in for vllm's LLM: plan[prompt index] lists (tokens, finish_reason) per sample; each
    token's teacher entry ranks the token itself first, then 20 others."""
    import sys

    class Logprob(types.SimpleNamespace):
        pass

    class Tok:
        def convert_tokens_to_ids(self, t):
            return {'<|im_end|>': 70, '<|endoftext|>': 79}[t]

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return f"{messages[0]['content']}|{int(enable_thinking)}"

        def encode(self, text, add_special_tokens):
            return [ord(c) % 60 for c in text][:6]

        def decode(self, ids):
            return ' '.join(map(str, ids))

    class LLM:
        calls = []

        def get_tokenizer(self):
            return Tok()

        def generate(self, prompts, params):
            out = []
            for k, (prompt, p) in enumerate(zip(prompts, params)):
                LLM.calls.append(p)
                samples = []
                for tokens, reason in plan[len(LLM.calls) - 1]:
                    entries = [{t: Logprob(rank=1, logprob=-0.1),
                                **{j: Logprob(rank=r + 2, logprob=-3.0 - r) for r, j in enumerate(x for x in range(30) if x != t)}}
                               for t in tokens]
                    samples.append(types.SimpleNamespace(token_ids=tokens, logprobs=entries, finish_reason=reason))
                out.append(types.SimpleNamespace(outputs=samples))
            return out

    module = types.ModuleType('vllm')
    module.SamplingParams = lambda **kw: kw
    inputs = types.ModuleType('vllm.inputs')
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    monkeypatch.setitem(sys.modules, 'vllm', module)
    monkeypatch.setitem(sys.modules, 'vllm.inputs', inputs)
    return LLM()


def test_generated_shards_feed_the_pool_with_the_stop_token_and_its_teacher_row(tmp_path, monkeypatch):
    import heal_shards
    problems = [{'id': 'q0', 'source': 'gsm8k', 'question': 'one', 'answer': '7', 'grader': 'number', 'samples': {'off': 2, 'on': 0}},
                {'id': 'q1', 'source': 'chat', 'question': 'two', 'answer': None, 'grader': None, 'samples': {'off': 1, 'on': 1}}]
    path = tmp_path / 'problems.jsonl'
    path.write_text(''.join(json.dumps(p) + '\n' for p in problems))
    # q0 off: "7" right (then the stop token), and an answer cut at the cap; q1 off finished, q1 on finished.
    plan = [[([7, 70], 'stop'), ([5, 5, 5], 'length')], [([3, 4, 70], 'stop')], [([9, 70], 'stop')]]
    llm = fake_vllm(monkeypatch, plan)
    out = tmp_path / 'pool' / 'shard_000'
    args = types.SimpleNamespace(problems=str(path), out=str(out), until=float('inf'), model=None)
    heal_shards.generate(args, llm)
    rows = [json.loads(l) for l in (out / 'samples.jsonl').open()]
    assert [(r['id'], r['sample'], r['finished'], r['correct']) for r in rows] == \
        [('q0', 0, True, True), ('q0', 1, False, False), ('q1', 0, True, None), ('q1', 2, True, None)]
    assert rows[0]['answer_ids'] == [7, 70] and llm.calls[0]['n'] == 2 and llm.calls[2]['n'] == 1
    teacher = np.load(out / 'teacher_0.npz')
    assert 'ids_1' not in teacher.files and teacher['ids_0'].shape == (2, 20)
    # Resuming finds nothing left to do.
    heal_shards.generate(args, llm)
    assert len(llm.calls) == 3
    pool = hs.Pool(tmp_path / 'pool', SELECT, EOT)
    # Items: 6 prompt + 2 answer + 1 separator, then 6 + 3 + 1: one 19-token row.
    (x, mask, tids, tlp), cursor = pool.batch([0, 0], 1, 19)
    where = mask[0].nonzero()[0]
    assert (tids[0, where, 0] == x[0, where + 1]).all() and 70 in x[0, where + 1]
    assert mask.sum() == 2 + 3 and cursor == [0, 2]


def test_teacher_tokens_outside_the_64k_vocabulary_become_the_students_pieces():
    import time
    import moe_carve as mc
    import slim_batch as sb
    mc.configure(sb.CARVE_RULES)
    full, short, kept = sb.load_tokenizers()
    t0 = time.time()
    tokens = hs.StudentTokens(full, short, kept, 248320)
    built = time.time() - t0
    kept = set(kept)
    text = 'Total liters:** 12 kittens.**<|im_end|>'
    ids = full.encode(text, add_special_tokens=False).ids
    outside = [p for p, t in enumerate(ids) if t not in kept]
    assert outside, 'the example must use tokens outside the vocabulary'
    # The teacher row at each position: the written token first, then a token outside the vocabulary.
    tids = np.array([[t, ids[outside[0]]] + [-1] * 18 for t in ids], np.int32)
    tlp = np.array([[-0.1, -2.5] + [-np.inf] * 18] * len(ids), np.float16)
    answer, rows, lps = tokens.answer(ids, tids, tlp)
    assert short.decode(answer) == text and all(t in kept for t in answer)
    assert all(t in kept for t in rows[rows >= 0])
    # Exactly the extra pieces lack a teacher row; every row's first entry is the token it predicts
    # (a piece's first piece is itself).
    assert (rows[:, 0] < 0).sum() == len(answer) - len(ids)
    have = rows[:, 0] >= 0
    assert (rows[have, 0] == np.asarray(answer)[have]).all()
    assert tokens.prompt(full.encode(text, add_special_tokens=False).ids) == short.encode(text)
    assert built < 60


def test_the_dense_canary_counts_docs_whose_dense_pass_differs_from_a_clean_run():
    ref = [{'id': f'd{i}', 'nll': {'dense': 3.0 + i}, 'trimmed': {'nll_dense': 1500.0 + i}} for i in range(4)]
    rows = copy.deepcopy(ref)
    rows[0]['nll']['dense'] += 1e-5            # rounding: not a fault
    rows[1]['trimmed']['nll_dense'] += 2100.0  # the 64K path blown up, as in check_75_off
    rows[2]['nll']['dense'] += 0.4
    assert hs.dense_canary(rows, ref) == {'docs': 4, 'faulty_docs': 2, 'faulty': ['d1', 'd2']}
