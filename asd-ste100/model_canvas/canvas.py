"""Canvas editor (PLAN.md 3.6, Model B), smoke-test version. No deployment runtime.

A bidirectional transformer reads [mode] [source pieces] and a canvas of 2n + SLACK
slots. Source piece i sits at canvas slot 2i, NULL fills the gaps, and the target is
placed by the tokenizer's alignment (align_ids). Training corrupts the target canvas
(MASK or the source-aligned token) and predicts it; inference starts from the source
canvas and fills it in a few passes, freezing confident slots and masking the rest.

Differences from the plan, kept for the smoke test: the canvas length is 2n + SLACK
instead of a 64/128/256 bucket, the checker features cover the source only (not the
canvas, which would need a checker call per corrupted sample), and the checker is a
report after the last pass, not a re-masking signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'model_tagger'))  # after model_canvas, so its train/decode win
from tagger import SteTokenizer, EncodingError, Lexicon, RULES, check, errors, checker_features

SLACK = 8
MASK_CASE = 4
TAUS = (.95, .9, .85, .8, .75)  # freeze thresholds; one more pass then accepts every slot


@dataclass(frozen=True)
class Config:
    width: int = 256
    layers: int = 6
    heads: int = 4
    ffn: int = 1024
    max_source: int = 128
    dropout: float = 0.  # CPU dropout in every layer (bernoulli_) was 60% of a step
    input_dropout: float = .1  # one dropout on the summed embeddings


def canvas_length(pieces):
    return 2 * pieces + SLACK


def source_canvas(row, tok):
    """The inference start: each source piece at 2i, NULL elsewhere (input IDs)."""
    null = tok.output_ids['NULL']
    length = canvas_length(len(row['source_ids']))
    ids, cases = [null] * length, [0] * length
    for i, (value, case) in enumerate(zip(row['source_ids'], row['source_case'])):
        ids[2 * i] = value
        cases[2 * i] = 0 if value in tok.slot_ids else case
    return ids, cases


def encode_source(tok, lexicon, text, mode):
    """Inference-time row: every detected span is a slot (no raw byte exposure)."""
    encoded = tok.encode_source(text, mode)
    return dict(encoded, src=text, mode=mode,
                features=checker_features(text, encoded, lexicon, mode).bool())


def encode_example(tok, lexicon, src, ste, mode):
    """A training row: source features plus the target canvas (output IDs and cases)."""
    encoded = tok.encode_pair({'src': src, 'ste': ste, 'mode': mode}, slot_probability=1.0)
    length = canvas_length(len(encoded['source_ids']))
    if encoded['target_positions'][-1] >= length:
        raise EncodingError('canvas_overflow')
    null = tok.output_ids['NULL']
    target, cases = [null] * length, [0] * length
    for position, value, case in zip(encoded['target_positions'], encoded['target_ids'], encoded['target_case']):
        target[position] = value
        cases[position] = 0 if value in tok.slot_ids else min(case, 2)
    row = {k: encoded[k] for k in ('source_ids', 'source_case', 'source_slot_refs', 'slots', 'mode_id')}
    return dict(row, src=src, ste=ste, mode=mode, target=target, target_case=cases,
                features=checker_features(src, encoded, lexicon, mode).bool())


def legal(tok, slots):
    return torch.tensor([x == 0 for x in tok.additive_mask(slots)])


def collate(rows, canvases, tok, mask_id, device='cpu'):
    """rows: encoded sources; canvases: (ids, cases) per row, ids are input IDs or mask_id."""
    pad = tok.output_ids['PAD']
    n, s, c = len(rows), max(len(r['source_ids']) for r in rows) + 1, max(len(ids) for ids, _ in canvases)
    src_ids = torch.full((n, s), pad, dtype=torch.long)
    src_case, src_refs = torch.zeros_like(src_ids), torch.zeros_like(src_ids)
    src_pad = torch.ones((n, s), dtype=torch.bool)
    features = torch.zeros((n, s, len(RULES)))
    canvas_ids = torch.full((n, c), pad, dtype=torch.long)
    canvas_case = torch.zeros_like(canvas_ids)
    canvas_pad = torch.ones((n, c), dtype=torch.bool)
    allowed = torch.zeros((n, tok.meta['output_size']), dtype=torch.bool)
    for i, (row, (ids, cases)) in enumerate(zip(rows, canvases)):
        length = len(row['source_ids']) + 1
        src_ids[i, :length] = torch.tensor([row['mode_id']] + row['source_ids'])
        src_case[i, 1:length] = torch.tensor(row['source_case'])
        src_refs[i, 1:length] = torch.tensor([r + 1 for r in row['source_slot_refs']])
        src_pad[i, :length] = False
        features[i, :length] = row['features'].float()
        canvas_ids[i, :len(ids)] = torch.tensor(ids)
        canvas_case[i, :len(ids)] = torch.tensor(cases)
        canvas_pad[i, :len(ids)] = False
        allowed[i] = legal(tok, row['slots'])
    return {k: v.to(device) for k, v in dict(src_ids=src_ids, src_case=src_case, src_refs=src_refs,
            src_pad=src_pad, features=features, canvas_ids=canvas_ids, canvas_case=canvas_case,
            canvas_pad=canvas_pad, legal=allowed).items()}


def corrupt(row, tok, mask_id, rng, start_share=.2):
    """One training canvas. With start_share it is the inference start (the whole source
    canvas); otherwise a fraction t ~ U(.05, 1) of slots is corrupted, each to MASK or to
    the source-aligned token with equal odds, and the rest show the target."""
    source, source_cases = source_canvas(row, tok)
    target, target_cases = row['target'], row['target_case']
    if rng.random() < start_share:
        return source, source_cases, [True] * len(target)
    t = .05 + .95 * rng.random()
    corrupted = [rng.random() < t for _ in target]
    if not any(corrupted):
        corrupted[rng.randrange(len(target))] = True
    ids, cases = [], []
    for p, hit in enumerate(corrupted):
        if not hit:
            ids.append(target[p]); cases.append(target_cases[p])
        elif rng.random() < .5:
            ids.append(mask_id); cases.append(MASK_CASE)
        else:
            ids.append(source[p]); cases.append(source_cases[p])
    return ids, cases, corrupted


def training_batch(rows, tok, mask_id, rng, start_share=.2, device='cpu'):
    null = tok.output_ids['NULL']
    canvases, targets, target_cases, selected, copies, cased = [], [], [], [], [], []
    for row in rows:
        ids, cases, corrupted = corrupt(row, tok, mask_id, rng, start_share)
        source, _ = source_canvas(row, tok)
        canvases.append((ids, cases))
        targets.append(row['target']); target_cases.append(row['target_case'])
        selected.append(corrupted); copies.append([a == b for a, b in zip(row['target'], source)])
        cased.append([v != null and v not in tok.slot_ids for v in row['target']])
    batch = collate(rows, canvases, tok, mask_id, device)
    c = batch['canvas_ids'].shape[1]
    def padded(values, fill, dtype):
        out = torch.full((len(values), c), fill, dtype=dtype)
        for i, v in enumerate(values):
            out[i, :len(v)] = torch.tensor(v, dtype=dtype)
        return out.to(device)
    batch.update(target=padded(targets, -100, torch.long), target_case=padded(target_cases, 0, torch.long),
                 selected=padded(selected, False, torch.bool), copy=padded(copies, False, torch.bool),
                 cased=padded(cased, False, torch.bool))
    return batch


class CanvasEditor(nn.Module):
    def __init__(self, tokenizer, config=Config()):
        super().__init__()
        self.config = config
        self.vocab = tokenizer.meta['output_size']
        self.mask_id = tokenizer.meta['input_size']
        width = config.width
        self.token = nn.Embedding(self.mask_id + 1, width, padding_idx=tokenizer.output_ids['PAD'])
        self.case = nn.Embedding(MASK_CASE + 1, width)
        self.slot_ref = nn.Embedding(self.vocab + 1, width, padding_idx=0)
        # Source piece i and canvas slots 2i, 2i+1 share position 1+i (0 is the mode token).
        self.position = nn.Embedding(config.max_source + SLACK // 2 + 2, width)
        self.parity = nn.Embedding(2, width)
        self.segment = nn.Embedding(2, width)
        self.violation = nn.Linear(len(RULES), width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.input_dropout = nn.Dropout(config.input_dropout)
        layer = nn.TransformerEncoderLayer(width, config.heads, config.ffn, config.dropout,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)
        # The output head is tied to the output-word rows of the input embedding.
        self.bias = nn.Parameter(torch.zeros(self.vocab))
        self.case_head = nn.Linear(width, 3)

    def forward(self, batch):
        src, canvas = batch['src_ids'], batch['canvas_ids']
        if src.shape[1] > self.config.max_source + 1:
            raise ValueError('position capacity exceeded')
        device = src.device
        i = torch.arange(src.shape[1], device=device)[None, :]
        k = torch.arange(canvas.shape[1], device=device)[None, :]
        source = (self.token(src) + self.case(batch['src_case']) + self.slot_ref(batch['src_refs'])
                  + self.position(i) + self.violation(batch['features']) + self.segment.weight[0])
        draft = (self.token(canvas) + self.case(batch['canvas_case']) + self.position(1 + k // 2)
                 + self.parity(k % 2) + self.segment.weight[1])
        hidden = self.encoder(self.input_dropout(self.norm(torch.cat([source, draft], 1))),
                              src_key_padding_mask=torch.cat([batch['src_pad'], batch['canvas_pad']], 1))
        hidden = hidden[:, src.shape[1]:]
        logits = F.linear(hidden, self.token.weight[:self.vocab], self.bias)
        return logits.masked_fill(~batch['legal'][:, None, :], float('-inf')), self.case_head(hidden)


def loss(logits, case_logits, batch, copy_weight=.4):
    """Cross-entropy on corrupted slots; slots whose target equals the source-aligned token weigh copy_weight."""
    selected = batch['selected'] & batch['target'].ne(-100)
    if not selected.any():
        raise ValueError('no supervised positions')
    weights = torch.where(batch['copy'], copy_weight, 1.0)[selected]
    token_loss = (F.cross_entropy(logits[selected].float(), batch['target'][selected], reduction='none')
                  * weights).sum() / weights.sum()
    words = selected & batch['cased']
    case_loss = (F.cross_entropy(case_logits[words].float(), batch['target_case'][words])
                 if words.any() else case_logits.sum() * 0)
    return token_loss + case_loss


def unfreeze_repeats(ids, frozen, confidence, null, tok, mask_id, cases):
    """Adjacent frozen copies of one word (NULL gaps skipped): mask the less confident one."""
    words = [p for p, v in enumerate(ids) if v != null]
    for p, q in zip(words, words[1:]):
        if frozen[p] and frozen[q] and ids[p] == ids[q] and ids[p] != mask_id and ids[p] not in tok.slot_ids:
            r = p if confidence[p] < confidence[q] else q
            frozen[r], ids[r], cases[r] = False, mask_id, MASK_CASE


def fit_slots(ids, cases, probs, slots, tok):
    """Give each active slot exactly its source count: keep its most probable positions,
    give extra copies their best non-slot word, and fill missing copies at the
    non-slot positions that most prefer the slot."""
    slot_ids = torch.tensor(sorted(tok.slot_ids))
    for slot in slots:
        value, need = tok.output_ids[slot['token']], slot['count']
        have = sorted((p for p, v in enumerate(ids) if v == value), key=lambda p: -float(probs[p][value]))
        for p in have[need:]:
            scores = probs[p].clone()
            scores[slot_ids] = -1
            ids[p], cases[p] = int(scores.argmax()), 0
        if len(have) < need:
            free = sorted((p for p, v in enumerate(ids) if v not in tok.slot_ids), key=lambda p: -float(probs[p][value]))
            for p in free[:need - len(have)]:
                ids[p], cases[p] = value, 0


@torch.no_grad()
def rewrite(model, tok, lexicon, text, mode='PROC', taus=TAUS):
    """Fill the source canvas in len(taus) + 1 passes. A slot is frozen once its top
    probability passes the pass threshold; unfrozen slots are masked for the next pass,
    as is the less confident of two adjacent frozen copies of a word, and the last
    pass accepts every slot. Then adjacent repeats are dropped and each copy slot is
    fitted to its source count."""
    model.eval()
    device = next(model.parameters()).device
    try:
        row = encode_source(tok, lexicon, text, mode)
    except EncodingError as error:
        return {'text': None, 'passes': 0, 'resolved': False, 'reason': str(error)}
    start, start_cases = source_canvas(row, tok)
    ids, cases, frozen = list(start), list(start_cases), [False] * len(start)
    null, passes = tok.output_ids['NULL'], 0
    probs, confidence = [None] * len(ids), [0.] * len(ids)
    for tau in (*taus, 0.):
        passes += 1
        logits, case_logits = model(collate([row], [(ids, cases)], tok, model.mask_id, device))
        distribution = logits[0].float().softmax(-1)
        best, predicted = distribution.max(-1)
        predicted_case = case_logits[0].argmax(-1)
        for p in range(len(ids)):
            if frozen[p]:
                continue
            if best[p] >= tau:
                frozen[p], probs[p], confidence[p] = True, distribution[p], float(best[p])
                ids[p] = int(predicted[p])
                cases[p] = 0 if ids[p] in tok.slot_ids else int(predicted_case[p])
            else:
                ids[p], cases[p] = model.mask_id, MASK_CASE
        if tau:
            unfreeze_repeats(ids, frozen, confidence, null, tok, model.mask_id, cases)
        if all(frozen):
            break
    words = [p for p, v in enumerate(ids) if v != null]
    for p, q in zip(words, words[1:]):
        if ids[p] == ids[q] and ids[q] not in tok.slot_ids:
            ids[q] = null
    fit_slots(ids, cases, probs, row['slots'], tok)
    if [(v, c) for v, c in zip(ids, cases) if v != null] == [(v, c) for v, c in zip(start, start_cases) if v != null]:
        updated = text  # nothing changed: keep the source string, not its detokenized form
    else:
        try:
            values = [(v, c) for v, c in zip(ids, cases) if v != null]
            updated = tok.detokenize([v for v, _ in values], row['slots'], [c for _, c in values]) if values else None
        except EncodingError as error:
            return {'text': None, 'passes': passes, 'resolved': False, 'reason': str(error)}
        if updated is None:
            return {'text': None, 'passes': passes, 'resolved': False, 'reason': 'empty_prediction'}
    clean = not errors(check(updated, lexicon, mode))
    return {'text': updated, 'passes': passes, 'resolved': clean, 'reason': 'checker_clean' if clean else 'unresolved'}
