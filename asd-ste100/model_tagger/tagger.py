"""Offline iterative tagger baseline. No trained weights or deployment runtime."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tokenizer'))
from ste_tokenizer import SteTokenizer, EncodingError, align_ids
from checker import Lexicon, check, errors, split_sentences

RULES = ('R1.1', 'R1.2', 'R3.1', 'R3.4', 'R3.6', 'R4.2', 'R5.1', 'R6.3', 'R6.6', 'R8.1')
TRANSFORMS = ('LOWER', 'TITLE', 'UPPER', 'ADD_S', 'ADD_ES', 'Y_IES', 'DROP_S', 'DROP_ES',
              'IES_Y', 'ADD_ED', 'ADD_D', 'Y_IED', 'DROP_ED', 'DROP_D', 'IED_Y',
              'ADD_ING', 'E_ING', 'DROP_ING', 'ING_E', 'MERGE_NEXT')


@dataclass(frozen=True)
class Config:
    width: int = 512
    layers: int = 6
    heads: int = 8
    ffn: int = 2048
    max_positions: int = 257
    dropout: float = .1


class Tags:
    KEEP, DELETE = 0, 1

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.meta['output_size']
        self.transform_offset = 2 + 2 * self.vocab
        self.size = self.transform_offset + len(TRANSFORMS)

    def replace(self, value):
        return 2 + value

    def append(self, value):
        return 2 + self.vocab + value

    def transform(self, name):
        return self.transform_offset + TRANSFORMS.index(name)

    def transformed(self, units, index, name):
        if not 0 <= index < len(units):
            return None
        value, case = units[index]
        nxt = units[index + 1][0] if name == 'MERGE_NEXT' and index + 1 < len(units) else -1
        return self._transformed(value, case, nxt, name)

    @lru_cache(maxsize=262144)
    def _transformed(self, value, case, nxt, name):
        tok = self.tokenizer
        words = tok.meta['input_word_tokens']
        if value in tok.slot_ids or value in tok.control_ids or not 0 <= value < len(words):
            return None
        word = words[value]
        if name in ('LOWER', 'TITLE', 'UPPER'):
            return (value, ('LOWER', 'TITLE', 'UPPER').index(name)) if value < self.vocab else None
        if name == 'MERGE_NEXT':
            if nxt < 0:
                return None
            if nxt in tok.slot_ids or nxt in tok.control_ids or not 0 <= nxt < len(words):
                return None
            candidate = word + words[nxt]
        else:
            rules = {'ADD_S': ('', 's'), 'ADD_ES': ('', 'es'), 'Y_IES': ('y', 'ies'),
                     'DROP_S': ('s', ''), 'DROP_ES': ('es', ''), 'IES_Y': ('ies', 'y'),
                     'ADD_ED': ('', 'ed'), 'ADD_D': ('', 'd'), 'Y_IED': ('y', 'ied'),
                     'DROP_ED': ('ed', ''), 'DROP_D': ('d', ''), 'IED_Y': ('ied', 'y'),
                     'ADD_ING': ('', 'ing'), 'E_ING': ('e', 'ing'),
                     'DROP_ING': ('ing', ''), 'ING_E': ('ing', 'e')}
            suffix, ending = rules[name]
            if not word.isalpha() or not word.endswith(suffix):
                return None
            candidate = (word[:-len(suffix)] if suffix else word) + ending
        result = tok.output_ids.get(candidate)
        if result is None or result in tok.control_ids or result in tok.slot_ids:
            return None
        return result, case

    def mask(self, units, slots):
        """One prefix anchor followed by one action per source surface span."""
        tok = self.tokenizer
        legal = torch.tensor([x == 0 for x in tok.additive_mask(slots)])
        for value in tok.control_ids:
            legal[value] = False  # Tag emissions never need NULL or other controls.
        mask = torch.zeros((len(units) + 1, self.size), dtype=torch.bool)
        mask[:, 2:2 + self.vocab] = legal
        mask[:, 2 + self.vocab:self.transform_offset] = legal
        mask[0, :] = False
        mask[0, self.KEEP] = True
        mask[0, 2 + self.vocab:self.transform_offset] = legal
        for i, (value, _) in enumerate(units):
            can_keep = 0 <= value < self.vocab and bool(legal[value])
            mask[i + 1, self.KEEP] = can_keep
            mask[i + 1, self.DELETE] = True
            # APPEND retains its anchor, so encoder-only words cannot use it.
            if not can_keep:
                mask[i + 1, 2 + self.vocab:self.transform_offset] = False
            for t, name in enumerate(TRANSFORMS):
                mask[i + 1, self.transform_offset + t] = self.transformed(units, i, name) is not None
        return mask

    def allowed(self, units, index, tag, slots):
        if not 0 <= tag < self.size:
            return False
        tok = self.tokenizer
        active = {tok.output_ids[s['token']] for s in slots}
        def ordinary_or_active(value):
            return 0 <= value < self.vocab and value not in tok.control_ids and (value not in tok.slot_ids or value in active)
        can_keep = index > 0 and ordinary_or_active(units[index - 1][0])
        if tag == self.KEEP:
            return index == 0 or can_keep
        if tag == self.DELETE:
            return index > 0
        if tag < 2 + self.vocab:
            return index > 0 and ordinary_or_active(tag - 2)
        if tag < self.transform_offset:
            return (index == 0 or can_keep) and ordinary_or_active(tag - 2 - self.vocab)
        return index > 0 and self.transformed(units, index - 1, TRANSFORMS[tag - self.transform_offset]) is not None

    def apply(self, units, tags, emitted_case, slots, require_slots=True):
        if len(tags) != len(units) + 1 or len(emitted_case) != len(tags):
            raise EncodingError('tag_shape')
        for i, tag in enumerate(tags):
            if not self.allowed(units, i, tag, slots):
                raise EncodingError('illegal_tag')
        result, skip = [], False
        for i, tag in enumerate(tags):
            if skip:
                if tag not in (self.KEEP, self.DELETE):
                    raise EncodingError('overlapping_merge')
                skip = False
                continue
            if tag == self.KEEP:
                if i:
                    result.append(units[i - 1])
            elif tag == self.DELETE:
                continue
            elif tag < 2 + self.vocab:
                result.append((tag - 2, emitted_case[i]))
            elif tag < self.transform_offset:
                if i:
                    result.append(units[i - 1])
                result.append((tag - 2 - self.vocab, emitted_case[i]))
            else:
                name = TRANSFORMS[tag - self.transform_offset]
                result.append(self.transformed(units, i - 1, name))
                skip = name == 'MERGE_NEXT'
        if not result:
            raise EncodingError('empty_prediction')
        if require_slots:
            self.tokenizer.restore_tokens([v for v, _ in result], slots, [c for _, c in result])
        return result


def source_units(encoded):
    """Pool raw BPE spans into atomic editing units, retaining their copy reference."""
    result = []
    for start, end, _, _ in encoded['source_spans']:
        refs = set(encoded['source_slot_refs'][start:end])
        if len(refs) != 1 or start >= end:
            raise EncodingError('invalid_span_refs')
        ref = next(iter(refs))
        if ref < 0 and end != start + 1:
            raise EncodingError('unreferenced_byte_span')
        result.append((ref if ref >= 0 else encoded['source_ids'][start],
                       0 if ref >= 0 else encoded['source_case'][start]))
    return result


def target_units(encoded, tokenizer):
    return [(value, 0 if value in tokenizer.slot_ids else case)
            for value, case in zip(encoded['target_ids'], encoded['target_case'])]


def checker_features(text, encoded, lexicon, mode):
    violations = check(text, lexicon, mode)
    sentences, cursor = [], 0
    for sentence in split_sentences(text):
        start = text.find(sentence, cursor)
        sentences.append((start, start + len(sentence)))
        cursor = start + len(sentence)
    features = torch.zeros((len(encoded['source_ids']) + 1, len(RULES)))
    for violation in violations:
        if violation.rule not in RULES or violation.severity == 'info':
            continue
        rule = RULES.index(violation.rule)
        features[0, rule] = 1
        for p, q, start, end in encoded['source_spans']:
            if violation.rule == 'R6.6' or (violation.sentence < len(sentences) and
                    start < sentences[violation.sentence][1] and end > sentences[violation.sentence][0]):
                features[p + 1:q + 1, rule] = 1
    return features


def collate(examples, tokenizer, lexicon, device='cpu'):
    """Examples contain src/mode plus encoded source features (e.g. ID-export rows)."""
    max_pieces = max(len(r['source_ids']) for r in examples) + 1
    max_spans = max(len(r['source_spans']) for r in examples) + 1
    n = len(examples)
    ids = torch.full((n, max_pieces), tokenizer.output_ids['PAD'], dtype=torch.long)
    cases = torch.zeros_like(ids)
    refs = torch.zeros_like(ids)
    padding = torch.ones_like(ids, dtype=torch.bool)
    spans = torch.zeros((n, max_spans, max_pieces))
    actions = torch.zeros((n, max_spans), dtype=torch.bool)
    features = torch.zeros((n, max_pieces, len(RULES)))
    tags = Tags(tokenizer)
    masks = torch.zeros((n, max_spans, tags.size), dtype=torch.bool)
    for i, row in enumerate(examples):
        length = len(row['source_ids']) + 1
        ids[i, :length] = torch.tensor([row['mode_id']] + row['source_ids'])
        cases[i, 1:length] = torch.tensor(row['source_case'])
        refs[i, 1:length] = torch.tensor([r + 1 for r in row['source_slot_refs']])
        padding[i, :length] = False
        spans[i, 0, 0] = 1
        for j, (p, q, _, _) in enumerate(row['source_spans'], 1):
            spans[i, j, p + 1:q + 1] = 1 / (q - p)
        count = len(row['source_spans']) + 1
        actions[i, :count] = True
        masks[i, :count] = tags.mask(source_units(row), row['slots'])
        masks[i, count:, tags.KEEP] = True  # Padded logits stay finite; loss ignores them.
        features[i, :length] = checker_features(row['src'], row, lexicon, row['mode'])
    return {k: v.to(device) for k, v in {'ids': ids, 'cases': cases, 'refs': refs,
            'padding': padding, 'spans': spans, 'actions': actions,
            'features': features, 'legal': masks}.items()}


class Tagger(nn.Module):
    def __init__(self, tokenizer, config=Config()):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(tokenizer.meta['input_size'], config.width, padding_idx=tokenizer.output_ids['PAD'])
        self.case = nn.Embedding(4, config.width)
        self.slot_ref = nn.Embedding(tokenizer.meta['output_size'] + 1, config.width, padding_idx=0)
        self.position = nn.Embedding(config.max_positions, config.width)
        self.violation = nn.Linear(len(RULES), config.width, bias=False)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, config.ffn,
                 config.dropout, activation='gelu', batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.tag_head = nn.Linear(config.width, Tags(tokenizer).size)
        self.case_head = nn.Linear(config.width, 3)

    def forward(self, batch):
        ids = batch['ids']
        if ids.shape[1] > self.config.max_positions:
            raise ValueError('position capacity exceeded')
        positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
        hidden = self.norm(self.token(ids) + self.case(batch['cases']) + self.slot_ref(batch['refs'])
                           + self.position(positions) + self.violation(batch['features']))
        hidden = self.encoder(hidden, src_key_padding_mask=batch['padding'])
        pooled = torch.bmm(batch['spans'], hidden)
        return self.tag_head(pooled).masked_fill(~batch['legal'], float('-inf')), self.case_head(pooled)


def loss(logits, case_logits, labels, cases, valid, tags, keep_weight=1.0, sample_weights=None):
    selected = valid & labels.ne(-100)
    if not selected.any():
        raise ValueError('no supervised positions')
    weights = torch.where(labels.eq(tags.KEEP), keep_weight, 1.0)
    if sample_weights is not None:
        weights = weights * sample_weights[:, None]
    tag_losses = F.cross_entropy(logits[selected], labels[selected], reduction='none')
    tag_loss = (tag_losses * weights[selected]).sum() / weights[selected].sum().clamp_min(1e-8)
    emitted = selected & labels.ge(2) & labels.lt(tags.transform_offset)
    if emitted.any():
        case_weights = torch.ones_like(weights) if sample_weights is None else sample_weights[:, None].expand_as(weights)
        case_loss = (F.cross_entropy(case_logits[emitted], cases[emitted], reduction='none') * case_weights[emitted]).sum() / case_weights[emitted].sum().clamp_min(1e-8)
    else:
        case_loss = case_logits.sum() * 0
    return tag_loss + case_loss


def supervision(units, targets, tags):
    """One deterministic edit pass; remaining insertions are handled in later passes."""
    targets = [(v, 0 if v in tags.tokenizer.slot_ids else c) for v, c in targets]
    edits, _, _ = align_ids(units, targets)
    labels, cases = [tags.KEEP] * (len(units) + 1), [0] * (len(units) + 1)
    for operation, i, j in edits:
        if operation == 'replace':
            value, case = targets[j]
            transform = next((name for name in TRANSFORMS[:-1] if tags.transformed(units, i, name) == targets[j]), None)
            labels[i + 1] = tags.transform(transform) if transform else tags.replace(value)
            cases[i + 1] = 0 if value in tags.tokenizer.slot_ids else case
        elif operation == 'delete':
            labels[i + 1] = tags.DELETE
        elif operation == 'insert':
            # An anchor has one action per pass. Defer extra or conflicting inserts.
            anchor = i
            if labels[anchor] == tags.KEEP and (not anchor or units[anchor - 1][0] < tags.vocab):
                labels[anchor] = tags.append(targets[j][0])
                cases[anchor] = 0 if targets[j][0] in tags.tokenizer.slot_ids else targets[j][1]
    consumed = set()
    for i in range(len(units) - 1):
        if i in consumed:
            continue
        merged = tags.transformed(units, i, 'MERGE_NEXT')
        if merged is None:
            continue
        value, case = merged
        a, b = labels[i + 1], labels[i + 2]
        if ((a == tags.replace(value) and cases[i + 1] == case and b == tags.DELETE) or
                (a == tags.DELETE and b == tags.replace(value) and cases[i + 2] == case)):
            labels[i + 1], labels[i + 2] = tags.transform('MERGE_NEXT'), tags.DELETE
            consumed.add(i + 1)
    return labels, cases


@torch.no_grad()
def rewrite(model, tokenizer, lexicon, text, mode='PROC', max_passes=5):
    if not 1 <= max_passes <= 5:
        raise ValueError('pass cap must be between 1 and 5')
    model.eval()
    device = next(model.parameters()).device
    tags = Tags(tokenizer)
    original = tokenizer.encode_source(text, mode)
    original_counts = Counter({s['surface']: s['count'] for s in original['slots']})
    seen = {text}
    for step in range(max_passes):
        encoded = tokenizer.encode_source(text, mode)
        row = dict(encoded, src=text, mode=mode)
        units = source_units(encoded)
        # The checker is a stop condition between edit passes. Run at least one
        # model pass: article omissions can be checker-clean at the input.
        batch = collate([row], tokenizer, lexicon, device)
        logits, case_logits = model(batch)
        chosen = logits[0].argmax(-1).tolist()
        case = case_logits[0].argmax(-1).tolist()
        try:
            output = tags.apply(units, chosen, case, encoded['slots'])
            values, casing = zip(*output)
            words = tokenizer.restore_tokens(list(values), encoded['slots'], list(casing))
            if Counter(w for w in words if w in original_counts) != original_counts:
                raise EncodingError('original_slot_fidelity')
            updated = tokenizer.detokenize(list(values), encoded['slots'], list(casing))
        except EncodingError as error:
            return {'text': None, 'passes': step + 1, 'resolved': False, 'reason': str(error)}
        clean = not errors(check(updated, lexicon, mode))
        if clean:
            return {'text': updated, 'passes': step + 1, 'resolved': True, 'reason': 'checker_clean'}
        if all(tag == tags.KEEP for tag in chosen) or updated in seen:
            return {'text': updated, 'passes': step + 1, 'resolved': False, 'reason': 'stalled'}
        seen.add(updated)
        text = updated
    return {'text': text, 'passes': max_passes, 'resolved': False, 'reason': 'pass_cap'}
