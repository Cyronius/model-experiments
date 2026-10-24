"""Closed output vocabulary, input-only byte BPE, and request-local copy slots.

Derived vocabulary artifacts belong under data/out/, never in version control.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data'))
from build_pairs import align, BUCKETS
from text_utils import Token, tokenize

CONTROLS = ['PAD', 'NULL', 'SEP', 'PROC', 'DESC']
SLOTS = [f'{kind}_{i}' for kind, count in [('TN', 32), ('TV', 16), ('NUM', 16), ('PROTECT', 16)]
         for i in range(1, count + 1)]
PUNCTUATION = list(".,?!:;'\"()[]{}-–—…/+*=<>%&|#\\^~@$_“”‘’")
CASES = ('lower', 'title', 'upper', 'other')
FUNCTION_WORDS = set('''a an the this that these those each every some any no all both
either neither another other such what whatever which whichever who whoever whom whose
i me my mine myself we us our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
is are was were be do does did have has had can could may might must
shall should will would and or but nor so yet if unless because although though
whether while as than when where why how not of to in on at by for from with without
about above below under over between among through across along around before after
until since during into onto out up down off per'''.split())
NUMBER = re.compile(r'\d+(?:[.,:]\d+)*(?:%)?')
NAME_WORD = re.compile(r'(?:[A-Z][a-z]+(?:-[A-Za-z]+)*|[A-Z]{2,})')
UNITS = set('mm cm m km mg g kg lb lbs oz ms s min h Hz kHz MHz GHz V mV kV A mA '
            'W kW MW C K Pa kPa MPa bar psi N Nm rpm B KB MB GB TB KiB MiB GiB TiB '
            'bit bits byte bytes in ft yd'.split())
NAME_START_EXCLUDE = {word.capitalize() for word in FUNCTION_WORDS} | set('''Use Open Close
Set Read Write Run Build Create Check Verify Ensure Install Remove Add Delete Update
Copy Move Save Load Return Print Show List Make Keep Start Stop Please'''.split())


def model_tokens(text):
    """Add conservative name and number/unit spans to the shared surface detector."""
    raw, result, i = tokenize(text), [], 0
    while i < len(raw):
        token = raw[i]
        start, end, stop = token.start, token.end, i + 1
        # A sign/currency symbol directly attached to a number belongs to that number.
        numeric = bool(NUMBER.fullmatch(token.text))
        if token.text in ('-', '+', '$', '£', '€') and stop < len(raw):
            nxt = raw[stop]
            if token.end == nxt.start and NUMBER.fullmatch(nxt.text):
                numeric, end, stop = True, nxt.end, stop + 1
        if numeric:
            unit = stop
            if unit < len(raw) and raw[unit].text == '°':
                unit += 1
            if unit < len(raw) and raw[unit].text in UNITS:
                gap = text[end:raw[stop].start]
                between = text[raw[stop].end:raw[unit].start] if unit > stop else ''
                if re.fullmatch(r'[ \t]*', gap) and re.fullmatch(r'[ \t]*', between):
                    end, stop = raw[unit].end, unit + 1
            result.append(Token(text[start:end], start, end, True))
        elif not token.protected and NAME_WORD.fullmatch(token.text) and token.text not in NAME_START_EXCLUDE:
            while stop < len(raw) and not raw[stop].protected and NAME_WORD.fullmatch(raw[stop].text):
                if not re.fullmatch(r'[ \t]+', text[end:raw[stop].start]):
                    break
                end, stop = raw[stop].end, stop + 1
            result.append(Token(text[start:end], start, end, stop > i + 1))
        else:
            result.append(token)
        i = stop
    return result


def align_ids(source, target):
    """Trim copied edges before DP; the resulting global path still has minimum cost."""
    prefix = 0
    while prefix < min(len(source), len(target)) and source[prefix] == target[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(source), len(target)) - prefix and source[-suffix - 1] == target[-suffix - 1]:
        suffix += 1
    ns, nt = len(source) - suffix, len(target) - suffix
    middle, _, _ = align(source[prefix:ns], target[prefix:nt])
    edits = [['keep', i, i] for i in range(prefix)]
    edits += [[op, i + prefix, j + prefix] for op, i, j in middle]
    edits += [['keep', ns + i, nt + i] for i in range(suffix)]
    positions, previous = [], -1
    for op, i, j in edits:
        if op != 'delete':
            previous = max(previous + 1, 2 * i if op != 'insert' else max(0, 2 * i - 1))
            positions.append(previous)
    return edits, positions, max(2 * len(source), previous + 1)


class EncodingError(ValueError):
    """A pair/request cannot satisfy the closed-vocabulary or copy contract."""


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def case_id(text: str) -> int:
    if text == text.lower():
        return 0
    if text == text[:1].upper() + text[1:].lower():
        return 1
    if text.isupper():
        return 2
    return 3


def words_from_forms(forms):
    return {t.text.lower() for form in forms for t in tokenize(form)
            if not t.protected and any(c.isalpha() for c in t.text)}


def build_vocabulary(lex_dir: Path, training_texts, output: Path, bpe_size=4096):
    """Fit fallback on training sources only. No dev/test text participates."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    sys.path.insert(0, str(ROOT / 'checker'))
    from checker import Lexicon
    lex = Lexicon(lex_dir)
    rows = [json.loads(line) for line in (lex_dir / 'lexicon.jsonl').read_text(encoding='utf-8').splitlines() if line]
    allowed = words_from_forms(lex.allowed) | FUNCTION_WORDS
    rejected = words_from_forms(form for row in rows if row['status'] != 'approved'
                                for form in row['forms']) - allowed
    # The converted dictionary often supplies only a rejected lemma. These
    # conservative regular forms are encoder-only; they never approve output.
    for row in rows:
        word = row['lemma']
        if row['status'] == 'approved' or not re.fullmatch('[a-z]+', word):
            continue
        if row['pos'] in ('n', 'v'):
            rejected.add(word + ('es' if word.endswith(('s', 'x', 'z', 'ch', 'sh')) else 's'))
            if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
                rejected.add(word[:-1] + 'ies')
        if row['pos'] == 'v':
            rejected.add(word + 'd' if word.endswith('e') else word + 'ed')
            rejected.add(word[:-1] + 'ing' if word.endswith('e') and not word.endswith('ee') else word + 'ing')
            if word.endswith('y') and len(word) > 1 and word[-2] not in 'aeiou':
                rejected.add(word[:-1] + 'ied')
    out_tokens = CONTROLS + SLOTS + sorted(set(PUNCTUATION) | allowed)
    in_tokens = out_tokens + sorted(rejected - set(out_tokens))
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=bpe_size, min_frequency=2, show_progress=False,
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # BPE sees individual surface tokens, as it will at encoding time.
    bpe.train_from_iterator((t.text for text in training_texts for t in tokenize(text)), trainer)
    output.mkdir(parents=True, exist_ok=True)
    bpe.save(str(output / 'fallback.json'))
    metadata = {'schema': 'ste-vocabulary-v1', 'output_tokens': out_tokens,
                'input_word_tokens': in_tokens, 'bpe_offset': len(in_tokens),
                'input_size': len(in_tokens) + bpe.get_vocab_size(),
                'output_size': len(out_tokens), 'case_features': list(CASES),
                'fallback_sha256': sha256(output / 'fallback.json'),
                'lexicon_sha256': {name: sha256(lex_dir / name) for name in ('lexicon.jsonl', 'tech_words.txt')
                                   if (lex_dir / name).exists()},
                'code_sha256': sha256(Path(__file__)),
                'function_words': sorted(FUNCTION_WORDS),
                'fit_scope': 'training_sources_only'}
    (output / 'vocab.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return metadata


@dataclass
class Slot:
    token: str
    surface: str
    count: int


class SteTokenizer:
    def __init__(self, directory: Path):
        from tokenizers import Tokenizer
        self.meta = json.loads((directory / 'vocab.json').read_text(encoding='utf-8'))
        if sha256(directory / 'fallback.json') != self.meta['fallback_sha256']:
            raise ValueError('fallback vocabulary fingerprint mismatch')
        self.output = self.meta['output_tokens']
        self.output_ids = {word: i for i, word in enumerate(self.output)}
        self.input_ids = {word: i for i, word in enumerate(self.meta['input_word_tokens'])}
        self.bpe = Tokenizer.from_file(str(directory / 'fallback.json'))
        self.bpe_offset = self.meta['bpe_offset']
        self.slot_ids = {self.output_ids[word] for word in SLOTS}
        self.control_ids = {self.output_ids[word] for word in CONTROLS}

    @lru_cache(maxsize=65536)
    def fallback(self, text):
        ids = tuple(i + self.bpe_offset for i in self.bpe.encode(text).ids)
        if not ids or self.bpe.decode([i - self.bpe_offset for i in ids]) != text:
            raise EncodingError('fallback_roundtrip')
        return ids

    def slot_kind(self, token):
        if token.protected:
            if re.match(r'^[+\-$£€]?\d', token.text):
                return 'NUM' if NUMBER.fullmatch(token.text) or ' ' in token.text or token.text[:1] in '+-$£€' else 'PROTECT'
            if len(token.text.split()) > 1 and all(NAME_WORD.fullmatch(word) for word in token.text.split()):
                return 'TN'
            return 'PROTECT'
        # A capital letter can denote a variable, including the article "A".
        if len(token.text) == 1 and token.text.isupper():
            if token.text != 'I':
                return 'PROTECT'
        if case_id(token.text) == 3 and any(c.isalpha() for c in token.text):
            return 'PROTECT'
        if token.text.lower() not in self.input_ids:
            return 'PROTECT'
        return None

    def encode_source(self, text: str, mode='PROC', slot_probability=1.0, seed=0):
        if mode not in {'PROC', 'DESC'} or not 0 <= slot_probability <= 1:
            raise ValueError('invalid mode or slot probability')
        tokens = model_tokens(text)
        if not tokens:
            raise EncodingError('empty_source')
        # Build this table exclusively from input, before inspecting any target.
        table, kinds, counts = {}, Counter(), Counter()
        for token in tokens:
            kind = self.slot_kind(token)
            if kind:
                if token.text not in table:
                    kinds[kind] += 1
                    name = f'{kind}_{kinds[kind]}'
                    if name not in self.output_ids:
                        raise EncodingError('slot_overflow_' + kind)
                    table[token.text] = name
                counts[token.text] += 1
        slots = [Slot(name, surface, counts[surface]) for surface, name in table.items()]
        ids, cases, refs, spans = [], [], [], []
        for index, token in enumerate(tokens):
            slot = table.get(token.text)
            ref = self.output_ids[slot] if slot else -1
            key = f'{seed}\0{text}\0{index}'.encode('utf-8')
            selected = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64 < slot_probability
            if slot and selected:
                pieces = [ref]
            elif slot:
                # Raw training exposure still carries a slot reference at each piece;
                # the output can copy the span without learning to generate its bytes.
                pieces = self.fallback(token.text)
            else:
                pieces = [self.input_ids[token.text.lower()]]
            start = len(ids)
            ids.extend(pieces)
            cases.extend([case_id(token.text)] * len(pieces))
            refs.extend([ref] * len(pieces))
            spans.append([start, len(ids), token.start, token.end])
        if 2 * len(ids) > BUCKETS[-1]:
            raise EncodingError('source_bucket_overflow')
        return {'source_ids': ids, 'source_case': cases, 'source_slot_refs': refs,
                'source_spans': spans, 'slots': [asdict(slot) for slot in slots],
                'mode_id': self.output_ids[mode]}

    def encode_pair(self, row, slot_probability=.7, seed=0):
        encoded = self.encode_source(row['src'], row['mode'], slot_probability, seed)
        table = {slot['surface']: self.output_ids[slot['token']] for slot in encoded['slots']}
        target, cases = [], []
        for token in model_tokens(row['ste']):
            if token.text in table:
                value = table[token.text]
            elif self.slot_kind(token):
                raise EncodingError('target_unseen_slot')
            elif token.text.lower() in self.output_ids:
                value = self.output_ids[token.text.lower()]
            else:
                raise EncodingError('target_outside_vocabulary')
            target.append(value)
            cases.append(case_id(token.text))
        if not target:
            raise EncodingError('empty_target')
        restored = self.restore_tokens(target, encoded['slots'], cases)
        if restored != [token.text for token in model_tokens(row['ste'])]:
            raise EncodingError('target_roundtrip')
        edits, positions, required = align_ids(encoded['source_ids'], target)
        bucket = next((size for size in BUCKETS if size >= required), None)
        if bucket is None:
            raise EncodingError('target_bucket_overflow')
        encoded.update(target_ids=target, target_case=cases, target_positions=positions,
                       edits=edits, bucket=bucket)
        return encoded

    def additive_mask(self, slots):
        """NULL and ordinary output words/punctuation plus this request's slots."""
        active = {self.output_ids[slot['token']] for slot in slots}
        return [float('-inf') if (i in self.slot_ids and i not in active)
                or (i in self.control_ids and token != 'NULL') else 0.0
                for i, token in enumerate(self.output)]

    def restore_tokens(self, ids, slots, cases=None):
        table = {self.output_ids[slot['token']]: slot for slot in slots}
        if cases is not None and len(cases) != len(ids):
            raise EncodingError('case_length')
        counts, restored = Counter(), []
        for index, value in enumerate(ids):
            if not isinstance(value, int) or not 0 <= value < len(self.output):
                raise EncodingError('invalid_output_id')
            word = self.output[value]
            if word == 'NULL':
                continue
            if value in self.slot_ids:
                if value not in table:
                    raise EncodingError('inactive_slot')
                word = table[value]['surface']
                counts[value] += 1
            elif value in self.control_ids:
                raise EncodingError('output_control_token')
            elif cases is not None:
                case = cases[index]
                if case not in range(len(CASES)):
                    raise EncodingError('invalid_case')
                word = word[:1].upper() + word[1:] if case == 1 else word.upper() if case == 2 else word
            restored.append(word)
        if counts != Counter({key: slot['count'] for key, slot in table.items()}):
            raise EncodingError('slot_fidelity')
        return restored

    def detokenize(self, ids, slots, cases=None):
        """Restore protected strings atomically; normalize only prose spacing/case."""
        words = self.restore_tokens(ids, slots, cases)
        result, previous, sentence_start = '', '', True
        protected = {slot['surface'] for slot in slots}
        for word in words:
            if sentence_start and cases is None and word not in protected:
                word = word[:1].upper() + word[1:]
            attach = word in '.,?!:;)]}’' or previous in ('(', '[', '{', '“')
            result += ('' if not result or attach else ' ') + word
            sentence_start = word in ('.', '?', '!')
            previous = word
        return result
