"""Deterministic, offline corruptions of clean STE. Derived output must stay ignored.

The reverse dictionary is a source of possible substitutions, not a word-sense oracle.
Every variant records its operation; none is claimed to have a human meaning review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from text_utils import match_case, normalized, protected_values, replace, tokenize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'checker'))
from checker import Lexicon  # noqa: E402


# Grammar only, not dictionary-derived data. Restrict voice changes to simple transitive clauses.
TRANSITIVE = set('add adjust apply attach change check clean close connect copy delete disconnect '
                 'find install keep lock move open read remove replace reset save select set '
                 'start stop test turn unlock update use write'.split())
IRREGULAR = {'find': 'found', 'keep': 'kept', 'read': 'read', 'set': 'set', 'write': 'written'}
DOUBLED = {'stop': 'stopped'}
BLOCKED = set('and or if when while because before after until unless then not never only '
              'always must can could should may might will would do does did please'.split())


def participle(verb: str) -> str:
    if verb in IRREGULAR:
        return IRREGULAR[verb]
    if verb in DOUBLED:
        return DOUBLED[verb]
    if verb.endswith('e'):
        return verb + 'd'
    if verb.endswith('y') and verb[-2] not in 'aeiou':
        return verb[:-1] + 'ied'
    return verb + 'ed'


def gerund(verb: str) -> str:
    if verb == 'stop':
        return 'stopping'
    if verb.endswith('e') and verb not in {'see', 'be'}:
        return verb[:-1] + 'ing'
    return verb + 'ing'


class Noiser:
    def __init__(self, lex_dir: Path = ROOT / 'lexicon'):
        self.lex = Lexicon(lex_dir)
        rows = [json.loads(line) for line in (lex_dir / 'lexicon.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        approved = {(r['lemma'], r['pos']) for r in rows if r['status'] == 'approved'}
        approved_pos = {}
        for lemma, pos in approved:
            approved_pos.setdefault(lemma, set()).add(pos)
        reverse = {}
        self.reverse_pos = {}
        for row in rows:
            # Limit the worst ambiguity: one alternative, same POS, no technical overrides.
            alts = row['alternatives']
            if row['status'] != 'rejected' or len(alts) != 1 or row['lemma'] in self.lex.allowed:
                continue
            alt = alts[0]
            if (alt['lemma'], alt['pos']) not in approved or alt['pos'] != row['pos']:
                continue
            # Leave function words and ambiguous noun/verb spellings alone.
            if alt['pos'] not in {'n', 'v', 'adj', 'adv', 'prep'} or len(approved_pos[alt['lemma']]) != 1:
                continue
            if alt['pos'] == 'v' and alt['lemma'] in {'be', 'do', 'have'}:
                continue
            if not re.fullmatch(r'[a-z]+(?: [a-z]+)*', row['lemma']):
                continue
            if not re.fullmatch(r'[a-z]+(?: [a-z]+)*', alt['lemma']):
                continue
            reverse.setdefault(tuple(alt['lemma'].split()), set()).add(row['lemma'])
            self.reverse_pos[tuple(alt['lemma'].split())] = alt['pos']
        self.reverse = {key: sorted(values) for key, values in sorted(reverse.items())}

    def candidates(self, text: str, mode: str) -> list[tuple[str, str]]:
        toks = tokenize(text)
        values = [t.text.lower() if not t.protected else '' for t in toks]
        edits = []
        for i, token in enumerate(toks):
            if token.protected:
                continue
            for size in (1, 2, 3, 4):
                key = tuple(values[i:i + size])
                if len(key) != size or key not in self.reverse:
                    continue
                if token.text[:1].isupper() and i > 0 and toks[i - 1].text not in {'.', '!', '?'}:
                    continue
                if self.reverse_pos[key] == 'v':
                    # Base-form substitutions only in imperatives and infinitives.
                    imperative = mode == 'PROC' and (i == 0 or toks[i - 1].text in {'.', '!'})
                    if not imperative and (i == 0 or values[i - 1] != 'to'):
                        continue
                last = toks[i + size - 1]
                # Only substitute word sequences separated by whitespace.
                old = text[token.start:last.end]
                if ' '.join(old.lower().split()) != ' '.join(key):
                    continue
                for word in self.reverse[key]:
                    edits.append(('reverse_dictionary', replace(text, token.start, last.end, match_case(old, word))))
            if values[i] in {'the', 'a', 'an'} and i + 1 < len(toks):
                # A capital A can be a variable, including at the start of a sentence.
                if token.text == 'A':
                    continue
                # Never remove text inside protected tokens, including code fences.
                end = toks[i + 1].start
                if text[token.end:end].isspace():
                    edits.append(('drop_article', replace(text, token.start, end, '')))
            for old, new in ((('before',), 'prior to'), (('to',), 'in order to'),
                             (('make', 'sure'), 'ensure')):
                if tuple(values[i:i + len(old)]) == old:
                    last = toks[i + len(old) - 1]
                    if text[token.start:last.end].lower() == ' '.join(old):
                        # "in order to" requires a verb, not a destination.
                        if old == ('to',) and (i + 1 >= len(toks) or values[i + 1] not in TRANSITIVE):
                            continue
                        edits.append(('verbose', replace(text, token.start, last.end, match_case(token.text, new))))

        # Merge complete assertions/instructions without changing a question into a statement.
        for i in range(len(toks) - 1):
            left, right = toks[i:i + 2]
            if left.text == '.' and not right.protected and right.text.isalpha() and text[left.end:right.start].isspace():
                # Preserve proper-name casing in descriptions; imperative verb casing is grammatical.
                head = right.text.lower() if mode == 'PROC' and right.text.lower() in TRANSITIVE else right.text
                edits.append(('merge_sentences', replace(text, left.start, right.end, ' and ' + head)))

        # Very narrow imperative transformations. Do not infer a subject or change tense in DESC.
        if mode == 'PROC':
            starts = [0] + [i + 1 for i, t in enumerate(toks[:-1]) if t.text in {'.', '!'}]
            for start in starts:
                stop = next((i for i in range(start, len(toks)) if toks[i].text in {'.', '?', '!'}), len(toks))
                clause = toks[start:stop]
                if len(clause) != 3 or values[start] not in TRANSITIVE or values[start + 1] not in {'the', 'a', 'an'}:
                    continue
                if stop < len(toks) and toks[stop].text == '?':
                    continue
                if not all(t.protected or t.text.isalpha() for t in clause):
                    continue
                if any(t.text.lower() in BLOCKED for t in clause[1:]):
                    continue
                verb, end = values[start], clause[-1].end
                obj = text[clause[1].start:end]
                obj_head = obj[:1].upper() + obj[1:]
                edits.append(('passive', replace(text, clause[0].start, end, obj_head + ' should be ' + participle(verb))))
                edits.append(('nominalize', replace(text, clause[0].start, end, 'Do the ' + gerund(verb) + ' of ' + obj)))
                edits.append(('verbose', replace(text, clause[0].start, clause[0].end, 'It is necessary to ' + verb)))

        # Past -> perfect is only attempted for an explicit unambiguous pronoun and regular past.
        if mode == 'DESC':
            for i in range(len(toks) - 1):
                if values[i] not in {'i', 'we', 'you', 'they', 'he', 'she', 'it'}:
                    continue
                # A pronoun after "get", "make", etc. can be an object rather than a subject.
                if i > 0 and toks[i - 1].text not in {'.', '!', '?', ','} and values[i - 1] not in {'and', 'but', 'so', 'that'}:
                    continue
                for verb in sorted(TRANSITIVE):
                    past = participle(verb)
                    if past.endswith('ed') and values[i + 1] == past:
                        aux = 'has' if values[i] in {'he', 'she', 'it'} else 'have'
                        edits.append(('perfect_tense', replace(text, toks[i + 1].start, toks[i + 1].end, aux + ' ' + past)))
        return edits

    def variants(self, text: str, mode: str, count: int = 2, seed: int = 0,
                 allowed_operations: set[str] | None = None) -> list[dict]:
        digest = hashlib.sha256(f'{seed}\0{mode}\0{text}'.encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], 'big'))
        # Sample across operations first, so the reverse dictionary cannot drown out structural noise.
        groups = {}
        seen = {normalized(text)}
        protected = protected_values(text)
        for operation, noisy in self.candidates(text, mode):
            if allowed_operations is not None and operation not in allowed_operations:
                continue
            key = normalized(noisy)
            if key in seen or protected_values(noisy) != protected:
                continue
            seen.add(key)
            groups.setdefault(operation, []).append(noisy)
        operations = sorted(groups)
        rng.shuffle(operations)
        for group in groups.values():
            rng.shuffle(group)
        result = []
        while operations and len(result) < count:
            for operation in operations[:]:
                result.append({'src': groups[operation].pop(), 'noise': operation})
                if not groups[operation]:
                    operations.remove(operation)
                if len(result) == count:
                    break
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='clean STE JSONL with id, ste, mode')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variants', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.variants < 0:
        ap.error('--variants must be nonnegative')
    if args.input.resolve() == args.out.resolve():
        ap.error('--out must differ from --input')
    noiser = Noiser()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open(encoding='utf-8') as source, args.out.open('w', encoding='utf-8', newline='\n') as out:
        for line in source:
            row = json.loads(line)
            for variant in noiser.variants(row['ste'], row['mode'], args.variants, args.seed):
                out.write(json.dumps({'id': row['id'], 'ste': row['ste'], 'mode': row['mode'], **variant}, ensure_ascii=False) + '\n')
                n += 1
    print(json.dumps({'rule_pairs': n}))


if __name__ == '__main__':
    main()
