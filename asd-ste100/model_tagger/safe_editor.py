"""Conservative inference with bounded, feedback-driven proposal retries.

Resolved means closed-vocabulary and checker-clean with these deterministic guards;
it is not a certification of meaning or full ASD-STE100 compliance.
"""
from collections import Counter
from dataclasses import dataclass
import re
import torch
from tagger import Tags, source_units, collate, EncodingError, check, errors
from quality_gates import unsafe_changes
from protected_regions import regions
from reviewed_rewrites import canonical_words

ARTICLES = {'a', 'an', 'the'}
# Only explicitly reviewed lexical equivalences; no automatic dictionary-sense swaps.
EQUIVALENTS = {'utilize': 'use', 'utilizes': 'uses', 'utilized': 'used'}


def content(text):
    chars = list(text)
    for a, b in regions(text):
        chars[a:b] = ' ' * (b - a)
    words=canonical_words(re.findall(r"[a-z]+(?:['’][a-z]+)?", ''.join(chars).lower()))
    return tuple(EQUIVALENTS.get(w,w) for w in words if w not in ARTICLES)


def validate(source, candidate):
    reasons = [issue['rule'] for issue in unsafe_changes(source, candidate)]
    # Compare literal multisets as well as syntax signatures, including quotes,
    # multi-backticks, language labels and names/numbers detected by encoding.
    if Counter(source[a:b] for a, b in regions(source)) != Counter(candidate[a:b] for a, b in regions(candidate)):
        reasons.append('literal_changed')
    prose=list(candidate)
    for a,b in regions(candidate):prose[a:b]=' '*(b-a)
    if re.search(r'\b(a|an|the)\s+\1\b',''.join(prose),re.I):
        reasons.append('duplicate_article')
    if content(source) != content(candidate):
        reasons.append('content_changed')
    return reasons


@dataclass
class Proposal:
    text: str | None
    actions: tuple = ()
    reason: str | None = None


def run_loop(text, propose, is_clean, max_passes=5):
    """A rejected state never becomes the next input. Feedback bans its actions."""
    if not 1 <= max_passes <= 5:
        raise ValueError('pass cap must be between 1 and 5')
    original, current, trace, banned, seen = text, text, [], set(), {text}
    for attempt in range(1, max_passes + 1):
        proposal = propose(current, frozenset(banned))
        reasons = [proposal.reason] if proposal.reason else []
        if proposal.text is not None:
            reasons += validate(original, proposal.text)
        elif not reasons:
            reasons = ['no_candidate']
        trace.append(dict(attempt=attempt, input=current, proposal=proposal.text,
                          rejected=reasons, actions=list(proposal.actions)))
        if reasons:
            additions = set(proposal.actions) - banned
            banned.update(additions)
            if not additions:
                return dict(text=current, passes=attempt, resolved=False, reason='exhausted', trace=trace)
            continue
        candidate = proposal.text
        if is_clean(candidate):
            return dict(text=candidate, passes=attempt, resolved=True, reason='checker_clean', trace=trace)
        if candidate in seen:
            return dict(text=current, passes=attempt, resolved=False, reason='stalled', trace=trace)
        current = candidate
        seen.add(current)
        banned.clear()  # Action positions are scoped to exactly this input state.
    return dict(text=current, passes=max_passes, resolved=False, reason='pass_cap', trace=trace)


@torch.no_grad()
def rewrite(model, tokenizer, lexicon, text, mode='PROC', max_passes=5, edit_margin=4.0):
    if edit_margin < 0:
        raise ValueError('edit margin must be nonnegative')
    model.eval()
    device = next(model.parameters()).device
    tags = Tags(tokenizer)

    def propose(current, banned):
        chosen_actions = ()
        try:
            encoded = tokenizer.encode_source(current, mode)
            units = source_units(encoded)
            batch = collate([dict(encoded, src=current, mode=mode)], tokenizer, lexicon, device)
            scores, casing = model(batch)
            scores = scores[0].clone()
            # Freeze slot anchors: duplication elsewhere is still checked by apply.
            for i, (value, _) in enumerate(units, 1):
                if value in tokenizer.slot_ids:
                    scores[i, :] = float('-inf')
                    scores[i, tags.KEEP] = 0
            for i, action in banned:
                scores[i, action] = float('-inf')
            scores[:, tags.KEEP] += edit_margin
            if not torch.isfinite(scores).any(-1).all():
                return Proposal(None, reason='no_legal_candidate')
            chosen = scores.argmax(-1).tolist()
            chosen_actions = tuple((i, action) for i, action in enumerate(chosen) if action != tags.KEEP)
            cases = casing[0].argmax(-1).tolist()
            output = tags.apply(units, chosen, cases, encoded['slots'])
            values, cases = zip(*output)
            candidate = tokenizer.detokenize(list(values), encoded['slots'], list(cases))
            # Keep the exact original bytes for a no-op, avoiding detokenizer churn.
            if not chosen_actions:
                candidate = current
            return Proposal(candidate, chosen_actions)
        except EncodingError as exc:
            return Proposal(None, chosen_actions, str(exc))

    return run_loop(text, propose, lambda candidate: not errors(check(candidate, lexicon, mode)), max_passes)


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from tagger import SteTokenizer, Tagger, Config, Lexicon
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('text')
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--mode', choices=('PROC', 'DESC'), default='PROC')
    args = parser.parse_args()
    torch.set_num_threads(4)
    tok = SteTokenizer(args.vocabulary)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    actual = hashlib.sha256((args.vocabulary / 'vocab.json').read_bytes()).hexdigest()
    if checkpoint['vocab_sha256'] != actual:
        raise ValueError('checkpoint/vocabulary fingerprint mismatch')
    policy = json.loads(args.policy.read_text())
    if policy['vocab_sha256'] != actual or policy['checkpoint_sha256'] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
        raise ValueError('policy/checkpoint/vocabulary fingerprint mismatch')
    model = Tagger(tok, Config(**checkpoint['model_config']))
    model.load_state_dict(checkpoint['model'])
    result = rewrite(model, tok, Lexicon(), args.text, args.mode,
                     max_passes=policy['max_passes'], edit_margin=policy['edit_margin'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
