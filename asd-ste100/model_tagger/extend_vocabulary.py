"""Extend a vocabulary and remap a checkpoint: new output words and new input-only words.

Like migrate_vocabulary.py, but general: the new output words are the approved words
(lexicon plus tech_words.txt) that the old output lacks, and the new input-only words are
`grammar_input_words` (contractions, been/being/am, -ing forms of approved and software
verbs) that the old input lacks. The train-only byte BPE is inherited. Existing
parameters are kept by token identity; a new input word starts from the mean of the
embeddings of its BPE pieces (how the encoder saw it as a raw copied span); new emission
rows are suppressed until trained, as before.
"""
import argparse
import json
from pathlib import Path
import shutil
import torch
from tagger import ROOT, Tags, SteTokenizer


def extend(old_vocab, checkpoint, destination):
    from checker import Lexicon
    from ste_tokenizer import CONTROLS, SLOTS, PUNCTUATION, FUNCTION_WORDS, grammar_input_words, sha256, words_from_forms
    destination.mkdir(parents=True, exist_ok=False)
    old = SteTokenizer(old_vocab)
    lex_dir = ROOT / 'lexicon'
    allowed = words_from_forms(Lexicon(lex_dir).allowed) | FUNCTION_WORDS | set(PUNCTUATION)
    old_words = set(old.output[len(CONTROLS + SLOTS):])
    if old_words - allowed:
        raise ValueError('old output words no longer approved: ' + ', '.join(sorted(old_words - allowed)))
    promoted = sorted(allowed - old_words)
    output = CONTROLS + SLOTS + sorted(old_words | set(promoted))
    added_inputs = sorted(grammar_input_words(lex_dir) - set(old.meta['input_word_tokens']) - set(output))
    inputs = output + sorted((set(old.meta['input_word_tokens']) | set(added_inputs)) - set(output))
    meta = dict(old.meta)
    meta.update(output_tokens=output, input_word_tokens=inputs, output_size=len(output), bpe_offset=len(inputs),
                input_size=len(inputs) + old.bpe.get_vocab_size(), parent_vocab_sha256=sha256(old_vocab / 'vocab.json'),
                code_sha256=sha256(ROOT / 'tokenizer/ste_tokenizer.py'),
                migration='tech_words promotions and input-only grammar words; inherited train-only BPE')
    meta['lexicon_sha256'] = {name: sha256(lex_dir / name) for name in meta['lexicon_sha256']}
    shutil.copyfile(old_vocab / 'fallback.json', destination / 'fallback.json')
    (destination / 'vocab.json').write_text(json.dumps(meta, indent=2) + '\n')
    new = SteTokenizer(destination)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = dict(ckpt['model'])
    table = state['token.weight']
    rows = []
    for word in inputs:
        if word in old.input_ids:
            rows.append(table[old.input_ids[word]])
        else:
            rows.append(table[torch.tensor(old.fallback(word))].mean(0))
    state['token.weight'] = torch.cat([torch.stack(rows), table[old.bpe_offset:]], 0)
    refs = torch.zeros(len(output) + 1, state['slot_ref.weight'].shape[1])
    refs[0] = state['slot_ref.weight'][0]
    for i, word in enumerate(output):
        if word in old.output_ids:
            refs[i + 1] = state['slot_ref.weight'][old.output_ids[word] + 1]
    state['slot_ref.weight'] = refs
    a, b = Tags(old), Tags(new)
    for name in ('weight', 'bias'):
        source = state['tag_head.' + name]
        target = source.new_zeros((b.size,) + source.shape[1:])
        if name == 'bias':
            target.fill_(-20)  # New emission rows need training, never random confident output.
        target[:2] = source[:2]
        for i, word in enumerate(output):
            if word in old.output_ids:
                j = old.output_ids[word]
                target[b.replace(i)] = source[a.replace(j)]
                target[b.append(i)] = source[a.append(j)]
        target[b.transform_offset:] = source[a.transform_offset:]
        state['tag_head.' + name] = target
    result = dict(model=state, model_config=ckpt['model_config'],
                  parent_checkpoint_sha256=sha256(checkpoint), vocab_sha256=sha256(destination / 'vocab.json'),
                  migration={'promoted': promoted, 'added_input_words': added_inputs,
                             'new_emissions_require_training': True})
    torch.save(result, destination / 'model.pt')
    (destination / 'migration.json').write_text(json.dumps({k: v for k, v in result.items() if k != 'model'}, indent=2) + '\n')
    return result['migration']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    found = extend(args.vocab, args.checkpoint, args.output)
    print(json.dumps({'promoted': len(found['promoted']), 'added_input_words': len(found['added_input_words'])}))
