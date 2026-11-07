"""Promote explicitly approved words, retaining train-only BPE and remapping weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from tagger import ROOT, Tags, SteTokenizer


def migrate(old_vocab, checkpoint, destination):
    destination.mkdir(parents=True, exist_ok=False)
    old = SteTokenizer(old_vocab)
    meta = dict(old.meta)
    promoted = ['file', 'files', 'delete', 'deletes', 'deleted', 'existing', 'otherwise']
    from checker import Lexicon
    from ste_tokenizer import CONTROLS, SLOTS, sha256
    approved = Lexicon().allowed
    if any(w not in approved for w in promoted):
        raise ValueError('promotion absent from explicit software whitelist')
    words = sorted(set(old.output[len(CONTROLS + SLOTS):]) | set(promoted))
    output = CONTROLS + SLOTS + words
    inputs = output + sorted(set(old.meta['input_word_tokens']) - set(output))
    if set(inputs) != set(old.meta['input_word_tokens']):
        raise ValueError('migration requires existing input embeddings')
    meta.update(output_tokens=output, input_word_tokens=inputs, output_size=len(output),
                parent_vocab_sha256=sha256(old_vocab / 'vocab.json'),
                code_sha256=sha256(ROOT / 'tokenizer/ste_tokenizer.py'),
                migration='explicit software promotions; inherited train-only BPE')
    meta['lexicon_sha256'] = {name: sha256(ROOT / 'lexicon' / name) for name in meta['lexicon_sha256']}
    shutil.copyfile(old_vocab / 'fallback.json', destination / 'fallback.json')
    (destination / 'vocab.json').write_text(json.dumps(meta, indent=2) + '\n')
    new = SteTokenizer(destination)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = dict(ckpt['model'])
    state['token.weight'] = torch.cat([state['token.weight'][torch.tensor([old.input_ids[w] for w in inputs])],
                                      state['token.weight'][old.bpe_offset:]], 0)
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
                  migration={'promoted': promoted, 'new_emissions_require_training': True})
    torch.save(result, destination / 'model.pt')
    (destination / 'migration.json').write_text(json.dumps({k:v for k,v in result.items() if k != 'model'}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vocab', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    migrate(args.vocab, args.checkpoint, args.output)
