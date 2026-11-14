#!/usr/bin/env python3
"""Check whether the LCLM full-text early end comes from prefix cache reuse."""
import argparse
import copy
import json
from pathlib import Path

import torch

from memory import load_decoder, make_prefix, prefill, question_suffix, setup, text_embeddings

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--data", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
setup(8)
model, tokenizer = load_decoder(args.checkpoint)
question = json.loads((args.data / "questions.json").read_text())[0]["question"]
doc = (args.data / "original.txt").read_text()
suffix = question_suffix(question)
with torch.inference_mode():
    prefix = make_prefix(model, tokenizer, document=doc)
    cache = prefill(model, prefix)
    ids = torch.tensor([tokenizer.encode(suffix, add_special_tokens=False)])
    cached = model(input_ids=ids, past_key_values=copy.deepcopy(cache), use_cache=True, logits_to_keep=1).logits[0, -1].float()
    joined = torch.cat([prefix, text_embeddings(model, tokenizer, suffix)], dim=1)
    direct = model(inputs_embeds=joined, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    report = {"question": question, "finite": bool(torch.isfinite(cached).all() and torch.isfinite(direct).all()),
              "cached_argmax": int(cached.argmax()), "direct_argmax": int(direct.argmax()),
              "max_logit_difference": float((cached - direct).abs().max()),
              "top_tokens": [{"id": int(i), "token": tokenizer.decode([int(i)]), "probability": float(p)}
                             for p, i in zip(*direct.softmax(-1).topk(5))]}
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
