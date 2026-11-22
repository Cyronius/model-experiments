"""Frozen Llama features -> small trainable resampler -> soft input memories."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    hidden_size: int
    width: int = 128
    slots: int = 16
    heads: int = 4
    max_tokens: int = 256
    embedding_rms: float = 0.02


class MemoryEncoder(nn.Module):
    """Shared across documents. It has no document-ID table and never sees a question."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.project = nn.Linear(config.hidden_size, config.width)
        self.positions = nn.Parameter(torch.randn(config.max_tokens, config.width) * 0.02)
        self.queries = nn.Parameter(torch.randn(config.slots, config.width) * 0.02)
        self.read = nn.MultiheadAttention(config.width, config.heads, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, config.width * 4), nn.GELU(),
                                nn.Linear(config.width * 4, config.width))
        self.output = nn.Linear(config.width, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        # Learn a scale per slot, initialized to the recipient's actual embedding scale.
        self.log_scale = nn.Parameter(torch.full((config.slots, 1), float(torch.tensor(config.embedding_rms).log())))

    def forward(self, features, mask):
        length = features.shape[1]
        if length > self.config.max_tokens:
            raise ValueError("Features exceed the encoder's position table")
        x = self.project(self.input_norm(features.float())) + self.positions[:length]
        query = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        slots = query + self.read(query, x, x, key_padding_mask=~mask.bool(), need_weights=False)[0]
        slots = slots + self.ff(self.norm(slots))
        return self.output_norm(self.output(slots)) * self.log_scale.exp()

    def specification(self):
        return asdict(self.config)


def prompt_parts(tokenizer, question, context=None, memory_site="prefix"):
    instruction = question.get("instruction", "Answer briefly.")
    content = f"{instruction}\n"
    marker = "<ENCODED_MEMORY_INSERTION_POINT>"
    if context is not None:
        content += f"Reference record:\n{context}\n\n"
    elif memory_site == "reference":
        content += f"Reference record:\n{marker}\n\n"
    content += "Question: " + question["question"]
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                             tokenize=False, add_generation_prompt=True)
    if memory_site == "reference" and context is None:
        if rendered.count(marker) != 1:
            raise ValueError("Memory insertion marker must occur exactly once")
        before, after = rendered.split(marker)
        return (tokenizer(before, add_special_tokens=False).input_ids,
                tokenizer(after, add_special_tokens=False).input_ids)
    # Put memory immediately after BOS, preserving the rest of the chat template.
    ids = tokenizer(rendered, add_special_tokens=False).input_ids
    cut = 1 if ids and ids[0] == tokenizer.bos_token_id else 0
    return ids[:cut], ids[cut:]


def make_batch(model, tokenizer, questions, memories=None, contexts=None, include_answers=True, memory_site="prefix"):
    """Right padding for teacher forcing. Only answer positions contribute to loss."""
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    rows, labels, masks = [], [], []
    eos = tokenizer.eos_token_id
    for i, question in enumerate(questions):
        before, after = prompt_parts(tokenizer, question, None if contexts is None else contexts[i],
                                     memory_site if memories is not None else "prefix")
        before_emb = embedding(torch.tensor(before, dtype=torch.long, device=device))
        after_emb = embedding(torch.tensor(after, dtype=torch.long, device=device))
        parts = [before_emb]
        if memories is not None:
            parts.append(memories[i].to(dtype=embedding.weight.dtype, device=device))
        parts.append(after_emb)
        prefix = torch.cat(parts, dim=0)
        if include_answers:
            target = tokenizer(question["answers"][0], add_special_tokens=False).input_ids
            if eos is not None:
                target.append(eos)
            answer_emb = embedding(torch.tensor(target, device=device))
            rows.append(torch.cat([prefix, answer_emb], dim=0))
            labels.append(torch.tensor([-100] * len(prefix) + target, device=device))
        else:
            rows.append(prefix)
            labels.append(torch.full((len(prefix),), -100, device=device))
        masks.append(torch.ones(len(rows[-1]), dtype=torch.long, device=device))
    return {"inputs_embeds": nn.utils.rnn.pad_sequence(rows, batch_first=True),
            "attention_mask": nn.utils.rnn.pad_sequence(masks, batch_first=True),
            "labels": nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)}


def answer_loss(model, batch):
    outputs = model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"], use_cache=False)
    logits, labels = outputs.logits[:, :-1], batch["labels"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=-100)


@torch.no_grad()
def greedy_answer(model, tokenizer, question, memory=None, context=None, max_new_tokens=12, memory_site="prefix"):
    batch = make_batch(model, tokenizer, [question], None if memory is None else [memory],
                       None if context is None else [context], include_answers=False, memory_site=memory_site)
    # Explicit prefill + cached decode avoids generation-version differences for inputs_embeds.
    output = model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"], use_cache=True)
    result = []
    stop = {tokenizer.eos_token_id}
    extra_stop = getattr(model.generation_config, "eos_token_id", None)
    stop.update(extra_stop if isinstance(extra_stop, list) else [extra_stop])
    mask = batch["attention_mask"]
    for i in range(max_new_tokens):
        token = output.logits[:, -1].argmax(-1)
        if token.item() in stop:
            break
        result.append(token.item())
        if i + 1 < max_new_tokens:
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
            output = model(input_ids=token[:, None], attention_mask=mask,
                           past_key_values=output.past_key_values, use_cache=True)
    return tokenizer.decode(result, skip_special_tokens=True).strip()
