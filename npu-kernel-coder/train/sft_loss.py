"""Loss on the answer tokens only, without materializing full-vocabulary logits for the prompt.

Qwen3.6's vocabulary is 248,320 tokens, so logits for a 16k-token sequence are 16 GB in fp32
(the first LoRA check ran out of memory on exactly that). The answer is the tail of each
example, so only its positions go through lm_head, in chunks, each recomputed in backward.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def answer_loss(model, input_ids: torch.Tensor, n_answer: int, chunk: int = 1024) -> torch.Tensor:
    """Mean next-token cross-entropy over the last `n_answer` tokens of input_ids [1, T]."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.model(input_ids=input_ids).last_hidden_state[0, -(n_answer + 1):-1]
    targets = input_ids[0, -n_answer:]

    def piece(h, t):
        return F.cross_entropy(base.lm_head(h).float(), t, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, n_answer, chunk):
        total = total + checkpoint(piece, hidden[i:i + chunk], targets[i:i + chunk], use_reentrant=False)
    return total / n_answer
