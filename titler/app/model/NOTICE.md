# titler-v2-q4_k_m.gguf

A 5.84M-parameter Llama model (8 layers, 4,352-token vocabulary), quantized to Q4_K_M. It turns a prompt into a short title.

## Training data

Prompts only (no model replies) came from these public datasets. Titles for them were written by Qwen3-14B
(Apache-2.0) and, for a few thousand rows, by OpenAI's gpt-6-luna. The general-text stage used FineWeb-Edu.

| Dataset | License | Note |
|---|---|---|
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M), [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) | ODC-BY 1.0 | Attribution: WildChat, Allen Institute for AI. |
| [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | CC-BY-4.0 for user prompts | Only the user prompts were used (the model replies are CC-BY-NC-4.0). |
| [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | LMSYS-Chat-1M Dataset License Agreement | Allows research and commercial use and training models. It forbids passing on the dataset. This repository contains no part of the dataset. |
| [OpenAssistant/oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | ODC-BY 1.0 | General-text stage. |

Rows about sexual content with minors were removed before training.

This file records what the dataset pages say on 2026-10-19. It is not legal advice.
