---
license: apache-2.0
language:
- en
pipeline_tag: summarization
tags:
- gguf
- llama
- title-generation
- summarization
- distillation
- tiny-model
- edge
- cpu
- onnx
- t5
- transformers.js
datasets:
- allenai/WildChat-1M
- OpenAssistant/oasst2
- lmsys/chatbot_arena_conversations
- lmsys/lmsys-chat-1m
- HuggingFaceFW/fineweb-edu
---

# titler

A tiny model that turns a prompt into a short title.

```text
how do I sort a list of dicts by a key in python
→ Sorting Dicts By Key in Python
```

This repository holds two models:

| | file | size | parameters | architecture | status |
|---|---|---:|---:|---|---|
| **v2** | `titler-v2-q4_k_m.gguf` | **4.1 MB** | 5.84M | Llama, 8 layers, 4,352-token vocabulary, Q4_K_M | **current** |
| v1.5 | `model.safetensors`, `onnx/`, `gguf/` | 5–15 MB | 7.61M | T5 encoder-decoder, 3 + 3 layers, 8,192-token vocabulary | legacy, kept for existing users |

**Source code, apps and write-ups:** https://github.com/Cyronius/model-experiments/tree/main/titler
**Try v2 in your browser** (runs locally with WebAssembly, nothing is sent anywhere): https://cyronius.github.io/model-experiments/titler/

## v2

v2 is a standard Llama-architecture GGUF. Any llama.cpp-based runtime can load it. It is meant to be used through the titler engine, which adds the tokenizer handling, beam search and title cleanup that give the results below.

- **4.1 MB** file, **5.84M** parameters, Q4_K_M.
- Runs on the CPU, offline, with no telemetry. About 25 MB of memory after loading.
- Output is 3 to 8 words in Title Case. The same input always gives the same output.
- Reads the first 256 tokens of the input (about 1,000 characters of English).
- **English only.** Other languages are not supported.

### Speed

Default setting (4 beams). With 1 beam, time is about half and titles are slightly worse.

| runtime | typical | slowest 5% (long prompts) |
|---|---:|---:|
| llama.cpp | 18 ms | 48 ms |
| Native Rust, one thread | 40–70 ms | 200–275 ms |
| Browser (WebAssembly) | 50–175 ms | |

### Quality

Judged pairwise against v1.5, the v2 title wins about **65%** of the pairs that have a winner.

### How to use it

The titler engine is in the GitHub repository. See [EMBEDDING.md](https://github.com/Cyronius/model-experiments/blob/main/titler/app/EMBEDDING.md) for every option. The quick versions:

| your app is | use |
|---|---|
| Rust | the `titler` crate |
| C, C++, C#, Go, Swift, Java | the C library, from the [releases](https://github.com/Cyronius/model-experiments/releases) |
| Python | the `titler` wheel, from the releases |
| A web page or Node | the `titler-wasm` npm package, from the releases |
| A shell | the `titler` command, from the releases |

```python
from titler import Titler
t = Titler()   # the model is inside the wheel; or Titler("titler-v2-q4_k_m.gguf")
print(t.title("how do I sort a list of dicts by a key in python"))
```

To use the GGUF file directly, you must write your own tokenization and cleanup. The vocabulary is in the GGUF. Without the titler engine, expect titles that differ from the ones measured above.

### Limits

- It titles. It does not summarize, translate or answer.
- It sometimes drops an apostrophe in a possessive (`Nick Wildes Judy Hopp Dialog Line`). This happened in 1 of 1,064 evaluation titles.
- Hard inputs, such as pasted logs and vague one-liners, give weaker titles than ordinary prompts.

### Training data

Prompts (no model replies) came from WildChat, OpenAssistant (oasst2), Chatbot Arena and LMSYS-Chat-1M. A general-text stage used FineWeb-Edu. Qwen3-14B wrote the training titles, and OpenAI's gpt-6-luna wrote a few thousand more. Rows about sexual content with minors were removed. The dataset licenses are listed in [NOTICE.md](NOTICE.md). Those datasets are not part of this repository.

## v1.5 (legacy)

The earlier model: a 7.61M-parameter T5 trained from scratch on about 481k prompt → title pairs. It is still here so existing links keep working. New projects should use v2.

| Metric | v1.5 |
| --- | ---: |
| Parameters | 7.61M |
| int8 ONNX size | 10.3 MB |
| CPU latency | ~160–220 ms |
| ROUGE-L vs teacher | 0.513 |
| Format compliance | 99.8% |

GGUF builds of v1.5 are in `gguf/`, measured against the f16 baseline on a 100-prompt held-out set (greedy, temp 0). LM Studio and Ollama do not run this T5 model correctly. Use llama.cpp.

| Quant | Size | ROUGE-L vs teacher | Outputs differing from f16 |
| --- | ---: | ---: | --- |
| F16 | 15.4 MB | 0.484 | baseline |
| Q8_0 | 8.3 MB | 0.485 | 6/100, paraphrases only |
| Q6_K | 6.4 MB | 0.490 | 11/100, paraphrases only |
| Q4_0 | 5.0 MB | 0.465 | 29/100, mostly paraphrases, a few real topic drifts |

Browser or Node, with Transformers.js (WASM backend):

```javascript
import { pipeline } from "@huggingface/transformers";

const titler = await pipeline("text2text-generation", "Cyronius/titler", { dtype: "q8" });
const result = await titler("how do I center a div with flexbox", {
  max_new_tokens: 32,
  num_beams: 2,
  no_repeat_ngram_size: 2
});
console.log(result[0].generated_text);
```

Python, with ONNX Runtime:

```python
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Cyronius/titler")
model = ORTModelForSeq2SeqLM.from_pretrained("Cyronius/titler", subfolder="onnx", use_merged=True, use_cache=True)
ids = tok("how do I center a div with flexbox", return_tensors="pt", truncation=True, max_length=512)
out = model.generate(**ids, max_new_tokens=32, num_beams=2, no_repeat_ngram_size=2)
print(tok.decode(out[0], skip_special_tokens=True))
```

The full history of v1.5 and the versions before it, including what was tried and rejected, is in [EXPERIMENTS.md](https://github.com/Cyronius/model-experiments/blob/main/titler/EXPERIMENTS.md).

## License

Code and model weights, v2 and v1.5, are dual-licensed. Use them under either the **MIT License** or the **Apache License 2.0**, at your option. The metadata above lists Apache-2.0, one of the two. Copyright (c) 2026 Cyrus Attoun.

The training data comes from public datasets with their own licenses. See [NOTICE.md](NOTICE.md).
