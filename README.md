# model-experiments

Small, cheap experiments with language models, mostly aimed at shared-memory hardware (CPU and iGPU boxes, no discrete GPU). Each experiment gets one folder. Failures are written up as carefully as the wins, because they hold most of the useful information.

Website: **https://cyronius.github.io/model-experiments/** (summary of the results, with the titler demo at [`/titler/`](https://cyronius.github.io/model-experiments/titler/)).

## Successful experiments

| result | what it is | size | links |
|---|---|---:|---|
| **Qwen3.6-27B-A2.8B** | Qwen3.6-35B-A3B with 10 of its 40 layers removed, then healed with a $53 LoRA run | 16.5 / 12 GB GGUF (from 22.9 GB Q4) | [write-up](qwen36-moe/qwen36-27b-a2.8b.md) · [folder](qwen36-moe/) · [Hugging Face](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B) |
| **Qwen3.8-Flash-Next-131B-A6B** | Qwen3.8-Flash-Next with 87.5% of its 51B-parameter n-gram table removed. No training. Scores at parity with the full model on tool use, GSM8K and MMLU | 64.8 GB GGUF (from 90 GB Q3) | [write-up](qwen38-flash-next-ngram/qwen38-flash-next-131b-a6b.md) · [folder](qwen38-flash-next-ngram/) · [Hugging Face](https://huggingface.co/Cyronius/Qwen3.8-Flash-Next-131B-A6B-GGUF) |
| **titler** | A 5.8M-parameter Llama-architecture model that turns a prompt into a short title. Runs on the CPU in 15–100 ms. Ships a Rust engine and C, Python, npm and WebAssembly packages | 4 MB GGUF | [README](titler/README.md) · [embedding guide](titler/app/EMBEDDING.md) · [folder](titler/) · [Hugging Face](https://huggingface.co/Cyronius/titler) · [try it](https://cyronius.github.io/model-experiments/titler/) |

The two Qwen prunes reached nearly opposite conclusions one model generation apart. On Qwen3.6, layer pruning was survivable and $53 of healing recovered it. On Qwen3.8, the same layer cut was fatal without healing, but most of the n-gram table came out for free.

Total cloud spend: about $80 for Qwen3.6 and $8 for Qwen3.8.

## In progress

| folder | question | state |
|---|---|---|
| [`manifold-compression/`](manifold-compression/) | Can a big model's activations be projected onto a smaller architecture to give a usable small model for a few dollars? | A Qwen3.5 2B → 0.8B projection beats the stock 0.8B on perplexity (13.85 vs 15.48), but loops in open-ended generation. Distillation has not fixed the loops yet. See [findings.md](manifold-compression/findings.md). |
| [`asd-ste100/`](asd-ste100/) | A tiny non-autoregressive editor that rewrites English into ASD-STE100 Simplified Technical English | Lexicon, checker and training data are done (8,587 top-rated pairs for $1.71 of pod time). The model is not built yet. See [HANDOFF.md](asd-ste100/HANDOFF.md). |
| [`ternary-embedder/`](ternary-embedder/) | A text-embedding model with ternary weights and int8 activations, sized for the Strix NPU | The first training grid is done. The 2 × 384 model matches all-MiniLM-L6-v2 on STS and is 5.1 points lower on retrieval. No NPU engine yet. |
| [`sprite-diffusion/`](sprite-diffusion/) | A 24M-parameter diffusion transformer for 32x32 RGBA pixel-art sprites on the AMD XDNA2 NPU | Generates 64-sprite sheets in about 5 seconds at 8 steps. Sampling defaults are 16 steps normally and 8 for fast mode; data coverage and balanced training are ongoing. Weights stay local. See [README](sprite-diffusion/README.md). |
| [`npu-kernel-coder/`](npu-kernel-coder/) | Fine-tune a small open MoE to write AMD XDNA2 NPU kernels | Scoping only. See [plan.md](npu-kernel-coder/plan.md). |

## Failed experiments

| folder | question | result |
|---|---|---|
| [`mow-nested-width/`](mow-nested-width/) | Do nested variable-width MLPs with a router ("Mixture of Widths") beat mixture-of-experts at matched cost? | No. Routed MoW is worse than a plain fixed width at every compute level, and the specialization predictions were refuted. One side result held: one nested weight matrix matches four separately trained dense widths. See [README](mow-nested-width/README.md). |
| [`ternary/`](ternary/) | Can post-training ternary quantization reproduce Prism's Ternary-Bonsai-1.7B from Qwen3-1.7B? | No. Every post-training method collapses the model (wikitext perplexity 63k–297k against 16.8 for the parent). Bonsai keeps 18.3, so its weights came from fine-tuning, not rounding. See [results/summary.md](ternary/results/summary.md). |

## Shared tools

[`bench/`](bench/) is the shared benchmark harness: tool use, GSM8K, MMLU, perplexity, IFEval and MT-Bench (Haiku as judge), plus pod scripts and the result logs. See [bench/README.md](bench/README.md).

Model weights, adapters and quantized builds are not in this repo. Use the Hugging Face links above.

## License

Code is Apache-2.0. See [LICENSE](LICENSE). The exception is [`titler/`](titler/), which is dual-licensed under MIT or Apache-2.0, at your option ([LICENSE-MIT](titler/LICENSE-MIT), [LICENSE-APACHE](titler/LICENSE-APACHE)). Model weights are on Hugging Face under their own licenses: Apache-2.0 for the Qwen3.6 derivative, MIT or Apache-2.0 for titler, and qwen-community-1.0 for the Qwen3.8 derivative.
