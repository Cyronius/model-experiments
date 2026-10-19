# titler app

A 4 MB model that turns a prompt into a short title. It runs on the CPU in about 15 ms.

| Folder | Content |
|---|---|
| `core/` | Tokenizer, beam search, title cleanup. No I/O. The model side is a `Backend` trait. |
| `llama/` | `Backend` for llama.cpp (through `llama-cpp-2`). |
| `cli/` | The `titler` command. The model is built into the binary. |
| `model/` | The shipped GGUF (Q4_K_M, 5.84M parameters). |

## Use

```
cargo build --release -p titler-cli
target/release/titler "how do I sort a list of dicts by a key in python"
target/release/titler -i        # type prompts, get titles
```

Building `titler-llama` compiles llama.cpp, so it needs CMake, a C++ compiler and LLVM (set `LIBCLANG_PATH` on Windows).

## Test

```
cargo test -p titler-core
```

The tests compare the Rust code with Python on golden files (`core/tests/golden/`). To rebuild the golden files:
`uv run python scripts/24_make_app_golden.py` from the repository root.
