# titler app

A 4 MB model that turns a prompt into a short title. It runs on the CPU in about 15 ms.

| Folder | Content |
|---|---|
| `core/` | Tokenizer, beam search, title cleanup. No I/O. The model side is a `Backend` trait. |
| `titler/` | The `titler` crate: load a model, call `title`. Shares one model between threads. |
| `capi/` | C library and `include/titler.h`. |
| `python/` | Python package (ctypes over the C library) and wheel builder. |
| `npm/` | The `titler-wasm` npm package. |
| `cpu/` | `Backend` in plain Rust (no C++, no dependencies). It runs in the browser. |
| `llama/` | `Backend` for llama.cpp (through `llama-cpp-2`). Faster on a desktop. |
| `cli/` | The `titler` command. The model is built into the binary. |
| `wasm/` | WebAssembly wrapper around `core` and `cpu`. |
| `website/` | The demo page. `build-site.sh` builds it into `website/dist/`. |
| `model/` | The shipped GGUF (Q4_K_M, 5.84M parameters) and `NOTICE.md` (data licenses). |

To use titler in your own app (Rust, C, Python, JavaScript), see [EMBEDDING.md](EMBEDDING.md).

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

## Website

```
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <version in Cargo.lock>
./build-site.sh
python -m http.server -d website/dist 8000
```

GitHub Actions publishes the site to GitHub Pages when you push a tag that starts with `titler-v`, for example `titler-v0.2.0` (`.github/workflows/titler-pages.yml` at the repository root).
The same tag builds the `titler` binaries for Linux, Windows and macOS and attaches them to a release.
