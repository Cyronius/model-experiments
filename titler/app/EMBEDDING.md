# Use titler in your app

titler turns a prompt into a short title (8 words at most). The model is one 4 MB file. It runs on the CPU, offline,
with no telemetry.

## Use the GGUF directly

[Download `titler-v2-q4_k_m.gguf`](https://huggingface.co/Cyronius/titler/resolve/main/titler-v2-q4_k_m.gguf)
from Hugging Face and load it in a runtime that supports Llama GGUF models, such as llama.cpp. The weights and
vocabulary are in that file; the titler language bindings are optional. A copy is also included in this
repository at `app/model/titler-v2-q4_k_m.gguf` (relative to `titler/`).

The trained sequence is `<s>{prompt}<sep>{title}</s>`, rather than a chat template. Encode BOS, up to 256 prompt
tokens and the special separator token, then generate the title through EOS. Let the runtime handle the GGUF
vocabulary and special tokens. The titler engine adds 4-beam search, repetition and filler-word rules, and title
cleanup. The quality and speed below were measured with that engine; default runtime generation can differ.

## Use the titler engine

The bindings below can be built from source. Prebuilt wheels, npm tarballs and binaries are not currently
published in this repository's GitHub releases; use the [build instructions](#build-from-source) below.

| Your app is | Use | Setup |
|---|---|---|
| Rust | the `titler` crate | `titler = { git = "https://github.com/Cyronius/model-experiments" }` |
| C, C++, C#, Go, Swift, Java (anything that loads a DLL) | the C library | Build `titler-capi` |
| Python | the `titler` wheel | Build the C library, then the platform wheel |
| A web page or Node | `titler-wasm` | Build the local npm package |
| A script or shell | the `titler` command | Build `titler-cli` |

Locally built wheels, npm packages and the command-line tool include the model.

## Rust

```rust
let titler = titler::Titler::from_path("titler-v2-q4_k_m.gguf")?;   // load once
let title = titler.title("how do I sort a list of dicts by a key in python")?;
```

`Titler` is `Send + Sync`. Share one between threads (in an `Arc`); `title` takes `&self`. `title_with_beams(prompt, 1)` is
the fast setting. For a faster native build, use `titler-llama` (llama.cpp, needs a C++ compiler) in place of `titler`.

## C and everything that can call C

Header: `capi/include/titler.h`. Library: `titler_c.dll`, `libtitler_c.so` or `libtitler_c.dylib` (and a static library).

```c
titler_t *t = titler_new_from_file("titler-v2-q4_k_m.gguf");
char out[256];
int64_t n = titler_title(t, "how do I sort a list of dicts by a key in python", 0 /* default beams */, out, sizeof out);
if (n < 0) fprintf(stderr, "%s\n", titler_last_error());
titler_free(t);
```

- Strings are UTF-8. A 256-byte buffer holds any title. The return value is the full length; if it is `cap` or more, the title was cut.
- `titler_title` can be called from many threads at once on one handle. Free the handle only when no call is running.
- Errors return NULL or -1. The message is in `titler_last_error()` for the calling thread.
- C#: `[DllImport("titler_c")]` with the same signatures. Go: cgo or `purego`. Both call plain C functions.

## Python

```python
from titler import Titler
t = Titler()                         # the model inside the wheel; or Titler("path.gguf")
print(t.title("how do I sort a list of dicts by a key in python"))
```

A `Titler` can be shared between threads. The wheel is built per platform (the C library is inside it).

## Web and Node

```js
import { createTitler } from "titler-wasm";
const titler = await createTitler();                      // loads the 4 MB model
titler.title("how do I sort a list of dicts by a key in python");
```

In a browser, run it in a Web Worker so the page does not freeze; `website/worker.js` is a complete example.
Host `pkg/titler_wasm_bg.wasm` and `titler.gguf` as static files next to your script.

## What you get

| | |
|---|---|
| Output | 3 to 8 words in Title Case, for example `Sorting Dicts By Key in Python`. Same input, same output. |
| Input | Any text. Only the first 256 tokens (about 1,000 characters of English) are read. English only: other languages are not supported. |
| Speed, 4 beams | Native Rust (one thread): median 40-70 ms, slowest 5% 200-275 ms (long prompts). llama.cpp: 18 ms / 48 ms. Browser: 50-175 ms. |
| Speed, 1 beam | About half of that, with slightly worse titles. |
| Memory | About 25 MB after loading (weights are held as 32-bit numbers). Load time about 30 ms. |
| Quality | Judged against the previous titler (v1.5): the new titles win about 65% of the pairs that have a winner. |
| Beams | 4 is the default the model was judged with. The maximum is 8. |

## Limits

- It titles; it does not summarize, translate or answer.
- It sometimes drops an apostrophe (`Nick Wildes Judy Hopp Dialog Line`). Weak inputs are vague one-liners and pasted logs.
- The training data and its licenses are listed in `model/NOTICE.md`.

## Build from source

Run from the repository root. C and CLI builds need Rust; Python wheels also need the Python `build` and
`wheel` modules. The npm build needs the `wasm32-unknown-unknown` Rust target and `wasm-bindgen-cli` matching
`Cargo.lock` (see [README.md](README.md)).

```bash
cd titler/app
cargo build --release -p titler-capi       # C library -> target/release/
cargo build --release -p titler-cli        # command-line tool -> target/release/titler
python python/build_wheel.py              # wheel -> python/dist/
bash npm/build.sh                         # builds the local package in npm/
```

Install a wheel from `python/dist/` with `python -m pip install <path-to-wheel>`. Install the built npm package
from your JavaScript project with `npm install <path-to-titler/app/npm>`.
