# Use titler in your app

titler turns a prompt into a short title (8 words at most). The model is one 4 MB file. It runs on the CPU, offline,
with no telemetry. Pick the way that fits your app:

| Your app is | Use | Install |
|---|---|---|
| Rust | the `titler` crate | `titler = { git = "https://github.com/Cyronius/model-experiments" }` |
| C, C++, C#, Go, Swift, Java (anything that loads a DLL) | the C library | `titler-capi-<platform>.zip` from the [releases](https://github.com/Cyronius/model-experiments/releases) |
| Python | the `titler` wheel | `pip install titler-0.1.0-py3-none-<platform>.whl` from the releases |
| A web page or Node | `titler-wasm` | `npm install titler-wasm-0.1.0.tgz` from the releases |
| A script or shell | the `titler` command | `titler-<platform>` from the releases |

The model file is `titler-v2-q4_k_m.gguf` in each release and in `app/model/`. The wheel, the npm package and the
command-line tool already contain it.

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

```
cd app
cargo build --release -p titler-capi       # C library -> target/release/
cargo build --release -p titler-cli        # command-line tool
python python/build_wheel.py               # Python wheel (from app/python)
npm/build.sh                               # npm package (needs wasm-bindgen-cli, see README.md)
```
