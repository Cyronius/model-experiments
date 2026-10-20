# titler-wasm

A 4 MB model that turns a prompt into a short title. It runs on the CPU with WebAssembly, in the browser and in Node.
Nothing is sent to a server.

```js
import { createTitler } from "titler-wasm";

const titler = await createTitler();                 // loads the model in this package (4 MB)
console.log(titler.title("how do I sort a list of dicts by a key in python"));   // Sorting Dicts By Key in Python
console.log(titler.title("a long prompt…", { beams: 1 }));                       // 1 beam is fastest
```

- `createTitler({ model })` takes a URL, a file path (Node) or the GGUF bytes, if you host the model yourself.
- A title takes about 50-150 ms. Run it in a Web Worker so that the page stays smooth (see `app/website/worker.js` in the repository).
- With a bundler, make sure `pkg/titler_wasm_bg.wasm` and `titler.gguf` are copied as files; the package finds them with `new URL(..., import.meta.url)`.
