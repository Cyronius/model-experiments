import init, { initSync, WasmTitler } from "./pkg/titler_wasm.js";

const isNode = typeof process !== "undefined" && !!process.versions?.node;

async function readBytes(source) {
  if (source instanceof Uint8Array) return source;
  if (source instanceof ArrayBuffer) return new Uint8Array(source);
  const url = source instanceof URL ? source : new URL(source, typeof location !== "undefined" ? location.href : import.meta.url);
  if (isNode && url.protocol === "file:") {
    const fs = "node:fs/promises";
    const { fileURLToPath } = await import(/* @vite-ignore */ "node:url");
    return new Uint8Array(await (await import(/* @vite-ignore */ fs)).readFile(fileURLToPath(url)));
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

let ready;
function loadWasm() {
  // Browsers refuse a synchronous compile on the main thread, so only Node takes the sync path.
  ready ??= isNode ? readBytes(new URL("./pkg/titler_wasm_bg.wasm", import.meta.url)).then((bytes) => initSync({ module: bytes })) : init();
  return ready;
}

/**
 * Load the model. `model` is a URL, a file path (Node), or the GGUF bytes; the default is the file in this package.
 * Returns an object with `title(prompt, { beams })` and `free()`.
 */
export async function createTitler({ model = new URL("./titler.gguf", import.meta.url) } = {}) {
  const [, bytes] = await Promise.all([loadWasm(), readBytes(model)]);
  const inner = new WasmTitler(bytes);
  return {
    title(prompt, { beams = 4 } = {}) {
      return inner.title(String(prompt), beams);
    },
    free() {
      inner.free();
    },
  };
}
