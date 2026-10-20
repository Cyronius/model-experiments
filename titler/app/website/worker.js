import init, { WasmTitler } from "./pkg/titler_wasm.js";

let titler = null;

async function load() {
  const [, response] = await Promise.all([init(), fetch("./titler.gguf")]);
  if (!response.ok) throw new Error(`model download failed: ${response.status}`);
  titler = new WasmTitler(new Uint8Array(await response.arrayBuffer()));
}

const ready = load().then(
  () => postMessage({ type: "ready" }),
  (e) => postMessage({ type: "error", message: String(e) }),
);

onmessage = async ({ data }) => {
  await ready;
  if (!titler) return;
  try {
    const start = performance.now();
    const title = titler.title(data.prompt, data.beams);
    postMessage({ type: "title", id: data.id, title, ms: performance.now() - start });
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: String(e) });
  }
};
