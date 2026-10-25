let titler = null;
let ready;

async function load() {
  const { default: init, WasmTitler } = await import("./pkg/titler_wasm.js");
  const [, response] = await Promise.all([init(), fetch("./titler.gguf")]);
  if (!response.ok) throw new Error(`model download failed: ${response.status}`);
  titler = new WasmTitler(new Uint8Array(await response.arrayBuffer()));
}

onmessage = async ({ data }) => {
  try {
    ready ??= load();
    await ready;
    postMessage({ type: "ready", id: data.id });
    const start = performance.now();
    const title = titler.title(data.prompt, data.beams);
    postMessage({ type: "title", id: data.id, title, ms: performance.now() - start });
  } catch (e) {
    postMessage({ type: "error", id: data.id, message: String(e) });
  }
};
