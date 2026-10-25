// The worker (and therefore WASM and the model) starts only on a valid submit.
for (const form of document.querySelectorAll("[data-titler]")) {
  const prompt = form.querySelector("textarea");
  const submit = form.querySelector('[type="submit"]');
  const cancel = form.querySelector('[data-cancel]');
  const status = form.querySelector('[role="status"]');
  const result = form.querySelector("output");
  // Without this handler, a native form submission could send the prompt to the host.
  submit.disabled = false;
  let worker;
  let pending = false;
  let request = 0;
  let timeout;
  let buttonWasFocused = false;

  function finish() {
    const restoreFocus = document.activeElement === cancel ||
      (buttonWasFocused && document.activeElement === document.body);
    clearTimeout(timeout);
    pending = false;
    prompt.readOnly = false;
    submit.disabled = false;
    submit.textContent = "Title it";
    cancel.hidden = true;
    result.setAttribute("aria-busy", "false");
    if (restoreFocus) submit.focus({ preventScroll: true });
  }

  function stop() {
    worker?.terminate();
    worker = undefined;
    finish();
  }

  function fail(message) {
    stop();
    status.textContent = message;
    status.dataset.error = "true";
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    if (pending) return;
    const text = prompt.value.trim();
    if (!text) {
      prompt.setCustomValidity("Enter a prompt to title.");
      prompt.reportValidity();
      return;
    }
    pending = true;
    buttonWasFocused = document.activeElement === submit;
    const id = ++request;
    prompt.readOnly = true;
    submit.disabled = true;
    submit.textContent = "Working…";
    cancel.hidden = false;
    result.setAttribute("aria-busy", "true");
    delete status.dataset.error;
    status.textContent = worker ? "Writing your title…" : "Downloading the model (4 MB)…";

    try {
      if (!worker) {
        worker = new Worker(new URL(form.dataset.worker, document.baseURI), { type: "module" });
        worker.onmessage = ({ data }) => {
          if (data.id !== request || !pending) return;
          if (data.type === "ready") {
            status.textContent = "Writing your title…";
          } else if (data.type === "title") {
            if (!data.title?.trim()) {
              fail("No title came back. Try a more descriptive prompt, then title it again.");
              return;
            }
            result.textContent = data.title;
            result.hidden = false;
            finish();
            status.textContent = "Done. Your prompt stayed in this browser.";
          } else if (data.type === "error") {
            fail("The model couldn’t finish. Check your connection, then try Title it again.");
          }
        };
        worker.onerror = (event) => {
          event.preventDefault();
          fail("The demo couldn’t start. Try Title it again or use a browser with WebAssembly support.");
        };
        worker.onmessageerror = () => fail("The demo lost its response. Try Title it again.");
      }
      worker.postMessage({ id, prompt: text, beams: 4 });
      timeout = setTimeout(() => fail("The model took too long. Check your connection, then try Title it again."), 120_000);
    } catch {
      fail("The demo couldn’t start. Try Title it again or use a browser with WebAssembly support.");
    }
  });

  cancel.addEventListener("click", () => {
    stop();
    status.textContent = "Cancelled. Your prompt is still here.";
    prompt.focus();
  });
  prompt.addEventListener("input", () => prompt.setCustomValidity(""));
  prompt.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (!pending && !event.isComposing) form.requestSubmit();
    }
  });
}
