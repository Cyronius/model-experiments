import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { createTitler } from "../index.js";

test("titles a prompt with the packaged model", async () => {
  const t = await createTitler();
  const title = t.title("how do I sort a list of dicts by a key in python");
  assert.match(title.toLowerCase(), /sort/);
  assert.match(title.toLowerCase(), /python/);
  assert.ok(title.split(" ").length <= 8);
  t.free();
});

test("accepts model bytes and a beam count", async () => {
  const bytes = await readFile(new URL("../titler.gguf", import.meta.url));
  const t = await createTitler({ model: new Uint8Array(bytes) });
  assert.ok(t.title("write a poem about autumn rain", { beams: 1 }).length > 0);
  assert.ok(t.title("", { beams: 99 }).length >= 0);
  t.free();
});

test("a missing model file is an error", async () => {
  await assert.rejects(createTitler({ model: new URL("./nope.gguf", import.meta.url) }));
});
