"""Local demo server: serves index.html and runs titler inference server-side
(ONNX Runtime via optimum), sidestepping browser WASM loading and the
Local-Network-Access fetch restrictions that break transformers.js when the
page isn't loaded from localhost.

Usage:
    uv run python examples/web/server.py [--model model/titler-v12-339k-web] [--port 8000]

Then open http://localhost:8000/
"""

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pydantic import BaseModel
from transformers import AutoTokenizer

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from common.validation import clean_title  # noqa: E402

GEN_KWARGS = {"max_new_tokens": 32, "num_beams": 2, "no_repeat_ngram_size": 2}

app = FastAPI()
tokenizer: AutoTokenizer = None
model: ORTModelForSeq2SeqLM = None


class TitleRequest(BaseModel):
    prompt: str


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/api/title")
def generate_title(req: TitleRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, **GEN_KWARGS)
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    title = clean_title(raw)
    return {"title": " ".join(title.split()[:8])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/titler-v12-339k-web")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ORTModelForSeq2SeqLM.from_pretrained(args.model, subfolder="onnx", use_merged=True, use_cache=True)
    print(f"loaded {args.model} -> http://localhost:{args.port}/")

    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
