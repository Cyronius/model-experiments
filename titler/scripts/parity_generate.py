"""Compare greedy titles from llama-completion (GGUF) with PyTorch greedy decoding (2026-10-19).

    python scripts/parity_generate.py --gguf model/v3-title-8L.f16.gguf --model model/v3-title-8L --n 100
"""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import LlamaForCausalLM

TOOL = str(Path(".llama-build/bin/Release/llama-completion.exe").resolve())
BOS, EOS, SEP, PAD = 1, 2, 3, 0


def llama_title(gguf: str, prompt: str, workdir: Path) -> str:
    f = workdir / "p.txt"
    f.write_bytes((prompt + "<sep>").encode("utf-8"))
    out = subprocess.run([TOOL, "-m", gguf, "-f", str(f), "-n", "40", "--temp", "0", "--top-k", "1", "-no-cnv",
                          "--no-display-prompt", "--no-escape", "--no-warmup", "-t", "4"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = re.sub(r"\[[0-9;]*m", "", out.stdout).replace("[end of text]", "")
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    tok = Tokenizer.from_file(f"{args.model}/tokenizer.json")
    model = LlamaForCausalLM.from_pretrained(args.model).eval()
    rows = [json.loads(line) for line in Path("data/eval/eval_set.jsonl").open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r["prompt"].isascii() and len(r["prompt"]) < 800][: args.n]
    same = 0
    shown = 0
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        for r in rows:
            ids = [BOS] + tok.encode(r["prompt"].rstrip()).ids[:256] + [SEP]
            x = torch.tensor([ids])
            gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=40, do_sample=False,
                                 eos_token_id=EOS, pad_token_id=PAD)
            want = tok.decode(gen[0, len(ids):].tolist(), skip_special_tokens=True).strip()
            got = llama_title(args.gguf, r["prompt"].rstrip(), Path(tmp))
            if got == want:
                same += 1
            elif shown < 8:
                shown += 1
                print(f"  torch: {want!r}\n  llama: {got!r}")
    print(f"identical titles: {same} of {len(rows)}")


if __name__ == "__main__":
    main()
