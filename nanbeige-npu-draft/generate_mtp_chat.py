"""Generate Nanbeige chat-template continuations for the MTP experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import capture
import run_capture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--validation-prompts", type=Path)
    parser.add_argument("--eval-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=192)
    parser.add_argument("--validation-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--server", type=Path, default=run_capture.SERVER)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with run_capture.server(False, args.output / "generation-server.log", args.server, args.threads):
        jobs = [
            (args.train_prompts, args.train_limit, args.output / "train-generated.jsonl"),
        ]
        if args.validation_prompts is not None:
            jobs.append((args.validation_prompts, args.validation_limit,
                         args.output / "validation-generated.jsonl"))
        jobs.append((args.eval_prompts, args.eval_limit, args.output / "eval-generated.jsonl"))
        for source, limit, output in jobs:
            capture.generate(argparse.Namespace(prompts=source, start=0, limit=limit,
                                                tokens=args.tokens, output=output,
                                                url=run_capture.URL))


if __name__ == "__main__":
    main()
