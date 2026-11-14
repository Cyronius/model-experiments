#!/usr/bin/env python3
"""Two prompt sets for the wider live run (run_wide_live.sh).

  prompts/code-v2.jsonl       the first 160 OSS-Instruct prompts of code_study.py's test part (no
                              table holds their answers and no reranker was trained on them);
                              code-v1.jsonl is its first 24. live_bench --per-domain 24 takes 48.
  prompts/code-edit-v1.jsonl  coding work unlike OSS-Instruct, 6 of each domain:
                              edit   a whole source file from this machine (Python's standard
                                     library, two Rust crates in ~/.cargo, llama.cpp) and a change
                                     to make; the answer is the complete file again
                              write  a long program from a short request
                              Each edit row names its file and the file's sha256.

Run from the repo root: target/manifold-audit/env/bin/python memdraft/make_prompts.py
"""
import glob
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
import code_study as cs   # noqa: E402
import gen_code           # noqa: E402

DATASET = os.path.expanduser('~/.cache/model-experiments/memdraft/code-prompts-src/data-oss_instruct-decontaminated.jsonl')
CRATES = (glob.glob(os.path.expanduser('~/.cargo/registry/src/*/')) or [''])[0]
LLAMA = os.path.expanduser('~/.cache/model-experiments/memdraft/llama.cpp')

EDITS = [
    ('python', '/usr/lib64/python3.14/bisect.py',
     'Add type hints to every function. Use a TypeVar and a Protocol for the "supports less-than" '
     'requirement, and type the optional key function. Do not change behaviour.'),
    ('python', '/usr/lib64/python3.14/sched.py',
     'Replace the Event namedtuple with a frozen dataclass that orders the same way, and add a method '
     'reschedule(event, delay) that cancels the event and enters it again with the new delay.'),
    ('python', '/usr/lib64/python3.14/colorsys.py',
     'Make every conversion function raise ValueError when a component is outside 0..1, and add a '
     'doctest example to the docstring of each conversion function.'),
    ('rust', f'{CRATES}aho-corasick-1.1.2/src/util/buffer.rs',
     'Add a method `clear(&mut self)` that empties the buffer without freeing its memory, and a '
     '#[cfg(test)] module with unit tests for roll and fill.'),
    ('rust', f'{CRATES}adler2-2.0.1/src/algo.rs',
     'Rename the U32X4 type to Lanes4 everywhere and add a doc comment to every function and impl block.'),
    ('cpp', f'{LLAMA}/common/trie.cpp',
     'Add a member function `bool contains(std::string_view word) const` that reports whether the '
     'exact word was inserted, and comment each function.'),
]

WRITES = [
    ('python', 'Write a complete command-line todo application in Python: SQLite storage, argparse '
     'subcommands add, list, done and remove, due dates, and a unittest test suite at the end.'),
    ('rust', 'Write a generic LRU cache in Rust backed by a HashMap and a doubly linked list stored in a '
     'Vec with indices (no unsafe). Include get, put, remove, len and iteration from most to least recent, '
     'and unit tests.'),
    ('cpp', 'Write a header-only C++17 thread pool: a fixed number of workers, submit() returning a '
     'std::future for any callable, graceful shutdown in the destructor, and an example main() that sums '
     'a large vector in parallel.'),
    ('typescript', 'Write a TypeScript Express REST API for a small bookstore: an in-memory store, CRUD '
     'routes for books and authors, request validation, pagination, and error-handling middleware.'),
    ('go', 'Write a Go HTTP server with a per-IP token-bucket rate-limiting middleware, request logging, '
     'a JSON health endpoint, and graceful shutdown on SIGINT and SIGTERM.'),
    ('python', 'Write a Python interpreter for a small arithmetic language: a tokenizer, a '
     'recursive-descent parser that builds an AST, variables, user-defined functions, error messages '
     'with line and column, and a REPL.'),
]

FENCE = {'python': 'python', 'rust': 'rust', 'cpp': 'cpp'}


def main():
    pool = gen_code.pick(DATASET, 2000, gen_code.SEED, gen_code.MAX_CHARS)
    test = [p for p in pool if cs.part(p['id']) == 'test'][:160]
    v2 = HERE / 'prompts/code-v2.jsonl'
    v2.write_text(''.join(json.dumps(
        {'id': p['id'], 'domain': 'python' if p['lang'] == 'python' else 'other-languages', 'lang': p['lang'],
         'prompt': p['prompt']}) + '\n' for p in test))
    v1 = [json.loads(line)['id'] for line in open(HERE / 'prompts/code-v1.jsonl')]
    assert [p['id'] for p in test[:len(v1)]] == v1, 'code-v1 should be the first prompts of code-v2'
    print(f'{v2}: {len(test)} test prompts')

    rows = []
    for n, (lang, path, change) in enumerate(EDITS):
        text = Path(path).read_text()
        name = Path(path).name
        prompt = (f'Here is the file `{name}`:\n\n```{FENCE[lang]}\n{text.rstrip()}\n```\n\n{change}\n\n'
                  'Reply with the complete updated file in one code block.')
        rows.append({'id': f'edit-{n}', 'domain': 'edit', 'lang': lang, 'source': path.replace(os.path.expanduser('~'), '~'),
                     'sha256': hashlib.sha256(text.encode()).hexdigest(), 'prompt': prompt})
    for n, (lang, prompt) in enumerate(WRITES):
        rows.append({'id': f'write-{n}', 'domain': 'write', 'lang': lang, 'prompt': prompt})
    out = HERE / 'prompts/code-edit-v1.jsonl'
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{out}: {len(rows)} prompts')


if __name__ == '__main__':
    main()
