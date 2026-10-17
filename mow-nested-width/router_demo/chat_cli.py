"""Interactive command-line client for the router demo.

Reads a line from the user, sends the running conversation to the router's
/chat endpoint, prints the reply and a one-line status showing which model
tier answered and why. Stdlib only.

Usage:
    python router_server.py   (in one terminal)
    python chat_cli.py        (in another)
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_ROUTER_URL = "http://127.0.0.1:8090"


def send(router_url: str, messages: list[dict], timeout_s: float = 180) -> dict:
    body = json.dumps({"messages": messages}).encode("utf-8")
    req = urllib.request.Request(
        f"{router_url}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", default=DEFAULT_ROUTER_URL,
                         help=f"router base URL (default: {DEFAULT_ROUTER_URL})")
    args = parser.parse_args()

    print(f"router demo chat -- connected to {args.router_url}")
    print("type a message and press enter; 'exit' or Ctrl+D to quit\n")

    messages: list[dict] = []
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("exit", "quit"):
            break

        messages.append({"role": "user", "content": line})
        try:
            result = send(args.router_url, messages)
        except urllib.error.URLError as exc:
            print(f"[error] could not reach router at {args.router_url}: {exc}", file=sys.stderr)
            messages.pop()
            continue
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[error] router returned {exc.code}: {detail}", file=sys.stderr)
            messages.pop()
            continue

        reply = result.get("reply", "")
        messages.append({"role": "assistant", "content": reply})

        print(f"bot> {reply}")
        print(f"[{result.get('tier')} | {result.get('tokens_per_sec')} tok/s | "
              f"reason: {result.get('reason')}]\n")


if __name__ == "__main__":
    main()
