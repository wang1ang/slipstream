"""Interactive REPL to feel batched speculative decoding. Loads the model once.

    python try_engine.py [--model PATH] [--raw] [-n N] [-d DEPTH]

-d is the draft depth k (tokens drafted per step); d=0 is pure AR (no draft).

Enter prompts one per line; a BLANK line runs them together (batched speculative,
take-min aligned). Greedy; each row's output == plain AR. Commands:
    :n <int>   set max tokens
    :raw       toggle chat-template on/off
    :q         quit

This layer only feeds prompts to the batched speculative entry point and shows
the result. It does NOT decide single-vs-batch or AR-vs-speculative — the batch
entry handles 1..N rows uniformly.
"""

import argparse
import sys
import time

import mlx.core as mx
from slipstream.engine import Engine
from slipstream.mtp import Drafter, speculative_generate_batch

MODEL = "~/.mtplx/models/Agents-A1-MTPLX"
MTP = MODEL + "/mtp.safetensors"


def to_ids(eng, text, raw):
    if raw:
        return eng.encode(text)
    return eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )


def run(eng, drafter, prompts, cfg):
    """Speculative decode for 1..N prompts — one parallel batch, streamed.

    Streams row 0 live (token by token as steps arrive); other rows accumulate
    and print when done."""
    ids = [to_ids(eng, p, cfg["raw"]) for p in prompts]
    B = len(prompts)
    produced = [[] for _ in range(B)]
    shown = ""
    t0 = time.time()
    if B > 1:
        print(f"--- prompt 1: {prompts[0][:50]!r}")
    for step in speculative_generate_batch(eng, drafter, ids, max_tokens=cfg["n"], k=cfg["k"]):
        for i in range(B):
            produced[i].extend(step[i])
        full = eng.decode(produced[0])          # stream row 0
        if full != shown:
            print(full[len(shown):], end="", flush=True)
            shown = full
    print()
    dt = time.time() - t0
    for i in range(1, B):
        print(f"--- prompt {i + 1}: {prompts[i][:50]!r}")
        print(eng.decode(produced[i]))
        print()
    total = sum(len(p) for p in produced)
    print(f"[{total} tok, {dt:.1f}s, {total/dt:.1f} tok/s]", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("-n", "--max-tokens", type=int, default=8192)
    ap.add_argument("-d", "--depth", type=int, default=1,
                    help="draft depth k (tokens drafted per step)")
    args = ap.parse_args()

    eng = Engine(args.model)
    print(f"[loaded in {eng.load_seconds:.1f}s, loading MTP head...]")
    drafter = Drafter(eng, MTP, bits=4)

    cfg = {"n": args.max_tokens, "raw": args.raw, "k": args.depth}
    buf = []

    while True:
        try:
            line = input(f"{len(buf) + 1}> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if line.strip() == "":
            if buf:
                run(eng, drafter, buf, cfg)
                buf = []
            continue

        s = line.strip()
        if s in (":q", ":quit", ":exit"):
            break
        if s in (":help", ":h"):
            print(__doc__)
            continue
        if s == ":raw":
            cfg["raw"] = not cfg["raw"]
            print(f"[raw={cfg['raw']}]")
            continue
        if s.startswith(":n "):
            cfg["n"] = int(s[3:]); continue
        if s.startswith(":"):
            print(f"[unknown command {s!r}; :help for list]")
            continue

        buf.append(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
