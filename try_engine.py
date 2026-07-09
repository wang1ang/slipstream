"""Chat CLI for testing dynamic batching and mid-generation joins.

    python try_engine.py [--model PATH] [--raw] [-n N] [-d DEPTH] [--debug]

A fixed input box sits at the bottom (like most CLIs); generated text scrolls
above it. Type a prompt + Enter to start; type another while it runs to add it
to the live batch. :q or Ctrl-C quits.

Drives multiplex.scheduler.Scheduler: new requests are chunk-prefilled and
merged into the running batch. -d = draft depth k (0 = pure AR).
"""

import os
import argparse
import asyncio

import mlx.core as mx
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document

from multiplex import registry
from multiplex.engine import Engine
from multiplex.mtp import find_drafter
from multiplex.scheduler import Scheduler, Req, PrefillGroup


def to_ids(tokenizer, text, raw):
    if raw:
        return tokenizer.encode(text)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )


def decode(tokenizer, token_ids, *, skip_special_tokens=True):
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model path or name; default: scan ~/.mtplx/models")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("-n", "--max-tokens", type=int, default=8192)
    ap.add_argument("-d", "--depth", type=int, default=1)
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                    help="show scheduler debug log in the log pane")
    args = ap.parse_args()

    entry = registry.select(args.model)
    eng = Engine(entry.path)
    tokenizer = eng.tokenizer
    drafter = find_drafter(eng)
    print(f"[loaded {entry.name}{' + MTP head' if drafter else ' (headless, pure AR)'}]")

    debug_lines = []

    def append_debug(line):
        debug_lines.append(line)
        del debug_lines[:-80]

    sch = Scheduler(
        eng, drafter, eos_token_ids=tokenizer.eos_token_ids,
        k=args.depth, chunk=512, debug=args.debug,
        output_decode=lambda ids: decode(tokenizer, ids, skip_special_tokens=False),
        log=append_debug if args.debug else None,
    )

    prompts = {}         # rid -> prompt text
    produced_text = {}   # rid -> decoded output so far

    # Buffers are read-only panes; putting the cursor at the end makes each
    # window auto-scroll to the bottom (follow latest output).
    output_buf = Buffer(read_only=True)
    log_buf = Buffer(read_only=True)

    def render():
        output_lines = [
            "[Type a prompt + Enter. Add more while it runs. :q quits.]",
            "",
        ]
        for rid in sorted(produced_text):
            output_lines.append(f"--- req{rid}: {prompts.get(rid, '')[:50]!r}")
            output_lines.extend(produced_text[rid].split("\n"))
            output_lines.append("")
        text = "\n".join(output_lines)
        output_buf.set_document(
            Document(text, cursor_position=len(text)), bypass_readonly=True
        )

        log_lines = ["[scheduler log]", ""]
        log_lines.extend(debug_lines[-80:] if debug_lines else ["(no logs yet)"])
        log_text = "\n".join(log_lines)
        log_buf.set_document(
            Document(log_text, cursor_position=len(log_text)), bypass_readonly=True
        )

    # --- UI: output and log on top, fixed input box at bottom ---
    output_win = Window(content=BufferControl(buffer=output_buf), wrap_lines=True)
    log_win = Window(content=BufferControl(buffer=log_buf), wrap_lines=True)
    top = VSplit([output_win, Window(width=1, char="│"), log_win])
    input_buf = Buffer(multiline=False)
    input_win = Window(content=BufferControl(buffer=input_buf), height=1)
    layout = Layout(
        HSplit([top, Window(height=1, char="─"), input_win]),
        focused_element=input_win,
    )

    next_rid = [0]

    def add(text):
        rid = next_rid[0]
        next_rid[0] += 1
        prompts[rid] = text
        produced_text[rid] = ""
        # Prefill the new request and merge it into the live batch. The
        # merge returns each joined request's FIRST token — show it now (it is
        # not part of the next step()'s output).
        group = PrefillGroup(req=Req(rid, to_ids(tokenizer, text, args.raw), args.max_tokens))
        while True:
            done = sch.prefill_chunk(group)
            if done is None:
                return
            if done:
                break
        for r, first in sch.merge_ready(group):
            produced_text[r] += decode(tokenizer, [first])
        render()

    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        text = input_buf.text.strip()
        input_buf.reset()
        if text == ":q":
            event.app.exit()
        elif text:
            add(text)

    @kb.add("c-c")
    def _(event):
        event.app.exit()

    render()

    app = Application(layout=layout, key_bindings=kb, full_screen=True,
                      mouse_support=True, refresh_interval=0.1)

    async def driver():
        # one scheduler step per loop iteration; yield to the UI between steps
        while True:
            if sch.has_rows():
                for rid, toks in sch.step():
                    produced_text[rid] = produced_text.get(rid, "") + decode(tokenizer, toks)
                render()
                app.invalidate()
            await asyncio.sleep(0.001)

    async def run_app():
        task = asyncio.create_task(driver())
        try:
            await app.run_async()
        finally:
            task.cancel()

    asyncio.run(run_app())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
