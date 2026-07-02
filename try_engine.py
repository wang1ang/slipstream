"""Interactive REPL for the dynamic-batch scheduler — test "入" (mid-flight join).

    python try_engine.py [--model PATH] [--raw] [-n N] [-d DEPTH] [--debug]

A fixed input box sits at the bottom (like most CLIs); generated text scrolls
above it. Type a prompt + Enter to start; type another WHILE it runs to add it
into the live batch (that's "入"). :q or Ctrl-C quits.

Drives slipstream.scheduler.Scheduler: new requests are chunk-prefilled and
merged into the running batch. -d = draft depth k (0 = pure AR).
"""

import os
import argparse
import asyncio

import mlx.core as mx
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout import Layout, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document

from slipstream import registry
from slipstream.engine import Engine, find_mtp
from slipstream.mtp import Drafter
from slipstream.scheduler import Scheduler, Req, PrefillGroup


def to_ids(eng, text, raw):
    if raw:
        return eng.encode(text)
    return eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model path or name; default: scan ~/.mtplx/models")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("-n", "--max-tokens", type=int, default=8192)
    ap.add_argument("-d", "--depth", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    entry = registry.select(args.model)
    mtp = find_mtp(entry.path)
    print(f"[loading {entry.name}{' + MTP head' if mtp else ' (headless, pure AR)'}...]")
    eng = Engine(entry.path)
    drafter = Drafter(eng, mtp, bits=4) if mtp else None
    sch = Scheduler(eng, drafter, k=args.depth, chunk=512, debug=args.debug)

    prompts = {}         # rid -> prompt text
    produced_text = {}   # rid -> decoded output so far
    output_lines = [""]

    # output shown via a read-only Buffer: putting the cursor at the end makes
    # the window auto-scroll to the bottom (follow latest output).
    output_buf = Buffer(read_only=True)

    def render():
        lines = ["[Type a prompt + Enter. Add more while it runs (入). :q quits.]", ""]
        for rid in sorted(produced_text):
            lines.append(f"--- req{rid}: {prompts.get(rid, '')[:50]!r}")
            lines.extend(produced_text[rid].split("\n"))
            lines.append("")
        text = "\n".join(lines)
        output_buf.set_document(
            Document(text, cursor_position=len(text)), bypass_readonly=True
        )

    # --- UI: scrolling output on top, fixed input box at bottom ---
    output_win = Window(content=BufferControl(buffer=output_buf), wrap_lines=True)
    input_buf = Buffer(multiline=False)
    input_win = Window(content=BufferControl(buffer=input_buf), height=1)
    layout = Layout(
        HSplit([output_win, Window(height=1, char="─"), input_win]),
        focused_element=input_win,
    )

    next_rid = [0]

    def add(text):
        rid = next_rid[0]
        next_rid[0] += 1
        prompts[rid] = text
        produced_text[rid] = ""
        # Prefill the new request and merge it into the live batch (入). The
        # merge returns each joined request's FIRST token — show it now (it is
        # not part of the next step()'s output).
        group = PrefillGroup(reqs=[Req(rid, to_ids(eng, text, args.raw), args.max_tokens)])
        while not sch.prefill_chunk(group):
            pass
        for r, first in sch.merge_ready(group):
            produced_text[r] += eng.decode([first])
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

    app = Application(layout=layout, key_bindings=kb, full_screen=True,
                      mouse_support=True, refresh_interval=0.1)

    async def driver():
        # one scheduler step per loop iteration; yield to the UI between steps
        while True:
            if sch.has_rows():
                for rid, toks in sch.step():
                    produced_text[rid] = produced_text.get(rid, "") + eng.decode(toks)
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
