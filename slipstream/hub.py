"""L4 — request hub: aggregate many API endpoints onto one scheduler.

Different endpoints (chat / responses / ...) translate their wire format into
the SAME internal request; the hub funnels them all into ONE Scheduler so they
share one live batch (真并发), and routes each request's streamed text back.

Threading (forced by MLX's thread-bound GPU stream): the model + scheduler run
on ONE background engine thread; HTTP handler threads only touch thread-safe
queues. An HTTP thread submits a request and drains its text queue for SSE.

    HTTP thread ─submit()→ [engine thread: prefill new reqs, scheduler.step loop]
                                        │ per-rid text deltas
                        drain queue ←───┘   ... then _DONE
"""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import time

from .bridge import normalize_messages_for_template
from .engine import Engine
from .mtp import Drafter
from .scheduler import Scheduler, Req, PrefillGroup


_DONE = object()   # sentinel pushed to a request's queue when it finishes


class Hub:
    def __init__(self, model_path, mtp_path, *, k=1, chunk=512, debug=False,
                 prefix_cache_dir="auto"):
        self.model_id = model_path.rstrip("/").split("/")[-1]
        self._cfg = dict(model_path=model_path, mtp_path=mtp_path,
                         k=k, chunk=chunk, debug=debug,
                         prefix_cache_dir=prefix_cache_dir)
        self._lock = threading.Lock()
        self._incoming = []                       # [Req] submitted, not yet added
        self._queues: dict[int, queue.Queue] = {}  # rid -> text-delta queue
        self._toks: dict[int, list] = {}           # rid -> all tokens (hub's copy)
        self._shown: dict[int, int] = {}           # rid -> chars already emitted
        self._cancelled: set[int] = set()          # rids whose client went away
        self._rid = 0
        self._request_log_dir = Path("logs") / "requests"
        # The model must be loaded AND used on the same thread (MLX's GPU stream
        # is thread-bound), so the engine thread loads it. Wait until ready.
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    # --- API side (any HTTP thread) -----------------------------------------
    def stream_messages(self, messages, max_tokens, tools=None, add_generation_prompt=True):
        """Yield decoded text deltas for one request.

        L5 passes protocol-normalized messages here. L4 owns chat-template
        rendering, tokenization, generation-prompt handling, and submission to
        L3. L5 never sees prompt token ids.
        """
        # if self._cfg["debug"]:
        #     print(f"[hub] L4 IN messages={messages!r} tools={tools!r} "
        #           f"max_tokens={max_tokens!r} "
        #           f"add_generation_prompt={add_generation_prompt!r}",
        #           file=sys.stderr, flush=True)
        prompt_ids, session_prompt_len = self._prompt_ids(
            messages, tools, add_generation_prompt=add_generation_prompt
        )
        yield from self._stream_prompt_ids(prompt_ids, session_prompt_len, max_tokens)

    def _prompt_ids(self, messages, tools=None, *, add_generation_prompt=True):
        """Render messages to token ids for L3."""
        msgs = normalize_messages_for_template(messages)
        tok = self.eng.tokenizer
        kw = {"tools": tools} if tools else {}
        prompt_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=add_generation_prompt, **kw
        )
        if not add_generation_prompt:
            return prompt_ids, len(prompt_ids)
        session_prompt_len = self._session_prompt_len(msgs, kw)
        return prompt_ids, session_prompt_len if session_prompt_len is not None \
            else len(prompt_ids)

    def _session_prompt_len(self, msgs, kw) -> int | None:
        text = self._assistant_content_start(msgs, kw)
        if text is None:
            return None
        return len(self._encode_template_text(self.eng.tokenizer, text))

    def _assistant_content_start(self, msgs, kw) -> str | None:
        sentinel = "SLIPSTREAMRESPONSESENTINEL"
        try:
            with_content = self.eng.tokenizer.apply_chat_template(
                msgs + [
                    {"role": "assistant", "content": sentinel},
                    {"role": "user", "content": "SLIPSTREAMDUMMYUSER"},
                ],
                add_generation_prompt=False, tokenize=False, **kw
            )
        except Exception:
            return None

        marker = with_content.find(sentinel)
        return None if marker < 0 else with_content[:marker]

    @staticmethod
    def _encode_template_text(tok, text: str) -> list[int]:
        try:
            return tok.encode(text, add_special_tokens=False)
        except TypeError:
            return tok.encode(text)

    def _stream_prompt_ids(self, prompt_ids, session_prompt_len, max_tokens):
        """Yield decoded text deltas for one request until it finishes. If the
        consumer stops early (client disconnects -> the SSE handler stops
        iterating -> this generator is closed), mark the request cancelled so the
        engine thread drops it instead of finishing generation for no one."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            r = Req(self._rid, list(prompt_ids), max_tokens,
                    session_prompt_len=session_prompt_len, session_cache=True)
            self._rid += 1
            self._incoming.append(r)
            self._queues[r.rid] = q
            self._toks[r.rid] = []
            self._shown[r.rid] = 0
        self._write_request_log(r, prompt_ids)
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                yield item
        finally:
            # Normal completion already cleaned up; this matters on early close.
            self._cancelled.add(r.rid)

    def _write_request_log(self, req, prompt_ids):
        if not self._cfg["debug"]:
            return
        try:
            try:
                prompt_text = self.eng.tokenizer.decode(
                    list(prompt_ids), skip_special_tokens=False
                )
            except TypeError:
                prompt_text = self.eng.tokenizer.decode(list(prompt_ids))

            self._request_log_dir.mkdir(parents=True, exist_ok=True)
            path = self._request_log_dir / f"req-{req.rid}.log"
            body = [
                "--- prompt ---",
                prompt_text,
                "",
            ]
            path.write_text("\n".join(body), encoding="utf-8")
        except Exception:
            pass

    def cancelled(self, rid) -> bool:
        return rid in self._cancelled

    # --- engine thread (only thread that touches MLX) -----------------------
    def _run(self):
        c = self._cfg
        self.eng = Engine(c["model_path"])
        # No MTP head path -> run pure AR (scheduler forces k=0).
        self.drafter = (Drafter(self.eng, c["mtp_path"])
                        if c["mtp_path"] is not None else None)
        self._sched = Scheduler(self.eng, self.drafter,
                                k=c["k"], chunk=c["chunk"], debug=c["debug"],
                                prefix_cache_dir=c["prefix_cache_dir"])
        self._ready.set()
        sched = self._sched
        waiting: list[Req] = []
        prefill_group: PrefillGroup | None = None
        while True:
            # admit newly submitted requests; prefill them serially, one chunk
            # per loop, while live rows keep decoding.
            with self._lock:
                pending, self._incoming = self._incoming, []
            # Skip any request whose client already went away before we started.
            pending = [r for r in pending if not self.cancelled(r.rid)]
            waiting.extend(pending)

            if prefill_group is None and waiting:
                prefill_group = PrefillGroup(reqs=[waiting.pop(0)])

            if prefill_group is not None:
                # Advance only one prefill chunk, then return to this loop so
                # live rows keep decoding and new requests can enter waiting.
                done = sched.prefill_chunk(prefill_group, self.cancelled)
                if done is None:
                    prefill_group = None
                elif done:
                    self._emit([(rid, [first])
                                for rid, first in sched.merge_ready(prefill_group)])
                    prefill_group = None

            if not sched.has_rows():
                if prefill_group is None and not waiting:
                    sched.flush_prefix_cache()
                    time.sleep(0.003)   # idle; wait for work
                continue

            # Drop live rows whose client disconnected mid-generation.
            gone = [rid for rid in sched.live_rids() if self.cancelled(rid)]
            if gone:
                sched.drop(gone)

            live_before = sched.live_rids()
            self._emit(sched.step())

            # finished = was live before this step but no longer live -> done
            for rid in live_before - sched.live_rids():
                q = self._queues.pop(rid, None)
                self._toks.pop(rid, None)
                self._shown.pop(rid, None)
                if q is not None:
                    q.put(_DONE)

    def _emit(self, emitted):
        """Accumulate new tokens per rid and push the decoded text delta."""
        for rid, toks in emitted:
            if rid not in self._toks:
                continue
            self._toks[rid].extend(toks)
            text = self.eng.decode(self._toks[rid])
            q = self._queues.get(rid)
            if q is not None and len(text) > self._shown[rid]:
                q.put(text[self._shown[rid]:])
                self._shown[rid] = len(text)
