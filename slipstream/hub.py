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

import queue
import threading
import time

from .bridge import normalize_messages_for_template
from .engine import Engine
from .mtp import Drafter
from .scheduler import Scheduler, Req, PrefillGroup


_DONE = object()   # sentinel pushed to a request's queue when it finishes


class Hub:
    def __init__(self, model_path, mtp_path, *, k=1, chunk=512, debug=False):
        self.model_id = model_path.rstrip("/").split("/")[-1]
        self._cfg = dict(model_path=model_path, mtp_path=mtp_path,
                         k=k, chunk=chunk, debug=debug)
        self._lock = threading.Lock()
        self._incoming = []                       # [Req] submitted, not yet added
        self._queues: dict[int, queue.Queue] = {}  # rid -> text-delta queue
        self._toks: dict[int, list] = {}           # rid -> all tokens (hub's copy)
        self._shown: dict[int, int] = {}           # rid -> chars already emitted
        self._cancelled: set[int] = set()          # rids whose client went away
        self._rid = 0
        # The model must be loaded AND used on the same thread (MLX's GPU stream
        # is thread-bound), so the engine thread loads it. Wait until ready.
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    # --- API side (any HTTP thread) -----------------------------------------
    def prompt_ids(self, messages, tools=None):
        """Render messages to (ids, stable_len). ids is the full prompt fed to
        the model (with the generation prompt); stable_len marks the boundary of
        the real client content — the history the client resends verbatim next
        turn — which is exactly the add_generation_prompt=False render. Only that
        prefix is cached; the trailing guide is not (see Scheduler)."""
        msgs = normalize_messages_for_template(messages)
        tok = self.eng.tokenizer
        kw = {"tools": tools} if tools else {}
        stable = tok.apply_chat_template(msgs, add_generation_prompt=False, **kw)
        full = tok.apply_chat_template(msgs, add_generation_prompt=True, **kw)
        stable_len = len(stable)
        # The guide must be a pure append; if a template rewrites earlier tokens
        # instead, don't trust the boundary — cache the whole thing (today's
        # behavior, still correct).
        if list(full[:stable_len]) != list(stable):
            stable_len = len(full)
        return full, stable_len

    def stream_text(self, prompt_ids, max_tokens, stable_len=None):
        """Yield decoded text deltas for one request until it finishes. If the
        consumer stops early (client disconnects -> the SSE handler stops
        iterating -> this generator is closed), mark the request cancelled so the
        engine thread drops it instead of finishing generation for no one."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            r = Req(self._rid, list(prompt_ids), max_tokens, stable_len=stable_len)
            self._rid += 1
            self._incoming.append(r)
            self._queues[r.rid] = q
            self._toks[r.rid] = []
            self._shown[r.rid] = 0
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                yield item
        finally:
            # Normal completion already cleaned up; this matters on early close.
            self._cancelled.add(r.rid)

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
                                k=c["k"], chunk=c["chunk"], debug=c["debug"])
        self._ready.set()
        sched = self._sched
        while True:
            # admit newly submitted requests: prefill them (batched) and merge in
            with self._lock:
                pending, self._incoming = self._incoming, []
            # Skip any request whose client already went away before we started.
            pending = [r for r in pending if not self.cancelled(r.rid)]
            if pending:
                group = PrefillGroup(reqs=pending)
                # prefill_chunk aborts (returns None) if the request is cancelled
                # mid-prefill, so a 60k-token prompt for a gone client stops early.
                while (done := sched.prefill_chunk(group, self.cancelled)) is False:
                    pass
                if done:
                    self._emit([(rid, [first]) for rid, first in sched.merge_ready(group)])

            if not sched.has_rows():
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
