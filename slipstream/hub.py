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


class RequestManager:
    """L4-owned request lifecycle state.

    HTTP threads submit/cancel requests here; the engine thread drains pending
    work, emits text, and finishes requests. Lower layers never see this object.
    """

    def __init__(self, output_log_dir: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._incoming: list[Req] = []
        self._queues: dict[int, queue.Queue] = {}
        self._toks: dict[int, list[int]] = {}
        self._shown: dict[int, int] = {}
        self._cancelled: set[int] = set()
        self._rid = 0
        self._output_log_dir = output_log_dir

    def submit(self, prompt_ids, max_tokens) -> tuple[Req, queue.Queue]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            req = Req(self._rid, list(prompt_ids), max_tokens, session_cache=True)
            self._rid += 1
            self._incoming.append(req)
            self._queues[req.rid] = q
            self._toks[req.rid] = []
            self._shown[req.rid] = 0
        self._reset_output_log(req.rid)
        return req, q

    def cancel(self, rid: int) -> None:
        with self._lock:
            if rid in self._queues:
                self._cancelled.add(rid)

    def is_cancelled(self, rid: int) -> bool:
        with self._lock:
            return rid in self._cancelled

    def drain_incoming(self) -> list[Req]:
        with self._lock:
            pending, self._incoming = self._incoming, []
        return pending

    def drain_cancelled(self) -> set[int]:
        with self._lock:
            rids = set(self._cancelled)
            self._cancelled.clear()
        return rids

    def finish(self, rid: int, *, notify: bool) -> None:
        with self._lock:
            q = self._queues.pop(rid, None)
            self._toks.pop(rid, None)
            self._shown.pop(rid, None)
            self._cancelled.discard(rid)
        if notify and q is not None:
            q.put(_DONE)

    def emit(self, emitted, decode) -> None:
        for rid, toks in emitted:
            with self._lock:
                if rid not in self._toks:
                    continue
                self._toks[rid].extend(toks)
                all_toks = list(self._toks[rid])
                shown = self._shown[rid]
                q = self._queues.get(rid)
            text = decode(all_toks)
            if q is None or len(text) <= shown:
                continue
            delta = text[shown:]
            self._append_output_log(rid, delta)
            q.put(delta)
            with self._lock:
                if rid in self._shown:
                    self._shown[rid] = len(text)

    def _output_log_path(self, rid: int) -> Path | None:
        if self._output_log_dir is None:
            return None
        return self._output_log_dir / f"req-{rid}.output.log"

    def _reset_output_log(self, rid: int) -> None:
        path = self._output_log_path(rid)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        except Exception:
            pass

    def _append_output_log(self, rid: int, text: str) -> None:
        path = self._output_log_path(rid)
        if path is None or not text:
            return
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            pass


class Hub:
    def __init__(self, model_path, mtp_path, *, k=1, chunk=512, debug=False,
                 prefix_cache_dir="auto"):
        self.model_id = model_path.rstrip("/").split("/")[-1]
        self._cfg = dict(model_path=model_path, mtp_path=mtp_path,
                         k=k, chunk=chunk, debug=debug,
                         prefix_cache_dir=prefix_cache_dir)
        self._request_log_dir = Path("logs") / "requests"
        output_log_dir = self._request_log_dir if debug else None
        self._requests = RequestManager(output_log_dir)
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
        prompt_ids = self._prompt_ids(
            messages, tools, add_generation_prompt=add_generation_prompt
        )
        yield from self._stream_prompt_ids(prompt_ids, max_tokens)

    def _prompt_ids(self, messages, tools=None, *, add_generation_prompt=True):
        """Render messages to token ids for L3."""
        msgs = normalize_messages_for_template(messages)
        tok = self.eng.tokenizer
        kw = {"tools": tools} if tools else {}
        prompt_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=add_generation_prompt, **kw
        )
        return prompt_ids

    def _stream_prompt_ids(self, prompt_ids, max_tokens):
        """Yield decoded text deltas for one request until it finishes. If the
        consumer stops early (client disconnects -> the SSE handler stops
        iterating -> this generator is closed), mark the request cancelled so the
        engine thread drops it instead of finishing generation for no one."""
        r, q = self._requests.submit(prompt_ids, max_tokens)
        self._write_request_log(r, prompt_ids)
        completed = False
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    completed = True
                    return
                yield item
        finally:
            if not completed:
                self._requests.cancel(r.rid)

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
            pending = self._requests.drain_incoming()
            cancelled = self._requests.drain_cancelled()
            if cancelled:
                pending = [r for r in pending if r.rid not in cancelled]
                waiting = [r for r in waiting if r.rid not in cancelled]
                if prefill_group is not None and prefill_group.req.rid in cancelled:
                    prefill_group = None
                sched.cancel(cancelled)
                for rid in cancelled:
                    self._requests.finish(rid, notify=False)
            waiting.extend(pending)

            if prefill_group is None and waiting:
                prefill_group = PrefillGroup(req=waiting.pop(0))

            if prefill_group is not None:
                # Advance only one prefill chunk, then return to this loop so
                # live rows keep decoding and new requests can enter waiting.
                done = sched.prefill_chunk(
                    prefill_group, self._requests.is_cancelled
                )
                if done is None:
                    self._requests.finish(prefill_group.req.rid, notify=False)
                    prefill_group = None
                elif done:
                    self._requests.emit(
                        [(rid, [first])
                         for rid, first in sched.merge_ready(prefill_group)],
                        self.eng.decode,
                    )
                    prefill_group = None

            if not sched.has_rows():
                if prefill_group is None and not waiting:
                    time.sleep(0.003)   # idle; wait for work
                continue

            live_before = sched.live_rids()
            self._requests.emit(sched.step(), self.eng.decode)

            # finished = was live before this step but no longer live -> done
            for rid in live_before - sched.live_rids():
                self._requests.finish(rid, notify=True)
