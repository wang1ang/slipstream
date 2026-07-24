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
import json
import queue
import sys
import threading
import time

from ..bridge import ThinkingParser, normalize_messages_for_template
from .engine import Engine
from .mtp import find_drafter
from .scheduler import Scheduler, Req, PrefillGroup


_DONE = object()   # sentinel pushed to a request's queue when it finishes


class RequestManager:
    """L4-owned request lifecycle state.

    HTTP threads submit/cancel requests here; the engine thread drains pending
    work, emits text, and finishes requests. Lower layers never see this object.
    """

    def __init__(self, decode) -> None:
        self._lock = threading.Lock()
        self._incoming: list[Req] = []
        self._queues: dict[int, queue.Queue] = {}
        self._toks: dict[int, list[int]] = {}
        self._shown: dict[int, int] = {}
        self._cancelled: set[int] = set()
        self._rid = 0
        self._decode = decode

    def submit(self, prompt_ids, max_tokens, *, temperature=None,
               top_p=None, top_k=None) -> tuple[Req, queue.Queue]:
        q: queue.Queue = queue.Queue()
        with self._lock:
            req = Req(
                self._rid, list(prompt_ids), max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                session_cache=True,
            )
            self._rid += 1
            self._incoming.append(req)
            self._queues[req.rid] = q
            self._toks[req.rid] = []
            self._shown[req.rid] = 0
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

    def emit(self, emitted) -> None:
        decode = self._decode
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
            q.put(delta)
            with self._lock:
                if rid in self._shown:
                    self._shown[rid] = len(text)


class Hub:
    def __init__(self, model_path, mtp_path, *, k=3, chunk=512, debug=False,
                 prefix_cache_dir="auto", dynamic_depth=True):
        self.model_id = model_path.rstrip("/").split("/")[-1]
        self._cfg = dict(model_path=model_path, mtp_path=mtp_path,
                         k=k, chunk=chunk, debug=debug,
                         prefix_cache_dir=prefix_cache_dir,
                         dynamic_depth=dynamic_depth)
        self._default_enable_thinking = self._load_default_enable_thinking(model_path)
        self._request_log_dir = Path("logs") / "requests"
        # The model must be loaded AND used on the same thread (MLX's GPU stream
        # is thread-bound), so the engine thread loads it. Wait until ready.
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    # --- API side (any HTTP thread) -----------------------------------------
    def stream_messages(self, messages, max_tokens, tools=None,
                        add_generation_prompt=True, enable_thinking=None):
        """Yield decoded text deltas for one request.

        L5 passes protocol-normalized messages here. L4 owns chat-template
        rendering, tokenization, generation-prompt handling, and submission to
        L3. L5 never sees prompt token ids.
        """
        for field, text in self.stream_message_parts(
            messages,
            max_tokens,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        ):
            if field == "content":
                yield text

    def stream_message_parts(self, messages, max_tokens, tools=None,
                             add_generation_prompt=True, enable_thinking=None):
        """Yield L4-normalized assistant deltas as ``(field, text)``.

        L3 only streams backend-native decoded text. L4 is the first layer that
        knows both the rendered prompt and the client-facing protocol shape, so
        it owns the Qwen-style ``<think>`` split before L5 formats SSE/JSON.
        """
        # if self._cfg["debug"]:
        #     print(f"[hub] L4 IN messages={messages!r} tools={tools!r} "
        #           f"max_tokens={max_tokens!r} "
        #           f"add_generation_prompt={add_generation_prompt!r}",
        #           file=sys.stderr, flush=True)
        effective_enable_thinking = self._effective_enable_thinking(enable_thinking)
        if getattr(self, "_cfg", {}).get("debug"):
            print(f"[hub] REQUEST enable_thinking={effective_enable_thinking}",
                  file=sys.stderr, flush=True)
        prompt_ids = self._prompt_ids(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=effective_enable_thinking,
        )
        starts_in_thinking = self._prompt_opens_thinking(prompt_ids)
        if starts_in_thinking or effective_enable_thinking is True:
            temperature, top_p, top_k = 0.6, 0.95, 20
        elif effective_enable_thinking is False:
            temperature, top_p, top_k = 0.7, 0.8, 20
        else:
            temperature = top_p = top_k = None
        thinking = ThinkingParser(starts_in_thinking=starts_in_thinking)
        for raw_delta in self._stream_prompt_ids(
            prompt_ids, max_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
        ):
            reasoning_delta, content_delta = thinking.feed(raw_delta)
            if reasoning_delta:
                yield "reasoning_content", reasoning_delta
            if content_delta:
                yield "content", content_delta
        reasoning_delta, content_delta = thinking.finish()
        if reasoning_delta:
            yield "reasoning_content", reasoning_delta
        if content_delta:
            yield "content", content_delta

    def _prompt_ids(self, messages, tools=None, *, add_generation_prompt=True,
                    enable_thinking=None):
        """Render messages to token ids for L3."""
        msgs = normalize_messages_for_template(messages)
        tok = self.tokenizer
        kw = {"tools": tools} if tools else {}
        if enable_thinking is not None:
            kw["enable_thinking"] = bool(enable_thinking)
        prompt_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=add_generation_prompt, **kw
        )
        return prompt_ids

    def _effective_enable_thinking(self, override):
        if override is None:
            return self._default_enable_thinking
        return bool(override)

    @staticmethod
    def _load_default_enable_thinking(model_path):
        try:
            path = Path(model_path).expanduser() / "mtplx_runtime.json"
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        runtime = data.get("runtime") if isinstance(data, dict) else None
        if isinstance(runtime, dict) and isinstance(runtime.get("enable_thinking"), bool):
            return runtime["enable_thinking"]
        if isinstance(data, dict) and isinstance(data.get("enable_thinking"), bool):
            return data["enable_thinking"]
        return None

    def _prompt_opens_thinking(self, prompt_ids) -> bool:
        try:
            prompt_text = self.tokenizer.decode(
                list(prompt_ids), skip_special_tokens=False
            )
        except TypeError:
            prompt_text = self.tokenizer.decode(list(prompt_ids))
        except Exception:
            return False
        open_index = prompt_text.rfind("<think>")
        if open_index < 0:
            return False
        close_index = prompt_text.rfind("</think>")
        return close_index < open_index

    def _stream_prompt_ids(self, prompt_ids, max_tokens, *,
                           temperature=None, top_p=None, top_k=None):
        """Yield decoded text deltas for one request until it finishes. If the
        consumer stops early (client disconnects -> the SSE handler stops
        iterating -> this generator is closed), mark the request cancelled so the
        engine thread drops it instead of finishing generation for no one."""
        r, q = self._requests.submit(
            prompt_ids, max_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
        )
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
                prompt_text = self.tokenizer.decode(
                    list(prompt_ids), skip_special_tokens=False
                )
            except TypeError:
                prompt_text = self.tokenizer.decode(list(prompt_ids))

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
        self.tokenizer = self.eng.tokenizer
        self._requests = RequestManager(self._decode)
        # Build the architecture-specific head before wrapping it in the
        # model-agnostic Drafter.  The configured path may be an explicit
        # --mtp override; without one, find_drafter also handles pair bundles.
        self.drafter = find_drafter(self.eng, c["mtp_path"])
        self._sched = Scheduler(self.eng, self.drafter,
                                eos_token_ids=self.tokenizer.eos_token_ids,
                                k=c["k"], chunk=c["chunk"], debug=c["debug"],
                                dynamic_depth=c["dynamic_depth"],
                                prefix_cache_dir=c["prefix_cache_dir"],
                                output_log_dir=self._request_log_dir if c["debug"] else None,
                                output_decode=lambda ids: self._decode(
                                    ids, skip_special_tokens=False
                                ))
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
                         for rid, first in sched.merge_ready(prefill_group)]
                    )
                    prefill_group = None

            if not sched.has_rows():
                if prefill_group is None and not waiting:
                    time.sleep(0.003)   # idle; wait for work
                continue

            live_before = sched.live_rids()
            self._requests.emit(sched.step())

            # finished = was live before this step but no longer live -> done
            for rid in live_before - sched.live_rids():
                self._requests.finish(rid, notify=True)

    def _decode(self, token_ids, *, skip_special_tokens=True):
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
