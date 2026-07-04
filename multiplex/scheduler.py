"""L3 — dynamic-batch mechanism (live decode batch + serial chunked prefill).

MECHANISM ONLY. L3 offers the operations; L4 (the hub) decides policy — which
request to prefill, when, and routes output. L3 never keeps a waiting queue.

Operations:
  * ``prefill_chunk(group)``  — advance one request's prefill by one chunk.
  * ``merge_ready(group)``    — merge a prefilled request into the live decode batch.
  * ``step()``                — one speculative round over the live batch (推),
    dropping EOS/max rows (出). Speculation IS generation (k=0 = pure AR).

Everything runs on one thread (MLX GPU stream is thread-bound); L4 owns that
thread and calls these in a loop.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time
from dataclasses import dataclass, field

import mlx.core as mx

from .engine import Engine, BatchState
from .mtp import Drafter
from .prefixcache.runtime import PrefixCacheRuntime


@dataclass
class Req:
    rid: int
    prompt: list[int]
    max_tokens: int
    # L4 marks chat/session requests so L3 can store generated block snapshots.
    # Direct callers/tests can leave this off and only use prompt block caching.
    session_cache: bool = False
    out: list[int] = field(default_factory=list)
    session_cache_pos: set[int] = field(default_factory=set)
    output_log_started: bool = False
    output_log_chars: int = 0
    advance_tokens: int = 0
    advance_seconds: float = 0.0


@dataclass
class PrefillGroup:
    """One request being prefetched before it joins the live decode batch."""
    req: Req
    single: BatchState | None = None
    first: int | None = None
    last_h: object = None
    state: BatchState | None = None               # single-row chunked prefill
    pos: int = 0
    cacheable: bool = False
    cached_h: object = None
    dcache: object = None
    mtp_prev_h: object = None
    started: bool = False


class Scheduler:
    def __init__(self, engine: Engine, drafter: Drafter | None, *, k=1, chunk=512,
                 prefix_cache=8, prefix_cache_dir=None, output_log_dir=None,
                 debug=False, log=None):
        self.eng = engine
        self.dr = drafter
        # No MTP head -> no speculation possible; k is forced to 0 (pure AR).
        self.k = k if drafter is not None else 0
        self.chunk = chunk
        self.eos = engine.eos_token_ids
        self.debug = debug
        self.log = log
        self._t = 0
        self.output_log_dir = Path(output_log_dir) if output_log_dir else None
        self.prefix_cache = PrefixCacheRuntime(
            engine, drafter=drafter, capacity=prefix_cache, disk_dir=prefix_cache_dir,
            chunk=chunk, log=self._log,
        )

        # live decode batch
        self.state: BatchState | None = None
        self.h = None
        self.primary = None
        self.dcache = drafter.make_cache() if drafter is not None else None
        self.rows: list[Req] = []      # row i -> Req

    def _log(self, msg):
        if self.debug:
            line = f"[sched t={self._t}] {msg}"
            if self.log is not None:
                self.log(line)
            else:
                print(line, file=sys.stderr, flush=True)

    def has_rows(self):
        return bool(self.rows)

    def live_rids(self) -> set[int]:
        return {r.rid for r in self.rows}

    # --- prefill mechanism -------------------------------------------------
    def prefill_chunk(self, group: PrefillGroup, cancelled=None):
        """Advance one request's prefill.

        Returns False after one unfinished chunk, True when the request is ready
        to merge, or None if the client was cancelled mid-prefill.

        ``cancelled(rid)`` (optional) is polled between prefill chunks so a long
        prompt for a departed client stops instead of running to completion."""
        req = group.req
        ids = req.prompt
        eng = self.eng
        if not group.started:
            self.prefix_cache.begin_prefill(req, group)
            group.started = True

        if cancelled and cancelled(req.rid):
            self._log(f"PREFILL CANCELLED rid={req.rid}")
            self.prefix_cache.prune_unreferenced()
            return None

        h = group.cached_h if group.pos == len(ids) else None
        if h is None:
            start = group.pos
            end = min(start + self.chunk, len(ids)) if self.chunk else len(ids)
            h = eng.prefill_piece(group.state, ids[start:end], len(ids),
                                  log=self._log)
            group.pos = end
            self._append_prefill_mtp_history(group, ids, start, end, h)
            self.prefix_cache.store_prompt_block(req, group, ids, start, end, h)
            if cancelled and cancelled(req.rid):
                self._log(f"PREFILL CANCELLED rid={req.rid}")
                self.prefix_cache.prune_unreferenced()
                return None
            if group.pos < len(ids):
                return False

        h = h[:, -1:, :]
        first = int(mx.argmax(eng.logits(h)[0, -1]))

        req.out.append(first)
        self._log_output(req, [first])
        group.single = eng.extract_row(group.state, 0)
        group.first = first
        group.last_h = h
        return True

    def merge_ready(self, group: PrefillGroup) -> list[tuple[int, int]]:
        """Merge one prefilled request into the live batch."""
        singles, hs, prims, reqs, dcaches = [], [], [], [], []
        for i, req in enumerate(self.rows):        # existing rows -> singles
            singles.append(self.eng.extract_row(self.state, i))
            hs.append(self.h[i:i + 1])
            prims.append(self.primary[i:i + 1])
            reqs.append(req)
            if self.dr is not None:
                dcaches.append(self.dr.extract_cache_row(self.dcache, i))
        joined = []
        r = group.req
        singles.append(group.single)
        hs.append(group.last_h)
        prims.append(mx.array([group.first]))
        reqs.append(r)
        if self.dr is not None:
            dcaches.append(group.dcache or self.dr.make_cache())
        joined.append((r.rid, group.first))
        self.state = self.eng.merge_states(singles)
        self.h = mx.concatenate(hs, axis=0)
        self.primary = mx.concatenate(prims, axis=0)
        self.rows = reqs
        if self.dr is not None:
            self.dcache = self.dr.merge_caches(dcaches)
        self._log(f"JOIN {[j[0] for j in joined]} -> {len(self.rows)} rows")
        return joined

    # --- decode mechanism: one speculative round (推 + 出) ------------------
    def step(self) -> list[tuple[int, list[int]]]:
        self._t += 1
        if not self.rows:
            return []
        eng, dr, k, eos = self.eng, self.dr, self.k, self.eos
        state, h, primary, rows = self.state, self.h, self.primary, self.rows
        B = len(rows)
        t0 = time.perf_counter()

        if k == 0:                                  # no head -> no draft
            draft_ids = [[] for _ in range(B)]
        else:
            dcache_base = int(self.dcache[0].size())
            drafts = dr.draft(h, primary, k, self.dcache)
            draft_ids = [[int(x) for x in drafts[i]] for i in range(B)]

        snap = eng.snapshot_ssm(state)
        lengths_before = list(state.lengths)
        verify_in = mx.array([[int(primary[i])] + draft_ids[i] for i in range(B)])
        vhidden = eng.forward(state, verify_in)
        trunk_logits = eng.logits(vhidden)
        trunk_pred = mx.argmax(trunk_logits, axis=-1)

        accs = []
        for i in range(B):
            a = 0
            for j in range(k):
                if draft_ids[i][j] == int(trunk_pred[i, j]):
                    a += 1
                else:
                    break
            accs.append(a)
        m = min(accs)

        emitted, finished = [], []
        for i in range(B):
            toks = draft_ids[i][:m] + [int(trunk_pred[i, m])]
            for j, t in enumerate(toks):
                if t in eos or len(rows[i].out) + j + 1 >= rows[i].max_tokens:
                    toks = toks[: j + 1]
                    finished.append(i)
                    break
            rows[i].out.extend(toks)
            self._log_output(rows[i], toks)
            emitted.append((rows[i].rid, toks))

        primary = trunk_pred[:, m]
        if k != 0:
            dr.trim_cache_to(self.dcache, dcache_base + 1)
            if m:
                accepted = mx.array([draft_ids[i][:m] for i in range(B)],
                                    dtype=mx.int32)
                dr.append_history(self.dcache, vhidden[:, :m, :], accepted)
        if m == k:
            h = vhidden[:, -1:, :]
        else:
            eng.restore_ssm(state, snap)
            eng.trim_attention(state, k - m)
            state.lengths = list(lengths_before)
            commit_in = mx.array([[int(verify_in[i, 0])] + draft_ids[i][:m] for i in range(B)])
            h = eng.forward(state, commit_in)[:, -1:, :]
        mx.eval(h, primary)
        dt = max(time.perf_counter() - t0, 1e-9)
        emitted_by_rid = {rid: toks for rid, toks in emitted}
        for req in rows:
            req.advance_tokens += len(emitted_by_rid.get(req.rid, ()))
            req.advance_seconds += dt
        total_tokens = sum(req.advance_tokens for req in rows)
        total_seconds = sum(req.advance_seconds for req in rows)
        tok_s = total_tokens / max(total_seconds, 1e-9)
        prob = self._bonus_probs(trunk_logits, trunk_pred, m) if self.debug else None
        prob_text = f" prob={prob}" if prob is not None else ""
        self._log(f"ADVANCE {[r.rid for r in rows]} accept={accs} min={m} "
                  f"{prob_text} {tok_s:.0f} tok/s")
        self.state, self.h, self.primary = state, h, primary
        self.prefix_cache.capture_session_blocks(rows, state, dcache=self.dcache, h=h)

        if finished:
            self._log(f"EXIT {[rows[i].rid for i in finished]}")
            self._keep([i for i in range(B) if i not in finished])
            self.prefix_cache.prune_unreferenced()

        return emitted

    def _append_prefill_mtp_history(self, group: PrefillGroup, ids: list[int],
                                    start: int, end: int, h) -> None:
        if self.dr is None or group.dcache is None:
            return
        hidden_parts = []
        token_parts = []
        if start > 0 and group.mtp_prev_h is not None:
            hidden_parts.append(group.mtp_prev_h)
            token_parts.append(int(ids[start]))
        local = max(0, end - start - 1)
        if local:
            hidden_parts.append(h[:, :local, :])
            token_parts.extend(int(t) for t in ids[start + 1:end])
        if hidden_parts:
            hidden = mx.concatenate(hidden_parts, axis=1) \
                if len(hidden_parts) > 1 else hidden_parts[0]
            self.dr.append_history(group.dcache, hidden, token_parts)
        group.mtp_prev_h = h[:, -1:, :]

    def _bonus_probs(self, logits, pred, pos: int) -> list[float]:
        probs = mx.softmax(logits[:, pos, :], axis=-1)
        return [
            round(float(probs[i, int(pred[i, pos])]), 4)
            for i in range(int(pred.shape[0]))
        ]

    def _log_output(self, req: Req, toks: list[int]) -> None:
        if self.output_log_dir is None:
            return
        try:
            self.output_log_dir.mkdir(parents=True, exist_ok=True)
            text_path = self.output_log_dir / f"req-{req.rid}.output.log"
            tok_path = self.output_log_dir / f"req-{req.rid}.output.tokens.log"
            if not req.output_log_started:
                text_path.write_text("", encoding="utf-8")
                tok_path.write_text("", encoding="utf-8")
                req.output_log_started = True

            raw = self.eng.tokenizer.decode(req.out, skip_special_tokens=False)
            delta = raw[req.output_log_chars:]
            if delta:
                with text_path.open("a", encoding="utf-8") as f:
                    f.write(delta)
                req.output_log_chars = len(raw)
            if toks:
                with tok_path.open("a", encoding="utf-8") as f:
                    f.write(" ".join(str(int(t)) for t in toks) + "\n")
        except Exception:
            pass

    def _keep(self, keep: list[int]) -> None:
        """Retain only the given row indices in the live batch, dropping the
        rest from every parallel structure (state cache, draft cache, primary,
        hidden, rows). Empty keep clears the batch."""
        if keep:
            self.eng.filter(self.state, keep)
            if self.dr is not None:
                self.dr.filter_cache(self.dcache, keep)
            self.primary = self.primary[mx.array(keep)]
            self.h = self.h[mx.array(keep)]
            self.rows = [self.rows[i] for i in keep]
        else:
            self.state = self.h = self.primary = None
            self.rows = []
            self.dcache = self.dr.make_cache() if self.dr is not None else None

    def drop(self, rids) -> None:
        """Remove rows for the given request ids (client disconnected)."""
        rids = set(rids)
        self._log(f"DROP {list(rids)}")
        self._keep([i for i, r in enumerate(self.rows) if r.rid not in rids])
        self.prefix_cache.prune_unreferenced()

    def cancel(self, rids) -> None:
        """Notify L3 that request ids were cancelled upstream."""
        live = self.live_rids() & set(rids)
        if live:
            self.drop(live)
        else:
            self.prefix_cache.prune_unreferenced()
