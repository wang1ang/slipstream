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

import hashlib
import os
from pathlib import Path
import sys
from dataclasses import dataclass, field

import mlx.core as mx

from .engine import Engine, BatchState
from .mtp import Drafter
from .prefixcache import PrefixCache


PROMPT_CACHE_MIN_TOKENS = 4096


@dataclass
class Req:
    rid: int
    prompt: list[int]
    max_tokens: int
    # L4 marks chat/session requests so L3 can store generated block snapshots.
    # Direct callers/tests can leave this off and only use prompt block caching.
    session_cache: bool = False
    out: list[int] = field(default_factory=list)
    session_ssm_snaps: dict[int, object] = field(default_factory=dict)


@dataclass
class PrefillGroup:
    """One request being prefetched before it joins the live decode batch."""
    reqs: list[Req]
    single: BatchState | None = None
    first: int | None = None
    last_h: object = None
    state: BatchState | None = None               # single-row chunked prefill
    pos: int = 0
    ssm_snaps: dict[int, list] = field(default_factory=dict)
    cacheable: bool = False
    cached_h: object = None
    started: bool = False


class Scheduler:
    def __init__(self, engine: Engine, drafter: Drafter | None, *, k=1, chunk=512,
                 prefix_cache=8, prefix_cache_dir=None, debug=False):
        self.eng = engine
        self.dr = drafter
        # No MTP head -> no speculation possible; k is forced to 0 (pure AR).
        self.k = k if drafter is not None else 0
        self.chunk = chunk
        self.eos = engine.eos_token_ids
        self.debug = debug
        self._t = 0
        # One prefix tree with two independent LRU pools. Prompt entries and
        # session entries share prefix structure but never evict each other.
        cache_dir = self._prefix_cache_dir(prefix_cache_dir)
        self.pc = PrefixCache(capacity={"prompt": prefix_cache,
                                        "session": prefix_cache},
                              disk_dir=cache_dir, log=self._log) \
            if prefix_cache else None

        # live decode batch
        self.state: BatchState | None = None
        self.h = None
        self.primary = None
        self.dcache = drafter.make_cache() if drafter is not None else None
        self.rows: list[Req] = []      # row i -> Req

    def _log(self, msg):
        if self.debug:
            print(f"[sched t={self._t}] {msg}", file=sys.stderr, flush=True)

    def _prefix_cache_dir(self, value):
        if value in (None, False, "", "none", "off"):
            return None
        if value != "auto":
            return value
        model_path = os.path.abspath(os.path.expanduser(self.eng.model_path))
        digest = hashlib.sha256(model_path.encode()).hexdigest()[:12]
        name = os.path.basename(model_path.rstrip(os.sep)) or "model"
        return Path.home() / ".cache" / "multiplex" / "prefixcache" / f"{name}-{digest}"

    def _find_cache(self, ids):
        return self.pc.find(ids) if self.pc is not None else None

    def _log_prefix_cache_decision(self, req: Req, ids, match) -> None:
        if not self.debug or self.pc is None:
            return
        chosen = match.prefix_len if match is not None else 0
        source = match.source if match is not None else None
        self._log(f"PREFIX CACHE FIND rid={req.rid} prompt_len={len(ids)} "
                  f"chosen={chosen} pool={getattr(match, 'pool', None)!r} "
                  f"source={source!r}")

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
        if len(group.reqs) != 1:
            raise ValueError("prefill is serial; PrefillGroup must contain one request")

        req = group.reqs[0]
        ids = req.prompt
        eng = self.eng
        if not group.started:
            hit = self._find_cache(ids)
            self._log_prefix_cache_decision(req, ids, hit)
            match = hit[1] if hit is not None else None

            group.cached_h = None
            group.cacheable = len(ids) > PROMPT_CACHE_MIN_TOKENS
            if match is not None:
                payload = match.payload
                if len(payload) == 3:
                    full, base_ssm, pos = payload
                else:
                    full, base_ssm, pos, group.cached_h = payload
                group.state = eng.restore_at(full, base_ssm, pos)
                group.pos = pos
                self._log(f"PREFIX HIT reuse={pos}/{len(ids)} tail={len(ids) - pos} "
                          f"rid={req.rid} source={match.source!r}")
                if group.cached_h is None and pos == len(ids):
                    group.state = BatchState(cache=eng._make_empty_cache(), lengths=[0])
                    group.pos = 0
                    group.cacheable = False
                    self._log(f"PREFIX HIT exact without hidden; cold rerun rid={req.rid}")
            else:
                group.state = BatchState(cache=eng._make_empty_cache(), lengths=[0])
                group.pos = 0
                self._log(f"PREFILL len={len(ids)} rid={req.rid} (cold)")
            group.started = True

        if cancelled and cancelled(req.rid):
            self._log(f"PREFILL CANCELLED rid={req.rid}")
            return None

        h = group.cached_h if group.pos == len(ids) else None
        if h is None:
            start = group.pos
            end = min(start + self.chunk, len(ids)) if self.chunk else len(ids)
            h = eng.prefill_piece(group.state, ids[start:end], len(ids),
                                  log=self._log)
            group.pos = end
            if (group.cacheable and self.pc is not None and self.chunk
                    and end < len(ids) and end >= len(ids) - 3 * self.chunk
                    and end - start == self.chunk
                    and end >= PROMPT_CACHE_MIN_TOKENS):
                group.ssm_snaps[end] = eng.clone_ssm(group.state.cache)
            if cancelled and cancelled(req.rid):
                self._log(f"PREFILL CANCELLED rid={req.rid}")
                return None
            if group.pos < len(ids):
                return False

        if group.cacheable and self.pc is not None:
            full = eng.clone_state(group.state)
            stored = False
            for p, ssm in group.ssm_snaps.items():
                if PROMPT_CACHE_MIN_TOKENS <= p < len(ids):
                    block = p // self.chunk if self.chunk else 0
                    self.pc.store(
                        ids, (full, ssm, p),
                        source=f"prompt rid={req.rid} block={block}",
                        pool="prompt",
                        save=False,
                    )
                    stored = True
                    self._log(f"PREFIX STORE block={block} "
                              f"len={p}/{len(ids)} rid={req.rid}")
            if stored:
                self._log(f"PREFIX STORE DEFER FLUSH rid={req.rid}")

        h = h[:, -1:, :]
        first = int(mx.argmax(eng.logits(h)[0, -1]))

        req.out.append(first)
        group.single = eng.extract_row(group.state, 0)
        group.first = first
        group.last_h = h
        return True

    def merge_ready(self, group: PrefillGroup) -> list[tuple[int, int]]:
        """Merge one prefilled request into the live batch."""
        if len(group.reqs) != 1:
            raise ValueError("prefill is serial; PrefillGroup must contain one request")

        singles, hs, prims, reqs = [], [], [], []
        for i, req in enumerate(self.rows):        # existing rows -> singles
            singles.append(self.eng.extract_row(self.state, i))
            hs.append(self.h[i:i + 1])
            prims.append(self.primary[i:i + 1])
            reqs.append(req)
        joined = []
        r = group.reqs[0]
        singles.append(group.single)
        hs.append(group.last_h)
        prims.append(mx.array([group.first]))
        reqs.append(r)
        joined.append((r.rid, group.first))
        self.state = self.eng.merge_states(singles)
        self.h = mx.concatenate(hs, axis=0)
        self.primary = mx.concatenate(prims, axis=0)
        self.rows = reqs
        if self.dr is not None:                    # merge resets the draft cache
            self.dcache = self.dr.make_cache()
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

        if k == 0:                                  # no head -> no draft
            draft_ids = [[] for _ in range(B)]
        else:
            drafts = dr.draft(h, primary, k, self.dcache)
            draft_ids = [[int(x) for x in drafts[i]] for i in range(B)]

        snap = eng.snapshot_ssm(state)
        lengths_before = list(state.lengths)
        verify_in = mx.array([[int(primary[i])] + draft_ids[i] for i in range(B)])
        vhidden = eng.forward(state, verify_in)
        trunk_pred = mx.argmax(eng.logits(vhidden), axis=-1)

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
        self._log(f"ADVANCE {[r.rid for r in rows]} accept={accs} min={m}")

        emitted, finished = [], []
        for i in range(B):
            toks = draft_ids[i][:m] + [int(trunk_pred[i, m])]
            for j, t in enumerate(toks):
                if t in eos or len(rows[i].out) + j + 1 >= rows[i].max_tokens:
                    toks = toks[: j + 1]
                    finished.append(i)
                    break
            rows[i].out.extend(toks)
            emitted.append((rows[i].rid, toks))

        primary = trunk_pred[:, m]
        if m == k:
            h = vhidden[:, -1:, :]
        else:
            eng.restore_ssm(state, snap)
            eng.trim_attention(state, k - m)
            state.lengths = list(lengths_before)
            commit_in = mx.array([[int(verify_in[i, 0])] + draft_ids[i][:m] for i in range(B)])
            h = eng.forward(state, commit_in)[:, -1:, :]
        self.state, self.h, self.primary = state, h, primary
        self._capture_session_blocks(rows, state)

        if finished:
            for i in finished:
                self._store_finished_session(rows[i], state, i)
            self._log(f"EXIT {[rows[i].rid for i in finished]}")
            self._keep([i for i in range(B) if i not in finished])

        return emitted

    def _capture_session_blocks(self, rows: list[Req], state: BatchState) -> None:
        if self.pc is None or not self.chunk:
            return
        for i, req in enumerate(rows):
            if not req.session_cache:
                continue
            pos = state.lengths[i]
            full_len = len(req.prompt) + len(req.out)
            if pos <= len(req.prompt) or pos > full_len or pos % self.chunk:
                continue
            if pos in req.session_ssm_snaps:
                continue
            single = self.eng.extract_row(state, i)
            req.session_ssm_snaps[pos] = self.eng.clone_ssm(single.cache)

    def _store_finished_session(self, req: Req, state: BatchState, row: int) -> None:
        """Store generated-session cache at block boundaries only."""
        if self.pc is None or not req.session_cache:
            return

        prefix = req.prompt + req.out
        if not prefix:
            return

        covered = state.lengths[row] - len(req.prompt)
        if covered < 0 or covered > len(req.out):
            self._log(f"SESSION STORE SKIP rid={req.rid} covered={covered} "
                      f"out={len(req.out)}")
            return

        single = self.eng.extract_row(state, row)
        pos = single.lengths[0]
        final_len = len(prefix)
        while self.chunk and pos < final_len:
            boundary = ((pos // self.chunk) + 1) * self.chunk
            if boundary > final_len:
                break
            h_piece = self.eng.forward(
                single, mx.array([prefix[pos:boundary]], dtype=mx.int32)
            )
            mx.eval(h_piece, *(c.state for c in single.cache))
            pos = boundary
            req.session_ssm_snaps[pos] = self.eng.clone_ssm(single.cache)

        if pos < final_len:
            h_final = self.eng.forward(
                single, mx.array([prefix[pos:final_len]], dtype=mx.int32)
            )
            mx.eval(h_final, *(c.state for c in single.cache))

        blocks = [(p, ssm) for p, ssm in sorted(req.session_ssm_snaps.items())
                  if len(req.prompt) < p <= final_len]
        if not blocks:
            return

        full = self.eng.clone_state(single)
        for p, ssm in blocks:
            block = p // self.chunk if self.chunk else 0
            self.pc.store(
                prefix, (full, ssm, p),
                source=f"session rid={req.rid} block={block}",
                pool="session",
                save=False,
            )
        self._log(f"SESSION STORE blocks={len(blocks)} len={len(prefix)} rid={req.rid}")

    def flush_prefix_cache(self) -> None:
        if self.pc is not None:
            self.pc.flush()

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
