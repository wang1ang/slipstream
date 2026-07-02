"""L3 — dynamic-batch mechanism (live decode batch + chunked prefill + merge).

MECHANISM ONLY. L3 offers the operations; L4 (the hub) decides policy — which
requests to prefill together, when, and routes output. L3 never keeps a waiting
queue.

Operations:
  * ``prefill_chunk(group)``  — advance a batched, chunked prefill of new requests
    one chunk; returns the finished ones (ready to merge) as they complete.
  * ``merge_ready(ready)``    — merge prefilled requests into the live decode batch.
  * ``step()``                — one speculative round over the live batch (推),
    dropping EOS/max rows (出). Speculation IS generation (k=0 = pure AR).

Everything runs on one thread (MLX GPU stream is thread-bound); L4 owns that
thread and calls these in a loop.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import mlx.core as mx

from .engine import Engine, BatchState
from .mtp import Drafter
from .prefixcache import PrefixCache


@dataclass
class Req:
    rid: int
    prompt: list[int]
    max_tokens: int
    # tokens [:stable_len] are the real client content (resent verbatim next
    # turn); anything after is L4's temporary generation guide. Only the stable
    # part is cached. None -> the whole prompt is stable (direct callers/tests).
    stable_len: int | None = None
    out: list[int] = field(default_factory=list)


@dataclass
class PrefillGroup:
    """A set of new requests to prefill and join (L4 forms it). prefill_chunk
    fills these per-request results (one single-row state + first token each)."""
    reqs: list[Req]
    singles: list = field(default_factory=list)   # per-req single-row BatchState
    firsts: list = field(default_factory=list)    # per-req first sampled token
    last_h: list = field(default_factory=list)    # per-req [1,1,H] trunk hidden


class Scheduler:
    def __init__(self, engine: Engine, drafter: Drafter | None, *, k=1, chunk=512,
                 prefix_cache=8, debug=False):
        self.eng = engine
        self.dr = drafter
        # No MTP head -> no speculation possible; k is forced to 0 (pure AR).
        self.k = k if drafter is not None else 0
        self.chunk = chunk
        self.eos = engine.eos_token_ids
        self.debug = debug
        self._t = 0
        # Prefix KV-cache: snapshots taken during prefill; a new prompt that
        # continues a cached prefix restores its state and prefills only the tail.
        self.pc = PrefixCache(capacity=prefix_cache) if prefix_cache else None

        # live decode batch
        self.state: BatchState | None = None
        self.h = None
        self.primary = None
        self.dcache = drafter.make_cache() if drafter is not None else None
        self.rows: list[Req] = []      # row i -> Req

    def _log(self, msg):
        if self.debug:
            print(f"[sched t={self._t}] {msg}", file=sys.stderr, flush=True)

    def has_rows(self):
        return bool(self.rows)

    def live_rids(self) -> set[int]:
        return {r.rid for r in self.rows}

    # --- prefill mechanism -------------------------------------------------
    def prefill_chunk(self, group: PrefillGroup, cancelled=None):
        """Prefill the group by LENGTH-SUBGROUP: requests of equal length are
        prefilled together (equal length = no padding = no SSM conv contamination
        from pad tokens); odd lengths prefill alone. Fills per-request single-row
        states + first tokens. Returns True when done, or None if a request was
        cancelled mid-prefill (client gone) — the caller then abandons the group.

        ``cancelled(rid)`` (optional) is polled between prefill chunks so a long
        prompt for a departed client stops instead of running to completion.

        Unequal-length prefill in one batch would left/right-pad the short rows
        and the pad tokens leak into the GatedDeltaNet conv — verified to corrupt
        the short row. Grouping by length avoids padding entirely."""
        group.singles = [None] * len(group.reqs)
        group.firsts = [None] * len(group.reqs)
        group.last_h = [None] * len(group.reqs)

        # A single request can reuse a cached prefix (concurrency is rare, so
        # long prompts arrive alone); batched prefill has per-row prefixes and
        # keeps the plain length-grouped path.
        if self.pc is not None and len(group.reqs) == 1:
            self._prefill_one(group, 0, cancelled)
            return None if group.singles[0] is None else True

        by_len: dict[int, list[int]] = {}
        for i, r in enumerate(group.reqs):
            by_len.setdefault(len(r.prompt), []).append(i)
        for L, idxs in by_len.items():
            rids = [group.reqs[i].rid for i in idxs]
            stop = (lambda: cancelled and any(cancelled(r) for r in rids))
            prompts = [group.reqs[i].prompt for i in idxs]
            state, hidden = self.eng.prefill(prompts, chunk=self.chunk,
                                             log=self._log, stop=stop)
            if hidden is None:            # cancelled mid-prefill
                self._log(f"PREFILL CANCELLED rids={rids}")
                return None
            # Chunked prefill (single long prompt) returns only the last chunk's
            # hidden, so the next-token position is that block's last, not L-1.
            pos = min(L - 1, hidden.shape[1] - 1)
            for j, i in enumerate(idxs):
                last_h = hidden[j:j + 1, pos:pos + 1, :]
                first = int(mx.argmax(self.eng.logits(last_h)[0, -1]))
                group.reqs[i].out.append(first)
                group.singles[i] = self.eng.extract_row(state, j)
                group.firsts[i] = first
                group.last_h[i] = last_h
            self._log(f"PREFILL len={L} rids={[group.reqs[i].rid for i in idxs]}")
        return True

    def _prefill_one(self, group, i, cancelled):
        """Prefill one request, reusing a cached prefix when possible. On a hit,
        restore the snapshot and forward only the tail; otherwise cold-prefill.
        Either way, cache the finished state's snapshot for the next turn. Fills
        group.singles[i] / firsts[i] / last_h[i]; leaves singles[i] None if
        cancelled mid-prefill."""
        req = group.reqs[i]
        ids = req.prompt
        eng = self.eng
        # Cache only the stable part (real client content); the trailing L4
        # generation guide differs every turn and must not enter the key.
        sl = req.stable_len if req.stable_len is not None else len(ids)
        match = self.pc.find(ids[:sl])

        # Continue from a reused prefix (restore at the matched boundary), or
        # cold-start from scratch — both then run the remaining tokens chunked,
        # snapshotting SSM at the tail boundaries so THIS turn's deeper prefix is
        # cached for the next turn (a hit must still refresh the cache, else the
        # pool stays stuck at the first turn's shallow boundaries).
        ssm_snaps: dict[int, list] = {}
        stop = (lambda: cancelled and cancelled(req.rid))
        if match is not None:
            full, base_ssm, pos = match.payload
            state = eng.restore_at(full, base_ssm, pos)
            self._log(f"PREFIX HIT reuse={pos}/{len(ids)} tail={len(ids) - pos} "
                      f"rid={req.rid}")
        else:
            state = BatchState(cache=eng._make_empty_cache(), lengths=[0])
            pos = 0
            self._log(f"PREFILL len={len(ids)} rid={req.rid} (cold)")

        h = eng._run_chunked(state, ids[pos:], self.chunk, log=self._log, stop=stop,
                             on_ssm=lambda p, s: ssm_snaps.__setitem__(p, s))
        if h is None:                               # cancelled mid-prefill
            self._log(f"PREFILL CANCELLED rid={req.rid}")
            return
        h = h[:, -1:, :]
        first = int(mx.argmax(eng.logits(h)[0, -1]))
        # Cache this turn's boundary snapshots, all sharing this full KV snapshot.
        # Only boundaries within the stable region: those keys are token-exactly
        # resent next turn, so they align; deeper boundaries (into L4's guide)
        # would poison the key.
        full = eng.clone_state(state)
        for p, ssm in ssm_snaps.items():
            if p <= sl:
                self.pc.store(ids[:p], (full, ssm, p))

        req.out.append(first)
        group.singles[i] = eng.extract_row(state, 0)
        group.firsts[i] = first
        group.last_h[i] = h

    def merge_ready(self, group: PrefillGroup) -> list[tuple[int, int]]:
        """Merge prefilled requests into the live batch. Rebuilds the batch from
        single rows (existing + new) so merge_states always sees clean singles."""
        singles, hs, prims, reqs = [], [], [], []
        for i, req in enumerate(self.rows):        # existing rows -> singles
            singles.append(self.eng.extract_row(self.state, i))
            hs.append(self.h[i:i + 1])
            prims.append(self.primary[i:i + 1])
            reqs.append(req)
        joined = []
        for i, r in enumerate(group.reqs):         # new rows (already singles)
            singles.append(group.singles[i])
            hs.append(group.last_h[i])
            prims.append(mx.array([group.firsts[i]]))
            reqs.append(r)
            joined.append((r.rid, group.firsts[i]))
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

        if finished:
            self._log(f"EXIT {[rows[i].rid for i in finished]}")
            self._keep([i for i in range(B) if i not in finished])

        return emitted

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
