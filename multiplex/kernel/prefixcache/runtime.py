"""L3 prefix-cache runtime hooks.

This module owns when prefix-cache policy/state helpers are invoked during
prefill and decode. The scheduler decides the generation flow; this adapter
decides how prefix-cache find/store/prune hooks attach to that flow.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine import BatchState, Engine
from .policy import PrefixCache
from .state import PrefixCacheState


PROMPT_CACHE_MIN_TOKENS = 4096


class PrefixCacheRuntime:
    def __init__(
        self,
        engine: Engine,
        *,
        drafter=None,
        capacity: int | dict[str, int] = 8,
        disk_dir=None,
        chunk: int = 512,
        log=None,
    ) -> None:
        self.engine = engine
        self.drafter = drafter
        self.chunk = chunk
        self.log = log
        cache_dir = self._cache_dir(disk_dir)
        cache_capacity = capacity if isinstance(capacity, dict) else {
            "prompt": capacity,
            "session": capacity,
        }
        self.cache = PrefixCache(
            capacity=cache_capacity,
            disk_dir=cache_dir,
            log=self._log,
        ) if capacity else None
        self.state = PrefixCacheState(engine) if self.cache is not None else None

    def _log(self, msg: str) -> None:
        if self.log is not None:
            self.log(msg)

    def _cache_dir(self, value):
        if value in (None, False, "", "none", "off"):
            return None
        if value != "auto":
            return value
        model_path = os.path.abspath(os.path.expanduser(self.engine.model_path))
        digest = hashlib.sha256(model_path.encode()).hexdigest()[:12]
        name = os.path.basename(model_path.rstrip(os.sep)) or "model"
        return Path.home() / ".cache" / "multiplex" / "prefixcache" / f"{name}-{digest}"

    def begin_prefill(self, req: Any, group: Any) -> None:
        ids = req.prompt
        match = self.cache.find(ids) if self.cache is not None else None
        self._log_find(req, ids, match)

        group.cached_h = None
        group.cacheable = len(ids) > PROMPT_CACHE_MIN_TOKENS
        if match is None:
            self._cold_prefill(req, group, ids)
            return

        restored = self.state.restore_match(match)
        if self.drafter is not None:
            needs_boundary_h = 0 < restored.pos < len(ids)
            if restored.mtp_blocks is None or (needs_boundary_h and restored.cached_h is None):
                self._cold_prefill(req, group, ids, log=False)
                self._log(
                    f"PREFIX HIT without MTP history; cold rerun rid={req.rid}"
                )
                return
            group.dcache = self.drafter.restore_cache_blocks(restored.mtp_blocks)
            group.mtp_prev_h = restored.cached_h
        group.state = restored.state
        group.pos = restored.pos
        group.cached_h = restored.cached_h
        pos = group.pos
        self._log(f"PREFIX HIT reuse={pos}/{len(ids)} tail={len(ids) - pos} "
                  f"rid={req.rid} source={match.source!r}")
        if group.cached_h is None and pos == len(ids):
            self._cold_prefill(req, group, ids, log=False)
            group.cacheable = False
            self._log(f"PREFIX HIT exact without hidden; cold rerun rid={req.rid}")

    def store_prompt_block(self, req: Any, group: Any, ids: list[int],
                           start: int, end: int, h) -> None:
        if self.cache is None or self.state is None or not self.chunk:
            return
        if end - start != self.chunk:
            return
        if not (group.cacheable or req.session_cache):
            return
        block_payload = self._pack_block(
            self.state.clone_attention_block(group.state, start, end),
            getattr(group, "dcache", None), start, end,
        )
        ssm = None
        source = None
        cached_h = None
        if group.cacheable and end >= PROMPT_CACHE_MIN_TOKENS:
            ssm = self.state.clone_ssm(group.state.cache)
            block = end // self.chunk
            source = f"prompt rid={req.rid} block={block}"
            cached_h = h[:, -1:, :]
            mx.eval(cached_h)
        stored = self.cache.store_block(
            ids, start, end, block_payload, ssm=ssm, source=source,
            pool="prompt", cached_h=cached_h,
        )
        if stored and ssm is not None:
            self._log(f"PREFIX STORE block={end // self.chunk} "
                      f"len={end}/{len(ids)} rid={req.rid}")

    def capture_session_blocks(self, rows: list[Any], state: BatchState, *,
                               dcache=None, h=None) -> None:
        if self.cache is None or self.state is None or not self.chunk:
            return
        for i, req in enumerate(rows):
            if not req.session_cache:
                continue
            pos = state.lengths[i]
            full_len = len(req.prompt) + len(req.out)
            if pos <= len(req.prompt) or pos > full_len or pos % self.chunk:
                continue
            if pos in req.session_cache_pos:
                continue
            single = self.engine.extract_row(state, i)
            start = pos - self.chunk
            prefix = req.prompt + req.out
            if len(prefix) < pos or start < 0:
                continue
            prefix = prefix[:pos]
            block_payload = self._pack_block(
                self.state.clone_attention_block(single, start, pos),
                dcache, start, pos, row=i,
            )
            ssm = self.state.clone_ssm(single.cache)
            cached_h = h[i:i + 1] if h is not None else None
            if cached_h is not None:
                mx.eval(cached_h)
            block = pos // self.chunk
            if self.cache.store_block(
                prefix, start, pos, block_payload, ssm=ssm,
                source=f"session rid={req.rid} block={block}",
                pool="session", cached_h=cached_h,
            ):
                req.session_cache_pos.add(pos)
                self._log(f"SESSION STORE block={block} len={pos} rid={req.rid}")

    def prune_unreferenced(self) -> None:
        if self.cache is not None:
            self.cache.prune_unreferenced()

    def _cold_prefill(self, req: Any, group: Any, ids: list[int], *,
                      log: bool = True) -> None:
        group.state = BatchState(cache=self.engine._make_empty_cache(), lengths=[0])
        group.pos = 0
        group.dcache = self.drafter.make_cache() if self.drafter is not None else None
        group.mtp_prev_h = None
        if log:
            self._log(f"PREFILL len={len(ids)} rid={req.rid} (cold)")

    def _pack_block(self, attn, dcache, start: int, pos: int, *, row=None):
        if self.drafter is None or dcache is None:
            return attn
        mtp = self.drafter.clone_cache_block(dcache, start, pos, row=row)
        return self.state.pack_cache_block(attn, mtp)

    def _log_find(self, req: Any, ids: list[int], match) -> None:
        if self.cache is None:
            return
        chosen = match.prefix_len if match is not None else 0
        source = match.source if match is not None else None
        self._log(f"PREFIX CACHE FIND rid={req.rid} prompt_len={len(ids)} "
                  f"chosen={chosen} pool={getattr(match, 'pool', None)!r} "
                  f"source={source!r}")
