"""L3 prefix-cache state adapter.

This module owns the model-cache mechanics needed by prefix reuse:
extracting attention KV block deltas, snapshotting SSM boundary state, and
rebuilding a single-row BatchState from a policy match. The engine stays focused
on forward/batch primitives; policy stays model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm.generate import _make_cache
from mlx_lm.models.cache import ArraysCache, BatchKVCache

from ..engine import BatchState

if TYPE_CHECKING:
    from ..engine import Engine
    from .policy import Match


COMBINED_BLOCK_VERSION = 1


@dataclass
class RestoredPrefix:
    state: BatchState
    pos: int
    cached_h: Any | None = None
    mtp_blocks: list[Any] | None = None


class PrefixCacheState:
    """Bridge between L3 prefix-cache policy and MLX cache objects."""

    def __init__(self, engine: "Engine"):
        self.engine = engine

    def clone_attention_block(self, state: BatchState, start: int, pos: int):
        """Clone attention-only KV for ``[start:pos]`` from a single-row state."""
        start = int(start)
        pos = int(pos)
        block = []
        leaves = []
        for c in state.cache:
            if isinstance(c, ArraysCache):
                block.append(None)
                continue
            keys, values, length = self._row_view(c)
            if start < 0 or pos > length or pos <= start:
                raise ValueError(
                    f"invalid attention block slice [{start}:{pos}] for length={length}"
                )
            k = keys[..., start:pos, :] + 0
            v = values[..., start:pos, :] + 0
            block.append([k, v])
            leaves.extend([k, v])
        if leaves:
            mx.eval(*leaves)
        return block

    def clone_ssm(self, cache: list[Any]):
        """Clone only SSM/ArraysCache state at a reusable boundary."""
        snap = []
        leaves = []
        for c in cache:
            if isinstance(c, ArraysCache):
                layer = [None if v is None else v + 0 for v in c.cache]
                leaves.extend(v for v in layer if v is not None)
                snap.append(layer)
            else:
                snap.append(None)
        if leaves:
            mx.eval(*leaves)
        return snap

    @staticmethod
    def pack_cache_block(attn, mtp=None):
        # Policy/disk see one opaque block payload; trunk and MTP deltas must
        # move together so restore cannot warm one cache and cold-start the other.
        if mtp is None:
            return attn
        return (COMBINED_BLOCK_VERSION, attn, mtp)

    def split_cache_blocks(self, blocks: list[Any]) -> tuple[list[Any], list[Any] | None]:
        attn_blocks = []
        mtp_blocks: list[Any] | None = []
        for block in blocks:
            if (
                isinstance(block, tuple)
                and len(block) == 3
                and block[0] == COMBINED_BLOCK_VERSION
            ):
                attn_blocks.append(block[1])
                if mtp_blocks is not None:
                    mtp_blocks.append(block[2])
            else:
                attn_blocks.append(block)
                mtp_blocks = None
        return attn_blocks, mtp_blocks

    def restore_match(self, match: "Match") -> RestoredPrefix:
        """Restore the block-tree payload returned by policy.find()."""
        payload = match.payload
        kind = payload[0] if payload else None
        if kind != "blocks":
            raise ValueError(f"unsupported prefix-cache payload: {kind!r}")
        _kind, blocks, ssm, pos = payload[:4]
        cached_h = payload[4] if len(payload) > 4 else None
        attn_blocks, mtp_blocks = self.split_cache_blocks(blocks)
        state = self.restore_blocks(attn_blocks, ssm, pos)
        return RestoredPrefix(
            state=state,
            pos=int(pos),
            cached_h=cached_h,
            mtp_blocks=mtp_blocks,
        )

    def restore_blocks(self, blocks: list[Any], ssm_snapshot, pos: int) -> BatchState:
        """Rebuild state by concatenating attention block deltas and applying SSM."""
        cache = _make_cache(self.engine.model, [0], None)
        for layer_idx, c in enumerate(cache):
            if isinstance(c, ArraysCache):
                layer_ssm = ssm_snapshot[layer_idx]
                c.cache = [None if v is None else v + 0 for v in layer_ssm]
                continue

            parts = [block[layer_idx] for block in blocks if block[layer_idx] is not None]
            if not parts:
                continue
            keys = mx.concatenate([p[0] for p in parts], axis=2) \
                if len(parts) > 1 else parts[0][0] + 0
            values = mx.concatenate([p[1] for p in parts], axis=2) \
                if len(parts) > 1 else parts[0][1] + 0
            mx.eval(keys, values)
            self._assign_attention(c, keys, values)
        return BatchState(cache=cache, lengths=[int(pos)])

    @staticmethod
    def _row_view(c):
        """Return valid single-row attention KV, skipping any left padding."""
        off = c.offset
        if hasattr(off, "shape") and off.shape:
            pad = int(c.left_padding[0])
            end = int(c._idx)
            return c.keys[:, :, pad:end, :], c.values[:, :, pad:end, :], end - pad
        n = int(off)
        return c.keys[..., :n, :], c.values[..., :n, :], n

    @staticmethod
    def _assign_attention(c, keys, values) -> None:
        if isinstance(c, BatchKVCache):
            c.keys, c.values = keys, values
            c.offset = mx.array([keys.shape[2]])
            c.left_padding = mx.array([0])
            c._idx = keys.shape[2]
        else:
            c.state = [keys, values]
