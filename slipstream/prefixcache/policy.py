"""Pure prefix-cache policy: longest-prefix matching + per-snapshot LRU.

No MLX, no engine, no model knowledge — only token ids and opaque payloads.

The key invariant: a prefix is reusable ONLY at a position where an SSM snapshot
was taken. SSM state is fixed-size and cannot be truncated, so it must have been
captured at that exact boundary; attention KV is truncatable from a shared full
snapshot. So prefix matching and SSM snapshots are bound: the set of snapshot
positions IS the set of reusable positions. find() returns the deepest snapshot
whose covered prefix is a prefix of the new prompt.

Each snapshot carries its own opaque payload (L3 stores the SSM state there and
knows the shared full-KV to truncate). policy only reasons about token prefixes
and picks which snapshot to reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def common_prefix_len(a, b) -> int:
    """Length of the shared leading run of two token sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class Snapshot:
    prefix: tuple[int, ...]   # token ids this snapshot's SSM state covers
    payload: Any              # opaque: L3's (full_kv, ssm_state, pos)


@dataclass
class Match:
    prefix_len: int           # reuse the first prefix_len tokens
    payload: Any              # the chosen snapshot's payload


class PrefixCache:
    """A pool of chunk-boundary snapshots with longest-prefix lookup and
    per-snapshot LRU eviction. ``capacity`` bounds the number of snapshots kept
    (each holds an SSM state, so keep it modest)."""

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._snaps: list[Snapshot] = []   # least-recently-selected first

    def find(self, token_ids) -> Match | None:
        """Deepest snapshot whose covered prefix is a full prefix of
        ``token_ids`` — i.e. reuse only at a boundary that actually has an SSM
        snapshot. None if none apply (cold prefill). Marks the winner MRU; the
        losers (e.g. shallow snapshots) age out."""
        token_ids = tuple(token_ids)
        best: Snapshot | None = None
        for s in self._snaps:
            if len(s.prefix) <= len(token_ids) and \
                    common_prefix_len(s.prefix, token_ids) == len(s.prefix):
                if best is None or len(s.prefix) > len(best.prefix):
                    best = s
        if best is None:
            return None
        self._snaps.remove(best)
        self._snaps.append(best)
        return Match(prefix_len=len(best.prefix), payload=best.payload)

    def store(self, prefix, payload) -> None:
        """Add or replace one snapshot, evicting LRU past capacity."""
        prefix = tuple(prefix)
        self._snaps = [s for s in self._snaps if s.prefix != prefix]
        self._snaps.append(Snapshot(prefix=prefix, payload=payload))
        while len(self._snaps) > self.capacity:
            self._snaps.pop(0)
