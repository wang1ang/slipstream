"""Prefix KV-cache reuse — skip recomputing a prompt prefix already processed.

Multi-turn / retrying clients resend an ever-growing prompt whose prefix is a
token-identical continuation of an earlier one. Recomputing all of it every turn
is the dominant single-call cost. This package keeps, per past request, snapshots
of the model state taken at chunk boundaries during prefill; a new request reuses
the longest matching snapshot and only prefills the tail.

Split into runtime plus two lower pieces under L3:
  * ``runtime`` — scheduler-facing hooks for find/restore/store/prune timing.
  * ``policy`` — pure logic (no MLX): trie longest-prefix matching over stored
    entries and per-pool LRU eviction. This is what could become a standalone
    library.
  * ``state`` — L3 adapter for MLX cache objects: clone attention blocks,
    snapshot SSM, and restore a matched prefix into a single-row BatchState.
  * L3 scheduler wires runtime into request flow.

The state object it stores is opaque here. ``policy`` only reasons about token
ids and which boundary can be reused; ``state`` knows how MLX caches are cloned
and restored.
"""

from .policy import PrefixCache, Match

__all__ = ["PrefixCache", "Match"]
