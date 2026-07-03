"""Prefix KV-cache reuse — skip recomputing a prompt prefix already processed.

Multi-turn / retrying clients resend an ever-growing prompt whose prefix is a
token-identical continuation of an earlier one. Recomputing all of it every turn
is the dominant single-call cost. This package keeps, per past request, snapshots
of the model state taken at chunk boundaries during prefill; a new request reuses
the longest matching snapshot and only prefills the tail.

Split in two:
  * ``policy`` — pure logic (no MLX): trie longest-prefix matching over stored
    entries and per-pool LRU eviction. This is what could become a standalone
    library.
  * the state clone/restore lives in the engine (L1); L3 wires them together.

The state object it stores is opaque here — the caller (L3) provides already
cloned per-layer snapshots and knows how to restore them. policy only reasons
about token ids and which snapshot position to reuse.
"""

from .policy import PrefixCache, Match, Snapshot, common_prefix_len

__all__ = ["PrefixCache", "Match", "Snapshot", "common_prefix_len"]
