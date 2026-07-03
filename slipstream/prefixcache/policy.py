"""Prefix-cache policy: trie longest-prefix matching + per-pool LRU.

The matching policy has no model knowledge: it reasons over token ids and
opaque payloads.  When disk persistence is enabled, the payloads are handed to
the SSD store for tensor/blob serialization.

The key invariant: a prefix is reusable ONLY at a position where an SSM snapshot
was taken. SSM state is fixed-size and cannot be truncated, so it must have been
captured at that exact boundary; attention KV is truncatable from a shared full
snapshot. So prefix matching and SSM snapshots are bound: the set of snapshot
positions IS the set of reusable positions. find() returns the deepest snapshot
whose covered prefix is a prefix of the new prompt.

Each snapshot carries its own opaque payload (L3 stores the SSM state there and
knows the shared full-KV to truncate). policy indexes prompt and session block
nodes in one token trie, but accounts LRU capacity separately by pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import pickle
from typing import Any

from .disk import PrefixCacheDiskStore


def common_prefix_len(a, b) -> int:
    """Length of the shared leading run of two token sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class Node:
    pos: int                  # reusable boundary inside ``full_prefix``
    ssm: Any                  # SSM state captured exactly at ``pos``
    source: str | None = None
    pool: str = "default"
    cached_h: Any | None = None
    touch: int = 0
    ssm_spec: Any | None = None
    cached_h_spec: Any | None = None
    dirty: bool = True


@dataclass
class Snapshot:
    full_prefix: tuple[int, ...]  # full KV's token sequence
    full: Any                     # opaque: L3's full attention/KV snapshot
    nodes: list[Node]             # many SSM boundaries over ``full_prefix``
    group_id: str | None = None
    full_spec: Any | None = None
    dirty_full: bool = True


@dataclass
class Match:
    prefix_len: int           # reuse the first prefix_len tokens
    payload: Any              # the chosen snapshot's payload
    source: str | None = None
    pool: str = "default"


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    entries: list[tuple[Snapshot, Node]] = field(default_factory=list)


class PrefixCache:
    """A pool of chunk-boundary snapshots with trie longest-prefix lookup and
    per-pool LRU eviction. ``capacity`` bounds the number of snapshots kept per
    pool (each holds an SSM state, so keep it modest)."""

    def __init__(self, capacity: int | dict[str, int] = 8,
                 disk_dir: str | os.PathLike | None = None, log=None):
        if isinstance(capacity, dict):
            self.capacity = {str(k): int(v) for k, v in capacity.items()}
            self._default_capacity = max(self.capacity.values(), default=0)
        else:
            self.capacity = {"default": int(capacity)}
            self._default_capacity = int(capacity)
        self._snaps: list[Snapshot] = []
        self._root = _TrieNode()
        self._clock = 0
        self._log = log
        self._dirty = False
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        self._store = PrefixCacheDiskStore(self.disk_dir, log=log) \
            if self.disk_dir else None
        self._legacy_file = self.disk_dir / "prefixcache.pkl" if self.disk_dir else None
        self._load_disk()

    def _debug(self, msg: str) -> None:
        if self._log is not None:
            self._log(f"PREFIX DISK {msg}")

    def _load_disk(self) -> None:
        if self._store is not None:
            try:
                loaded = self._store.load(Snapshot, Node)
                if loaded is not None:
                    self._snaps, self._clock = loaded
                    self._evict()
                    self._rebuild_index()
                    self._debug(f"LOAD entries={self._entry_count()} "
                                f"groups={len(self._snaps)} "
                                f"path={self._store.manifest_path}")
                    return
            except Exception as e:
                self._snaps = []
                self._debug(f"LOAD FAILED path={self._store.manifest_path} error={e!r}")
                return

        if self._legacy_file is None or not self._legacy_file.exists():
            return
        try:
            with self._legacy_file.open("rb") as f:
                data = pickle.load(f)
            raw = data.get("snaps", []) if isinstance(data, dict) else []
            self._snaps = self._coerce_loaded(raw)
            self._evict()
            self._rebuild_index()
            self._debug(f"LOAD LEGACY entries={self._entry_count()} "
                        f"groups={len(self._snaps)} path={self._legacy_file}")
        except Exception as e:
            self._snaps = []
            self._debug(f"LOAD LEGACY FAILED path={self._legacy_file} error={e!r}")

    def _save_disk(self, *, wait: bool = False) -> None:
        # Load-on-start stays enabled, but writes are disabled for now because
        # synchronous SSD persistence was hurting interactive batching latency.
        _ = wait
        if self._store is None:
            return
        self._debug("SAVE SKIP writes_disabled=True")
        self._dirty = False

    def _coerce_loaded(self, raw) -> list[Snapshot]:
        groups: list[Snapshot] = []
        max_touch = 0
        for item in raw:
            if hasattr(item, "full_prefix") and hasattr(item, "nodes"):
                item.dirty_full = getattr(item, "full_spec", None) is None
                item.full_spec = getattr(item, "full_spec", None)
                item.group_id = getattr(item, "group_id", None)
                groups.append(item)
                for node in item.nodes:
                    node.pool = getattr(node, "pool", None) or \
                        self._pool_from_source(getattr(node, "source", None))
                    node.dirty = getattr(node, "ssm_spec", None) is None
                    node.ssm_spec = getattr(node, "ssm_spec", None)
                    node.cached_h_spec = getattr(node, "cached_h_spec", None)
                    max_touch = max(max_touch, getattr(node, "touch", 0))
                continue
            # Migrate v1 snapshots: they only stored the reusable prefix, not the
            # full prompt prefix, so each old entry becomes its own one-node group.
            if hasattr(item, "prefix") and hasattr(item, "payload"):
                payload = item.payload
                if not isinstance(payload, tuple) or len(payload) < 3:
                    continue
                full, ssm, pos = payload[:3]
                cached_h = payload[3] if len(payload) > 3 else None
                self._clock += 1
                source = getattr(item, "source", None)
                node = Node(pos=pos, ssm=ssm, source=getattr(item, "source", None),
                            pool=self._pool_from_source(source),
                            cached_h=cached_h, touch=self._clock)
                groups.append(Snapshot(full_prefix=tuple(item.prefix), full=full,
                                       nodes=[node]))
        self._clock = max(self._clock, max_touch)
        return groups

    @staticmethod
    def _pool_from_source(source: str | None) -> str:
        if source and source.startswith("session "):
            return "session"
        if source and source.startswith("prompt "):
            return "prompt"
        return "default"

    def _capacity_for(self, pool: str) -> int:
        return self.capacity.get(pool, self._default_capacity)

    def _entry_count(self, pool: str | None = None) -> int:
        if pool is None:
            return sum(len(s.nodes) for s in self._snaps)
        return sum(1 for _group_i, _snap, node in self.iter_entries()
                   if node.pool == pool)

    def iter_entries(self):
        for group_i, snap in enumerate(self._snaps):
            for node in snap.nodes:
                yield group_i, snap, node

    def _rebuild_index(self) -> None:
        self._root = _TrieNode()
        for snap in self._snaps:
            for node in snap.nodes:
                self._index_entry(snap, node)

    def _index_entry(self, snap: Snapshot, node: Node) -> None:
        cur = self._root
        for tok in snap.full_prefix[:node.pos]:
            cur = cur.children.setdefault(tok, _TrieNode())
        cur.entries.append((snap, node))

    def _evict(self) -> None:
        changed = False
        pools = set(self.capacity)
        pools.update(node.pool for _group_i, _snap, node in self.iter_entries())
        for pool in pools:
            while self._entry_count(pool) > self._capacity_for(pool):
                victim_group = victim_node = None
                for snap in self._snaps:
                    for node in snap.nodes:
                        if node.pool != pool:
                            continue
                        if victim_node is None or node.touch < victim_node.touch:
                            victim_group, victim_node = snap, node
                if victim_group is None or victim_node is None:
                    break
                victim_group.nodes.remove(victim_node)
                if not victim_group.nodes:
                    self._snaps.remove(victim_group)
                changed = True
        if changed:
            self._rebuild_index()

    def find(self, token_ids) -> Match | None:
        """Deepest snapshot whose covered prefix is a full prefix of
        ``token_ids`` — i.e. reuse only at a boundary that actually has an SSM
        snapshot. None if none apply (cold prefill). Marks the winner MRU; the
        losers (e.g. shallow snapshots) age out."""
        token_ids = tuple(token_ids)
        best: tuple[Snapshot, Node] | None = None
        cur = self._root
        for tok in token_ids:
            cur = cur.children.get(tok)
            if cur is None:
                break
            if cur.entries:
                candidate = max(cur.entries, key=lambda item: item[1].touch)
                if (best is None or candidate[1].pos > best[1].pos
                        or (candidate[1].pos == best[1].pos
                            and candidate[1].touch > best[1].touch)):
                    best = candidate
        if best is None:
            return None
        snap, node = best
        self._clock += 1
        node.touch = self._clock
        payload = (snap.full, node.ssm, node.pos)
        if node.cached_h is not None:
            payload += (node.cached_h,)
        return Match(prefix_len=node.pos, payload=payload, source=node.source,
                     pool=node.pool)

    def flush(self, *, wait: bool = False) -> None:
        """Persist the current cache to disk, if persistence is enabled."""
        if not self._dirty:
            return
        self._save_disk(wait=wait)

    def store(self, full_prefix, payload, *, source: str | None = None,
              pool: str = "default", save: bool = True) -> None:
        """Add or replace one snapshot, evicting LRU past capacity."""
        pool = str(pool)
        full_prefix = tuple(full_prefix)
        full, ssm, pos = payload[:3]
        cached_h = payload[3] if len(payload) > 3 else None
        group = next((s for s in self._snaps if s.full_prefix == full_prefix), None)
        if group is None:
            group = Snapshot(full_prefix=full_prefix, full=full, nodes=[])
            self._snaps.append(group)
        else:
            if group.full is not full:
                group.full = full
                group.dirty_full = True
            group.nodes = [n for n in group.nodes
                           if not (n.pos == pos and n.pool == pool)]
        self._clock += 1
        group.nodes.append(Node(pos=pos, ssm=ssm, source=source,
                                pool=pool, cached_h=cached_h,
                                touch=self._clock))
        self._dirty = True
        self._evict()
        self._rebuild_index()
        if save:
            self._save_disk()

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
