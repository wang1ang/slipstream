"""Prefix-cache policy: block-tree longest-prefix matching + per-pool LRU.

This module has no model knowledge. It only tracks token prefixes and opaque
payloads supplied by the L3 state adapter.

The core invariant is:
  * attention KV is stored as per-block deltas on tree edges;
  * SSM is stored on reusable boundary nodes;
  * a match restores the parent-chain attention blocks plus that node's SSM.

When ``disk_dir`` is set, block records are written in the background. Startup
restores only metadata and loads tensor blobs lazily on cache hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from .disk import AsyncPrefixDiskStore, DiskBlockRecord


@dataclass
class Node:
    pos: int
    attn: Any | None = None
    ssm: Any | None = None
    start: int = 0
    prefix: tuple[int, ...] = ()
    parent: "Node | None" = None
    children: dict[tuple[int, ...], "Node"] = field(default_factory=dict)
    source: str | None = None
    pool: str = "default"
    cached_h: Any | None = None
    touch: int = 0
    disk_key: str | None = None
    reusable: bool = False


@dataclass
class Match:
    prefix_len: int
    payload: Any
    source: str | None = None
    pool: str = "default"


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    entries: list[Node] = field(default_factory=list)


class PrefixCache:
    """Block-tree prefix cache with independent LRU pools.

    ``capacity`` bounds reusable boundary nodes per pool. Attention-only ancestor
    blocks are retained only while a reusable descendant needs them.
    """

    def __init__(
        self,
        capacity: int | dict[str, int] = 8,
        disk_dir: str | os.PathLike | None = None,
        log=None,
    ):
        if isinstance(capacity, dict):
            self.capacity = {str(k): int(v) for k, v in capacity.items()}
            self._default_capacity = max(self.capacity.values(), default=0)
        else:
            self.capacity = {"default": int(capacity)}
            self._default_capacity = int(capacity)

        self._block_root = Node(pos=0, prefix=())
        self._blocks: dict[tuple[int, ...], Node] = {(): self._block_root}
        self._root = _TrieNode()
        self._clock = 0
        self._log = log
        self.disk_dir = Path(disk_dir).expanduser() if disk_dir else None
        self._disk = (
            AsyncPrefixDiskStore(self.disk_dir, log=log) if self.disk_dir else None
        )
        self._load_disk()

    def _debug(self, msg: str) -> None:
        if self._log is not None:
            self._log(f"PREFIX DISK {msg}")

    def _load_disk(self) -> None:
        """Restore record metadata only; tensor blobs stay lazy until a hit."""
        if self.disk_dir is None:
            return
        manifest = self.disk_dir / "manifest.json"
        legacy = self.disk_dir / "prefixcache.pkl"
        if manifest.exists():
            self._debug(f"LOAD SKIP incompatible_format path={manifest}")
        elif legacy.exists():
            self._debug(f"LOAD SKIP incompatible_legacy path={legacy}")
        if self._disk is None:
            return

        loaded = 0
        key_to_node: dict[str | None, Node] = {None: self._block_root}
        for record in sorted(self._disk.records(), key=lambda r: r.pos):
            parent = key_to_node.get(record.parent)
            if parent is None:
                self._debug(f"LOAD SKIP missing_parent key={record.key[:12]}")
                continue
            node = self._node_from_record(record, parent)
            self._blocks[node.prefix] = node
            parent.children[tuple(record.tokens)] = node
            key_to_node[record.key] = node
            self._clock = max(self._clock, record.touch)
            loaded += 1
        if loaded:
            self._evict()
            self._rebuild_index()
            self._debug(f"LOAD records={loaded} path={self.disk_dir}")

    def _node_from_record(self, record: DiskBlockRecord, parent: Node) -> Node:
        prefix = parent.prefix + tuple(record.tokens)
        return Node(
            pos=record.pos,
            start=record.start,
            prefix=prefix,
            parent=parent,
            source=record.source,
            pool=record.pool,
            touch=record.touch,
            disk_key=record.key,
            reusable=record.ssm_spec is not None,
        )

    def _capacity_for(self, pool: str) -> int:
        return self.capacity.get(pool, self._default_capacity)

    def iter_entries(self):
        for node in self._blocks.values():
            if node is not self._block_root and self._is_reusable_node(node):
                yield node

    @staticmethod
    def _is_reusable_node(node: Node) -> bool:
        return node.ssm is not None or node.reusable

    def _entry_count(self, pool: str | None = None) -> int:
        if pool is None:
            return sum(1 for _ in self.iter_entries())
        return sum(1 for node in self.iter_entries() if node.pool == pool)

    def _rebuild_index(self) -> None:
        self._root = _TrieNode()
        for node in self.iter_entries():
            self._index_block_entry(node)

    def _index_block_entry(self, node: Node) -> None:
        cur = self._root
        for tok in node.prefix:
            cur = cur.children.setdefault(tok, _TrieNode())
        cur.entries.append(node)

    def _evict(self) -> None:
        changed = False
        pools = set(self.capacity)
        pools.update(node.pool for node in self.iter_entries())
        for pool in pools:
            while self._entry_count(pool) > self._capacity_for(pool):
                victim = min(
                    (node for node in self.iter_entries() if node.pool == pool),
                    key=lambda node: node.touch,
                    default=None,
                )
                if victim is None:
                    break
                self._drop_reusable(victim)
                self._prune_block(victim)
                changed = True
        if changed:
            self._rebuild_index()

    def _prune_block(self, node: Node) -> None:
        """Drop leaf attention blocks that no reusable descendant needs."""
        while (
            node is not self._block_root
            and not node.children
            and not self._is_reusable_node(node)
        ):
            parent = node.parent
            if parent is None:
                return
            parent.children.pop(tuple(node.prefix[node.start:node.pos]), None)
            self._blocks.pop(node.prefix, None)
            node = parent

    def prune_unreferenced(self) -> None:
        """Drop attention-only leaves that are not anchoring a reusable node."""
        changed = False
        for node in list(self._blocks.values()):
            if node is self._block_root:
                continue
            if not self._is_reusable_node(node) and not node.children:
                self._prune_block(node)
                changed = True
        if changed:
            self._rebuild_index()

    def find(self, token_ids) -> Match | None:
        """Return the deepest reusable prefix of ``token_ids``."""
        token_ids = tuple(token_ids)
        best: Node | None = None
        cur = self._root
        for tok in token_ids:
            cur = cur.children.get(tok)
            if cur is None:
                break
            if cur.entries:
                candidate = max(cur.entries, key=lambda node: node.touch)
                if (
                    best is None
                    or candidate.pos > best.pos
                    or (candidate.pos == best.pos and candidate.touch > best.touch)
                ):
                    best = candidate
        if best is None:
            return None

        self._clock += 1
        best.touch = self._clock
        if not self._load_reusable_payload(best):
            self._drop_reusable(best)
            self._rebuild_index()
            return self.find(token_ids)
        blocks = self._path_blocks(best)
        if blocks is None:
            self._drop_reusable(best)
            self._rebuild_index()
            return self.find(token_ids)

        payload = ("blocks", blocks, best.ssm, best.pos)
        if best.cached_h is not None:
            payload += (best.cached_h,)
        return Match(prefix_len=best.pos, payload=payload, source=best.source,
                     pool=best.pool)

    def _path_blocks(self, node: Node) -> list[Any] | None:
        blocks = []
        cur = node
        while cur is not self._block_root:
            if cur.parent is None or not self._load_attention(cur):
                return None
            blocks.append(cur.attn)
            cur = cur.parent
        blocks.reverse()
        return blocks

    def _disk_record(self, node: Node) -> DiskBlockRecord | None:
        if self._disk is None or node.disk_key is None:
            return None
        return self._disk.get_record(node.disk_key)

    def _drop_reusable(self, node: Node) -> None:
        node.ssm = None
        node.cached_h = None
        node.reusable = False
        node.source = None

    def _load_attention(self, node: Node) -> bool:
        if node.attn is not None:
            return True
        record = self._disk_record(node)
        if record is None:
            return False
        node.attn = self._disk.load_attn(record)
        return node.attn is not None

    def _load_reusable_payload(self, node: Node) -> bool:
        if node.ssm is None:
            record = self._disk_record(node)
            if record is None or record.ssm_spec is None:
                return False
            node.ssm = self._disk.load_ssm(record)
            node.reusable = True
            if node.cached_h is None and record.cached_h_spec is not None:
                node.cached_h = self._disk.load_cached_h(record)
        return node.ssm is not None

    def store_block(
        self,
        full_prefix,
        start: int,
        pos: int,
        attn,
        *,
        ssm=None,
        source: str | None = None,
        pool: str = "default",
        cached_h=None,
    ) -> bool:
        """Add one attention block and optionally make its end reusable."""
        full_prefix = tuple(full_prefix)
        start = int(start)
        pos = int(pos)
        if pos <= start or start < 0 or pos > len(full_prefix):
            raise ValueError("invalid prefix-cache block range")

        parent_prefix = full_prefix[:start]
        prefix = full_prefix[:pos]
        parent = self._blocks.get(parent_prefix)
        if parent is None:
            self._debug(f"STORE BLOCK SKIP missing_parent start={start} pos={pos}")
            return False

        key = tuple(full_prefix[start:pos])
        disk_key = None
        if self._disk is not None:
            disk_key = self._disk.make_block_key(key, parent=parent.disk_key)
        node = self._blocks.get(prefix)
        new_node = node is None
        if node is None:
            node = Node(
                pos=pos,
                start=start,
                prefix=prefix,
                parent=parent,
                attn=attn,
                disk_key=disk_key,
            )
            self._blocks[prefix] = node
        else:
            node.pos = pos
            node.start = start
            node.prefix = prefix
            node.parent = parent
            node.attn = attn
            node.disk_key = disk_key or node.disk_key
        parent.children[key] = node

        if ssm is not None:
            self._clock += 1
            node.ssm = ssm
            node.source = source
            node.pool = str(pool)
            node.cached_h = cached_h
            node.reusable = True
            node.touch = self._clock
            self._evict()
            self._rebuild_index()
        if (
            self._disk is not None
            and node.disk_key is not None
            and (new_node or ssm is not None)
        ):
            self._disk.submit_block(
                key=node.disk_key,
                parent=parent.disk_key,
                tokens=key,
                start=start,
                pos=pos,
                attn=attn,
                ssm=ssm,
                cached_h=cached_h,
                pool=node.pool,
                source=node.source,
                touch=node.touch,
            )
        return True

    def flush(self) -> None:
        if self._disk is not None:
            self._disk.flush()

    def close(self) -> None:
        if self._disk is not None:
            self._disk.close()
