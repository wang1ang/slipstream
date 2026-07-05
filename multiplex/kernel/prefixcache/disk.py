"""Low-level disk helpers for PrefixCache persistence.

This module owns tensor-tree encoding, content-addressed blobs, atomic JSON
records, and the bounded background writer used by the in-memory block tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np


RECORD_FORMAT_VERSION = 1


_MLX_TO_NUMPY_DTYPE: dict[str, Any] = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

_NUMPY_TO_MLX_DTYPE: dict[str, Any] = {
    "bool": mx.bool_,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float64,
}


def encode_tree(value: Any, *, block_size: int = 256) -> tuple[Any, dict[str, bytes]]:
    tensors: dict[str, bytes] = {}
    block_size = max(1, int(block_size))

    def put_tensor(raw: bytes) -> str:
        digest = hashlib.sha256(raw).hexdigest()
        tensors.setdefault(digest, raw)
        return digest

    def encode(value: Any) -> Any:
        if value is None:
            return {"k": "none"}
        if isinstance(value, np.integer):
            return {"k": "int", "v": int(value)}
        if isinstance(value, int):
            return {"k": "int", "v": int(value)}
        if isinstance(value, mx.array):
            return encode_tensor(value)
        if isinstance(value, tuple):
            return {"k": "tuple", "v": [encode(x) for x in value]}
        if isinstance(value, list):
            return {"k": "list", "v": [encode(x) for x in value]}
        raise TypeError(f"unsupported prefix cache leaf: {type(value)!r}")

    def encode_tensor(value: mx.array) -> Any:
        shape = [int(dim) for dim in value.shape]
        dtype = _dtype_name(value.dtype)
        if len(shape) >= 3 and shape[2] >= block_size * 2:
            blocks = []
            for start in range(0, shape[2], block_size):
                end = min(shape[2], start + block_size)
                chunk = value[(slice(None), slice(None), slice(start, end), ...)]
                raw, _, chunk_shape = _array_bytes(chunk)
                blocks.append({
                    "sha": put_tensor(raw),
                    "start": start,
                    "end": end,
                    "shape": chunk_shape,
                    "nbytes": len(raw),
                })
            return {
                "k": "blocks",
                "dtype": dtype,
                "shape": shape,
                "axis": 2,
                "block_size": block_size,
                "blocks": blocks,
            }
        raw, dtype, shape = _array_bytes(value)
        return {
            "k": "tensor",
            "sha": put_tensor(raw),
            "dtype": dtype,
            "shape": shape,
            "nbytes": len(raw),
        }

    return encode(value), tensors


def decode_tree(spec: Any, read_blob: Callable[[str], bytes]) -> Any:
    kind = spec.get("k") if isinstance(spec, dict) else None
    if kind == "none":
        return None
    if kind == "int":
        return int(spec["v"])
    if kind == "tuple":
        return tuple(decode_tree(x, read_blob) for x in spec.get("v", []))
    if kind == "list":
        return [decode_tree(x, read_blob) for x in spec.get("v", [])]
    if kind == "tensor":
        return _decode_tensor(spec, read_blob)
    if kind == "blocks":
        chunks = [
            _decode_tensor({**block, "dtype": spec["dtype"]}, read_blob)
            for block in spec.get("blocks", [])
        ]
        if not chunks:
            return mx.zeros(tuple(int(dim) for dim in spec.get("shape") or []))
        arr = mx.concatenate(chunks, axis=int(spec.get("axis", 2)))
        arr = arr.reshape(tuple(int(dim) for dim in spec["shape"]))
        mx.eval(arr)
        return arr
    raise ValueError(f"unsupported prefix cache spec kind: {kind!r}")


class BlobStore:
    """Content-addressed blob store used by prefix-cache records."""

    def __init__(self, base_dir: str | os.PathLike) -> None:
        self.base_dir = Path(base_dir).expanduser()

    def write_blobs(self, tensors: dict[str, bytes]) -> tuple[int, int]:
        """Write blobs by digest. Returns ``(wrote, deduped)`` counts."""
        wrote = deduped = 0
        for digest, raw in tensors.items():
            path = self._blob_path(digest)
            if path.exists():
                deduped += 1
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                os.replace(tmp, path)
                wrote += 1
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        return wrote, deduped

    def read_blob(self, digest: str) -> bytes:
        return self._blob_path(digest).read_bytes()

    def _blob_path(self, digest: str) -> Path:
        return self.base_dir / "blobs" / digest[:2] / f"{digest}.bin"


@dataclass(frozen=True)
class DiskBlockTask:
    """One immutable prefix-cache block write handed to the disk writer."""

    key: str
    parent: str | None
    tokens: tuple[int, ...]
    start: int
    pos: int
    pool: str
    source: str | None
    touch: int
    attn: Any
    ssm: Any | None = None
    cached_h: Any | None = None


@dataclass
class DiskBlockRecord:
    """Metadata-only view of a persisted prefix-cache block."""

    key: str
    parent: str | None
    tokens: tuple[int, ...]
    start: int
    pos: int
    pool: str
    source: str | None
    touch: int
    attn_spec: Any
    ssm_spec: Any | None = None
    cached_h_spec: Any | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DiskBlockRecord":
        if int(data.get("version", 0)) != RECORD_FORMAT_VERSION:
            raise ValueError(f"bad prefix-cache record version: {data.get('version')!r}")
        return cls(
            key=str(data["key"]),
            parent=data.get("parent"),
            tokens=tuple(int(t) for t in data.get("tokens", [])),
            start=int(data["start"]),
            pos=int(data["pos"]),
            pool=str(data.get("pool", "default")),
            source=data.get("source"),
            touch=int(data.get("touch", 0) or 0),
            attn_spec=data.get("attn"),
            ssm_spec=data.get("ssm"),
            cached_h_spec=data.get("cached_h"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": RECORD_FORMAT_VERSION,
            "key": self.key,
            "parent": self.parent,
            "tokens": list(self.tokens),
            "start": int(self.start),
            "pos": int(self.pos),
            "pool": self.pool,
            "source": self.source,
            "touch": int(self.touch),
            "attn": self.attn_spec,
            "ssm": self.ssm_spec,
            "cached_h": self.cached_h_spec,
        }


class AsyncPrefixDiskStore:
    """Async block-record store for PrefixCache persistence.

    The engine thread should hand this store already-cloned/evaluated MLX arrays.
    The writer thread serializes those arrays into content-addressed blobs and
    writes a small JSON record last, so startup can rebuild the tree from records
    without reading tensor bytes.
    """

    def __init__(
        self,
        base_dir: str | os.PathLike,
        *,
        block_size: int = 256,
        max_bytes: int = 10 * 1024**3,
        queue_depth: int = 64,
        cleanup_every: int = 64,
        cleanup_interval: float = 30.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).expanduser()
        self.block_size = max(1, int(block_size))
        self.max_bytes = int(max_bytes)
        self.cleanup_every = max(1, int(cleanup_every))
        self.cleanup_interval = float(cleanup_interval)
        self._log = log
        self._blob_store = BlobStore(self.base_dir)
        self._records: dict[str, DiskBlockRecord] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[DiskBlockTask | None] = queue.Queue(
            maxsize=max(1, int(queue_depth))
        )
        self._closed = threading.Event()
        self._writes_since_cleanup = 0
        self._last_cleanup = time.monotonic()
        self._stats = {
            "submitted": 0,
            "dropped": 0,
            "written": 0,
            "errors": 0,
            "cleanups": 0,
            "records_deleted": 0,
            "blobs_deleted": 0,
        }

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._load_records()
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="multiplex-prefix-disk",
            daemon=True,
        )
        self._thread.start()

    def make_block_key(
        self,
        tokens: tuple[int, ...] | list[int],
        *,
        parent: str | None = None,
    ) -> str:
        h = hashlib.sha256()
        h.update(b"multiplex-prefix-block-v1\0")
        h.update((parent or "").encode("ascii"))
        h.update(b"\0")
        for token in tokens:
            h.update(int(token).to_bytes(8, byteorder="little", signed=True))
        return h.hexdigest()

    def submit_block(
        self,
        *,
        key: str,
        parent: str | None,
        tokens: tuple[int, ...] | list[int],
        start: int,
        pos: int,
        attn: Any,
        ssm: Any | None = None,
        cached_h: Any | None = None,
        pool: str = "default",
        source: str | None = None,
        touch: int = 0,
    ) -> bool:
        """Queue one block for asynchronous persistence.

        Returns False when the bounded queue is full or the store is closed. A
        skipped disk write does not affect the in-memory cache.
        """
        if self._closed.is_set():
            return False
        task = DiskBlockTask(
            key=str(key),
            parent=parent,
            tokens=tuple(int(t) for t in tokens),
            start=int(start),
            pos=int(pos),
            pool=str(pool),
            source=source,
            touch=int(touch),
            attn=attn,
            ssm=ssm,
            cached_h=cached_h,
        )
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
            self._debug(f"WRITE DROP queue_full key={task.key[:12]}")
            return False
        with self._lock:
            self._stats["submitted"] += 1
        return True

    def records(self) -> list[DiskBlockRecord]:
        with self._lock:
            return list(self._records.values())

    def get_record(self, key: str) -> DiskBlockRecord | None:
        with self._lock:
            return self._records.get(key)

    def load_attn(self, record: DiskBlockRecord) -> Any:
        return decode_tree(record.attn_spec, self._blob_store.read_blob)

    def load_ssm(self, record: DiskBlockRecord) -> Any | None:
        if record.ssm_spec is None:
            return None
        return decode_tree(record.ssm_spec, self._blob_store.read_blob)

    def load_cached_h(self, record: DiskBlockRecord) -> Any | None:
        if record.cached_h_spec is None:
            return None
        return decode_tree(record.cached_h_spec, self._blob_store.read_blob)

    def flush(self) -> None:
        self._queue.join()

    def close(self, *, wait: bool = True) -> None:
        if self._closed.is_set():
            return
        if wait:
            self.flush()
        self._closed.set()
        try:
            if wait:
                self._queue.put(None)
            else:
                self._queue.put_nowait(None)
        except queue.Full:
            pass
        if wait:
            self._thread.join(timeout=5.0)

    @property
    def pending_writes(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def cleanup(self) -> dict[str, int]:
        """Enforce max_bytes and garbage-collect unreferenced blobs."""
        if self.max_bytes <= 0:
            return {"records_deleted": 0, "blobs_deleted": 0, "bytes_before": 0}

        records = self._scan_record_files()
        bytes_before = self._disk_bytes()
        if bytes_before <= self.max_bytes:
            blobs_deleted = self._delete_unreferenced_blobs(records)
            return {
                "records_deleted": 0,
                "blobs_deleted": blobs_deleted,
                "bytes_before": bytes_before,
            }

        target = int(self.max_bytes * 0.9)
        by_key = {record.key: record for record in records}
        keep: set[str] = set()
        for record in sorted(records, key=lambda r: r.touch, reverse=True):
            chain = self._ancestor_chain(record, by_key)
            if not chain:
                continue
            proposed = keep | chain
            proposed_bytes = self._referenced_bytes(proposed, by_key)
            if proposed_bytes <= target or not keep:
                keep = proposed

        deleted = 0
        for record in records:
            if record.key in keep:
                continue
            path = self._record_path(record.key)
            try:
                path.unlink(missing_ok=True)
                deleted += 1
            except OSError as e:
                self._debug(f"CLEANUP record_delete_failed key={record.key[:12]} error={e!r}")

        kept_records = [record for record in records if record.key in keep]
        blobs_deleted = self._delete_unreferenced_blobs(kept_records)
        self._cleanup_tmp_files()
        self._load_records()
        with self._lock:
            self._stats["cleanups"] += 1
            self._stats["records_deleted"] += deleted
            self._stats["blobs_deleted"] += blobs_deleted
        self._debug(
            f"CLEANUP records_deleted={deleted} blobs_deleted={blobs_deleted} "
            f"bytes_before={bytes_before}"
        )
        return {
            "records_deleted": deleted,
            "blobs_deleted": blobs_deleted,
            "bytes_before": bytes_before,
        }

    def _writer_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._write_task(task)
                self._maybe_cleanup()
            except Exception as e:
                with self._lock:
                    self._stats["errors"] += 1
                self._debug(f"WRITE FAILED error={e!r}")
            finally:
                self._queue.task_done()

    def _write_task(self, task: DiskBlockTask) -> None:
        attn_spec, tensors = encode_tree(task.attn, block_size=self.block_size)
        ssm_spec = None
        cached_h_spec = None
        all_tensors = dict(tensors)
        if task.ssm is not None:
            ssm_spec, tensors = encode_tree(task.ssm, block_size=self.block_size)
            all_tensors.update(tensors)
        if task.cached_h is not None:
            cached_h_spec, tensors = encode_tree(task.cached_h, block_size=self.block_size)
            all_tensors.update(tensors)

        self._blob_store.write_blobs(all_tensors)
        record = DiskBlockRecord(
            key=task.key,
            parent=task.parent,
            tokens=task.tokens,
            start=task.start,
            pos=task.pos,
            pool=task.pool,
            source=task.source,
            touch=task.touch,
            attn_spec=attn_spec,
            ssm_spec=ssm_spec,
            cached_h_spec=cached_h_spec,
        )
        write_json_atomic(self._record_path(task.key), record.to_json())
        with self._lock:
            self._records[task.key] = record
            self._stats["written"] += 1
            self._writes_since_cleanup += 1

    def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        with self._lock:
            enough_writes = self._writes_since_cleanup >= self.cleanup_every
            enough_time = now - self._last_cleanup >= self.cleanup_interval
            if not enough_writes and not enough_time:
                return
            self._writes_since_cleanup = 0
            self._last_cleanup = now
        self.cleanup()

    def _load_records(self) -> None:
        records = self._scan_record_files()
        with self._lock:
            self._records = {record.key: record for record in records}

    def _scan_record_files(self) -> list[DiskBlockRecord]:
        out: list[DiskBlockRecord] = []
        for path in self._record_root().glob("*/*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out.append(DiskBlockRecord.from_json(data))
            except Exception as e:
                self._debug(f"LOAD SKIP path={path} error={e!r}")
        return out

    def _record_root(self) -> Path:
        return self.base_dir / "records"

    def _record_path(self, key: str) -> Path:
        return self._record_root() / key[:2] / f"{key}.json"

    def _ancestor_chain(
        self,
        record: DiskBlockRecord,
        by_key: dict[str, DiskBlockRecord],
    ) -> set[str]:
        chain: set[str] = set()
        cur: DiskBlockRecord | None = record
        while cur is not None:
            if cur.key in chain:
                return set()
            chain.add(cur.key)
            cur = by_key.get(cur.parent) if cur.parent else None
        return chain

    def _referenced_bytes(
        self,
        keep: set[str],
        by_key: dict[str, DiskBlockRecord],
    ) -> int:
        total = 0
        blob_refs: set[str] = set()
        for key in keep:
            record = by_key.get(key)
            if record is None:
                continue
            path = self._record_path(key)
            try:
                total += path.stat().st_size
            except OSError:
                pass
            _collect_blob_refs(record.attn_spec, blob_refs)
            _collect_blob_refs(record.ssm_spec, blob_refs)
            _collect_blob_refs(record.cached_h_spec, blob_refs)
        for digest in blob_refs:
            try:
                total += self._blob_store._blob_path(digest).stat().st_size
            except OSError:
                pass
        return total

    def _delete_unreferenced_blobs(self, records: list[DiskBlockRecord]) -> int:
        keep: set[str] = set()
        for record in records:
            _collect_blob_refs(record.attn_spec, keep)
            _collect_blob_refs(record.ssm_spec, keep)
            _collect_blob_refs(record.cached_h_spec, keep)

        deleted = 0
        blob_root = self.base_dir / "blobs"
        if not blob_root.exists():
            return 0
        for path in blob_root.glob("*/*.bin"):
            digest = path.stem
            if digest in keep:
                continue
            try:
                path.unlink()
                deleted += 1
            except OSError as e:
                self._debug(f"CLEANUP blob_delete_failed path={path} error={e!r}")
        return deleted

    def _cleanup_tmp_files(self) -> None:
        if not self.base_dir.exists():
            return
        now = time.time()
        for path in self.base_dir.rglob("*"):
            if not path.is_file():
                continue
            name = path.name
            if not (name.endswith(".tmp") or ".tmp-" in name):
                continue
            try:
                if now - path.stat().st_mtime > 60.0:
                    path.unlink()
            except OSError:
                pass

    def _disk_bytes(self) -> int:
        total = 0
        if not self.base_dir.exists():
            return 0
        for path in self.base_dir.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    def _debug(self, msg: str) -> None:
        if self._log is not None:
            self._log(f"PREFIX DISK {msg}")


def write_json_atomic(path: str | os.PathLike, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    os.replace(tmp, path)


def _collect_blob_refs(spec: Any, out: set[str]) -> None:
    if spec is None:
        return
    if isinstance(spec, dict):
        sha = spec.get("sha")
        if isinstance(sha, str):
            out.add(sha)
        for value in spec.values():
            _collect_blob_refs(value, out)
        return
    if isinstance(spec, list):
        for value in spec:
            _collect_blob_refs(value, out)


def _array_bytes(value: mx.array) -> tuple[bytes, str, list[int]]:
    mx.eval(value)
    dtype = _dtype_name(value.dtype)
    shape = [int(dim) for dim in value.shape]
    if dtype == "bfloat16":
        value = value.view(mx.uint16)
        mx.eval(value)
    return bytes(memoryview(value)), dtype, shape


def _decode_tensor(spec: dict[str, Any], read_blob: Callable[[str], bytes]) -> mx.array:
    dtype = str(spec["dtype"])
    raw = read_blob(str(spec["sha"]))
    if dtype == "bfloat16":
        arr = mx.array(np.frombuffer(raw, dtype=np.uint16)).view(mx.bfloat16)
    else:
        arr = mx.array(
            np.frombuffer(raw, dtype=_MLX_TO_NUMPY_DTYPE[dtype]),
            dtype=_NUMPY_TO_MLX_DTYPE[dtype],
        )
    arr = arr.reshape(tuple(int(dim) for dim in spec["shape"]))
    mx.eval(arr)
    return arr


def _dtype_name(dtype: Any) -> str:
    raw = str(dtype)
    if raw.startswith("mlx.core."):
        raw = raw.removeprefix("mlx.core.")
    if raw == "bfloat16" or raw in _MLX_TO_NUMPY_DTYPE:
        return raw
    raise TypeError(f"unsupported MLX dtype for prefix cache SSD: {dtype!r}")
