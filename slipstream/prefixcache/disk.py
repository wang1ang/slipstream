"""Small SSD persistence layer for PrefixCache.

Data layout:
  manifest.json          small metadata and tree specs
  blobs/xx/<sha256>.bin  content-addressed tensor bytes

It is intentionally synchronous and compact.  The big win is that unchanged KV
blobs are not rewritten when a new SSM boundary is added.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np


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


class PrefixCacheDiskStore:
    FORMAT_VERSION = 4

    def __init__(self, base_dir: str | os.PathLike, *,
                 log: Callable[[str], None] | None = None,
                 block_size: int = 256,
                 queue_depth: int = 16) -> None:
        del queue_depth
        self.base_dir = Path(base_dir).expanduser()
        self.manifest_path = self.base_dir / "manifest.json"
        self.block_size = max(1, int(block_size))
        self._log = log

    def load(self, snapshot_cls, node_cls) -> tuple[list[Any], int] | None:
        if not self.manifest_path.exists():
            return None
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(data.get("version", 0)) != self.FORMAT_VERSION:
            raise ValueError(f"bad prefix cache manifest version: {data.get('version')!r}")

        groups = []
        clock = int(data.get("clock", 0) or 0)
        for rec in data.get("groups", []):
            try:
                full = decode_tree(rec["full"], self._read_blob)
                nodes = []
                for node_rec in rec.get("nodes", []):
                    cached_h_spec = node_rec.get("cached_h")
                    source = node_rec.get("source")
                    node = node_cls(
                        pos=int(node_rec["pos"]),
                        ssm=decode_tree(node_rec["ssm"], self._read_blob),
                        source=source,
                        pool=node_rec.get("pool") or _pool_from_source(source),
                        cached_h=None if cached_h_spec is None else
                        decode_tree(cached_h_spec, self._read_blob),
                        touch=int(node_rec.get("touch", 0) or 0),
                    )
                    node.ssm_spec = node_rec["ssm"]
                    node.cached_h_spec = cached_h_spec
                    node.dirty = False
                    nodes.append(node)
                    clock = max(clock, int(node.touch))
                snap = snapshot_cls(
                    full_prefix=tuple(int(t) for t in rec.get("full_prefix", [])),
                    full=full,
                    nodes=nodes,
                )
                snap.group_id = str(rec["group_id"])
                snap.full_spec = rec["full"]
                snap.dirty_full = False
                groups.append(snap)
            except Exception as e:
                if self._log is not None:
                    self._log(f"PREFIX DISK LOAD SKIP group={rec.get('group_id')!r} error={e!r}")
        return groups, clock

    def save(self, snaps: list[Any], *, clock: int, wait: bool = False) -> dict[str, int]:
        del wait
        self.base_dir.mkdir(parents=True, exist_ok=True)
        encoded = wrote = deduped = 0

        for snap in snaps:
            if not getattr(snap, "group_id", None):
                snap.group_id = _prefix_digest(snap.full_prefix)
            if getattr(snap, "dirty_full", True) or getattr(snap, "full_spec", None) is None:
                snap.full_spec, tensors = encode_tree(snap.full, block_size=self.block_size)
                w, d = self._write_blobs(tensors)
                wrote += w
                deduped += d
                encoded += 1
                snap.dirty_full = False
            for node in snap.nodes:
                if not getattr(node, "dirty", True) and getattr(node, "ssm_spec", None) is not None:
                    continue
                node.ssm_spec, tensors = encode_tree(node.ssm, block_size=self.block_size)
                w, d = self._write_blobs(tensors)
                wrote += w
                deduped += d
                if node.cached_h is None:
                    node.cached_h_spec = None
                else:
                    node.cached_h_spec, tensors = encode_tree(
                        node.cached_h, block_size=self.block_size
                    )
                    w, d = self._write_blobs(tensors)
                    wrote += w
                    deduped += d
                encoded += 1
                node.dirty = False

        manifest = {
            "version": self.FORMAT_VERSION,
            "clock": int(clock),
            "block_size": self.block_size,
            "groups": [self._group_record(s) for s in snaps],
        }
        self._write_json_atomic(self.manifest_path, manifest)
        return {
            "groups": len(snaps),
            "entries": sum(len(s.nodes) for s in snaps),
            "encoded": encoded,
            "wrote": wrote,
            "deduped": deduped,
        }

    def close(self) -> None:
        pass

    def _group_record(self, snap) -> dict[str, Any]:
        return {
            "group_id": getattr(snap, "group_id", None) or _prefix_digest(snap.full_prefix),
            "full_prefix": list(snap.full_prefix),
            "full_len": len(snap.full_prefix),
            "full": snap.full_spec,
            "nodes": [
                {
                    "pos": int(node.pos),
                    "source": node.source,
                    "pool": getattr(node, "pool", "default"),
                    "touch": int(node.touch),
                    "ssm": node.ssm_spec,
                    "cached_h": node.cached_h_spec,
                }
                for node in snap.nodes
            ],
        }

    def _write_blobs(self, tensors: dict[str, bytes]) -> tuple[int, int]:
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

    def _read_blob(self, digest: str) -> bytes:
        return self._blob_path(digest).read_bytes()

    def _blob_path(self, digest: str) -> Path:
        return self.base_dir / "blobs" / digest[:2] / f"{digest}.bin"

    @staticmethod
    def _write_json_atomic(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        os.replace(tmp, path)


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


def _prefix_digest(prefix: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for token in prefix:
        h.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return h.hexdigest()


def _pool_from_source(source: str | None) -> str:
    if source and source.startswith("session "):
        return "session"
    if source and source.startswith("prompt "):
        return "prompt"
    return "default"
