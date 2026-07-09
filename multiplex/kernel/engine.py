"""L1 — engine layer.

The lowest layer. Its only job: run correct forward passes and manage the
batched cache. It does NOT sample or make scheduling decisions. It gives
whatever token tensor it is fed a correct forward.

Two capabilities, nothing more:
  * serial prefill: feed one prompt piece into one row.
  * batched decode/verify: feed q tokens per row and return all q positions'
    logits (q=1 and q>1 are the same primitive here).

Correctness facts (verified by experiment):
  * Batched decode/verify of this hybrid SSM+attention model is numerically
    EXACT vs single-sequence when rows are equal length. Verified token-for-token.
  * Residual divergence between a batched row and the same sequence run alone is
    pure floating-point accumulation (batch reduction order differs from B=1).
    This is inherent to batched inference — NOT a bug — and both trajectories are
    valid. It shows up only after many steps as an occasional token flip.
  * The cache is ours to roll back: ``snapshot_ssm`` / ``restore_ssm`` save &
    restore SSM recurrent state (it can't be trimmed), and ``trim_attention``
    trims attention KV.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import _make_cache
from mlx_lm.models.cache import ArraysCache, BatchKVCache


def _resolve_load_path(model_path: str) -> str:
    """An MTPLX assistant-pair bundle has no config.json at its root — the
    runnable text model lives under a subdirectory named by mtplx_pair.json's
    ``layout.target`` (default ``target/``). Point the loader there. A plain
    single-model directory (config.json at root) is returned unchanged."""
    pair = os.path.join(model_path, "mtplx_pair.json")
    if not os.path.isfile(pair):
        return model_path
    try:
        layout = (json.load(open(pair)).get("layout") or {})
    except (OSError, json.JSONDecodeError):
        layout = {}
    target = os.path.join(model_path, str(layout.get("target") or "target"))
    if os.path.isfile(os.path.join(target, "config.json")):
        return target
    return model_path


@dataclass
class BatchState:
    """Batched cache + per-row bookkeeping for B aligned sequences."""

    cache: list[Any]        # batched per-layer caches (BatchKVCache / ArraysCache)
    lengths: list[int]      # committed token count per row

    @property
    def batch_size(self) -> int:
        return len(self.lengths)


class Engine:
    """Loads a model. Runs correct batched forwards. Nothing else."""

    def __init__(self, model_path: str):
        # A local-path argument that doesn't exist would otherwise be treated as
        # an HF repo id and fail with an opaque huggingface validation error —
        # catch it here with a clear message. Only absolute (/…) or home/relative
        # (~ ./ ../) forms are treated as local paths; a bare "namespace/repo" is
        # a valid HF id and is left for mlx-lm to download.
        looks_local = model_path.startswith(("/", "~", "./", "../"))
        if looks_local:
            expanded = os.path.expanduser(model_path)
            if not os.path.isdir(expanded):
                raise FileNotFoundError(f"model directory not found: {expanded}")
            model_path = expanded
        t0 = time.time()
        load_path = _resolve_load_path(model_path)
        self.model, self.tokenizer = load(load_path)
        # mlx-lm's multimodal wrappers (e.g. gemma4) expose the text stack under
        # ``.language_model``; a plain text model (e.g. gemma4_text, used as the
        # target of an assistant-pair bundle) IS the text stack. Alias it so the
        # forward paths below can always go through ``.language_model``.
        if not hasattr(self.model, "language_model"):
            self.model.language_model = self.model
        self.model_path = load_path
        self.load_seconds = time.time() - t0
        # The most recent forward's trunk cache. An external drafter (Gemma
        # shared-KV) borrows the trunk's live K/V from here at draft time; the
        # native-head path never reads it.
        self.last_trunk_cache = None

    def logits(self, hidden: mx.array) -> mx.array:
        """Trunk head over hidden -> logits ``[..., vocab]``. Mirrors mlx-lm's
        own tie handling: tied models project through the embedding, untied use
        a separate lm_head."""
        lm = self.model.language_model
        if lm.args.tie_word_embeddings:
            return lm.model.embed_tokens.as_linear(hidden)
        return lm.lm_head(hidden)

    def _make_empty_cache(self) -> list:
        """A fresh single-row cache (all layers) to prefill into from scratch."""
        return _make_cache(self.model, [0], None)

    def prefill(self, state: BatchState, ids: list[int]) -> mx.array:
        """Feed one prefill piece into a single-row state."""
        piece = mx.array([ids], dtype=mx.int32)
        h = self.model.language_model.model(piece, cache=state.cache)
        mx.eval(h, *(c.state for c in state.cache))
        state.lengths[0] += len(ids)
        self.last_trunk_cache = state.cache
        return h

    def forward(self, state: BatchState, tokens: mx.array) -> mx.array:
        """Feed ``tokens`` (``[B, k]``) per row, return hidden ``[B, k, H]``.

        Returns ALL k positions' hidden (pre-lm_head; use ``logits()``); does
        not slice or sample. Advances the cache by k.
        """
        k = int(tokens.shape[1])
        h = self.model.language_model.model(tokens, cache=state.cache)
        state.lengths = [n + k for n in state.lengths]
        self.last_trunk_cache = state.cache
        return h

    def snapshot_ssm(self, state: BatchState) -> list:
        """Clone the SSM (ArraysCache) recurrent state of every SSM layer.

        SSM state can't be trimmed because it evolves sequentially. Attention
        layers are skipped (they trim instead). The clone forces evaluation off
        the lazy graph (``v + 0``) so later cache writes don't mutate the
        snapshot.
        """
        snap = []
        for c in state.cache:
            if isinstance(c, ArraysCache):
                snap.append([None if v is None else v + 0 for v in c.cache])
            else:
                snap.append(None)
        return snap

    def restore_ssm(self, state: BatchState, snap: list) -> None:
        """Write a snapshot_ssm() result back into the SSM layers."""
        for c, s in zip(state.cache, snap):
            if s is not None:
                c.cache = [None if v is None else v + 0 for v in s]

    def has_rotating_cache(self, state: BatchState) -> bool:
        """True if any attention layer uses a sliding-window (rotating) KV cache.

        A rotating cache reuses a fixed-size ring buffer and its ``trim`` only
        rewinds the write index — it does NOT restore the buffer contents. So a
        rejected speculative tail cannot be rolled back by trimming; the round
        must instead be fully rewound and the committed prefix replayed."""
        from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
        return any(
            isinstance(c, (BatchRotatingKVCache, RotatingKVCache))
            for c in state.cache
        )

    def trim_attention(self, state: BatchState, n: int) -> None:
        """Trim n positions off every attention (KVCache) layer. SSM layers are
        left untouched — restore them with restore_ssm()."""
        if n <= 0:
            return
        for c in state.cache:
            if not isinstance(c, ArraysCache):
                c.trim(n)

    def filter(self, state: BatchState, keep: list[int]) -> None:
        """Keep only rows ``keep`` (by row index) in the batched cache."""
        for c in state.cache:
            c.filter(keep)
        state.lengths = [state.lengths[i] for i in keep]

    def extract_row(self, state: BatchState, i: int) -> BatchState:
        """Pull row ``i`` out of a batched state into its own single-row state,
        WITHOUT modifying ``state``."""
        cache = [c.extract(i) for c in state.cache]
        return BatchState(cache=cache, lengths=[state.lengths[i]])

    def merge_states(self, states: list[BatchState]) -> BatchState:
        """Merge several single-row states into one batched state.

        Attention: rows may differ in length, so left-pad every row's KV to the
        max length (per-row left_padding). SSM (ArraysCache): recurrent state is
        fixed-size, so just stack along the batch dim.
        """
        nlayers = len(states[0].cache)
        merged = []
        for li in range(nlayers):
            merged.append(self._merge_layer([s.cache[li] for s in states]))
        lengths = [s.lengths[0] for s in states]
        self.last_trunk_cache = merged
        return BatchState(cache=merged, lengths=lengths)

    @staticmethod
    def _row_view(c):
        """Return (valid_keys, valid_values, length) for a SINGLE-row attention
        cache, skipping any existing left_padding. Handles both a plain KVCache
        (from extract; scalar offset, no padding) and a 1-row BatchKVCache (from
        a prior merge; [1] offset with left_padding)."""
        off = c.offset
        if hasattr(off, "shape") and off.shape:      # BatchKVCache (1 row)
            pad = int(c.left_padding[0])
            end = int(c._idx)
            return c.keys[:, :, pad:end, :], c.values[:, :, pad:end, :], end - pad
        n = int(off)                                  # plain KVCache
        return c.keys[..., :n, :], c.values[..., :n, :], n

    def _merge_layer(self, caches):
        if isinstance(caches[0], ArraysCache):
            out = ArraysCache(len(caches[0].cache))
            out.cache = [
                mx.concatenate([c.cache[si] for c in caches], axis=0)
                if caches[0].cache[si] is not None else None
                for si in range(len(caches[0].cache))
            ]
            return out
        # Attention: left-pad each row's valid KV to the max length.
        views = [self._row_view(c) for c in caches]
        lengths = [v[2] for v in views]
        max_len = max(lengths)
        B = len(caches)
        k0 = caches[0].keys
        H, Dk, Dv = k0.shape[1], k0.shape[3], caches[0].values.shape[3]
        keys = mx.zeros((B, H, max_len, Dk), dtype=k0.dtype)
        values = mx.zeros((B, H, max_len, Dv), dtype=k0.dtype)
        padding = [max_len - n for n in lengths]
        for i, (vk, vv, n) in enumerate(views):
            keys[i:i + 1, :, padding[i]:padding[i] + n] = vk
            values[i:i + 1, :, padding[i]:padding[i] + n] = vv
        out = BatchKVCache(padding)
        out.keys, out.values = keys, values
        out.offset = out.offset + max_len
        out._idx = max_len
        return out
