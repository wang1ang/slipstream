"""L1 — engine layer.

The lowest layer. Its only job: run correct batched forward passes and manage
the batched cache. It does NOT sample or make scheduling decisions. It gives
whatever token tensor it is fed a correct forward.

Two capabilities, nothing more:
  * batch: process B sequences together in one forward.
  * next-k: feed k tokens per row at once and return all k positions' logits
    (k=1 and k>1 are the same primitive here).

Correctness facts (verified by experiment):
  * Batched forward of this hybrid SSM+attention model is numerically EXACT vs
    single-sequence when rows are equal length. Verified token-for-token.
  * Unequal-length prefill is handled by right-padding + masking. An mlx-lm bug
    left SSM padding unmasked (see prefill() for the fix); with the fix, the pad
    positions are correctly masked and do NOT corrupt the recurrent state.
  * Residual divergence between a batched row and the same sequence run alone is
    pure floating-point accumulation (batch reduction order differs from B=1).
    This is inherent to batched inference — NOT a bug — and both trajectories are
    valid. It shows up only after many steps as an occasional token flip.
  * The cache is ours to roll back: ``snapshot_ssm`` / ``restore_ssm`` save &
    restore SSM recurrent state (it can't be trimmed), and ``trim_attention``
    trims attention KV.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import _make_cache, _right_pad_prompts
from mlx_lm.models.cache import ArraysCache, BatchKVCache


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
        self.model, self.tokenizer = load(model_path)
        self.model_path = model_path
        self.load_seconds = time.time() - t0

    # --- tokenization ---
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)


    def decode(self, token_ids: list[int]) -> str:
        # skip_special_tokens drops <|im_end|>/<|endoftext|> etc from the text.
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def eos_token_ids(self) -> set[int]:
        return set(self.tokenizer.eos_token_ids)

    def logits(self, hidden: mx.array) -> mx.array:
        """Trunk head over hidden -> logits ``[..., vocab]``. Mirrors mlx-lm's
        own tie handling: tied models project through the embedding, untied use
        a separate lm_head."""
        lm = self.model.language_model
        if lm.args.tie_word_embeddings:
            return lm.model.embed_tokens.as_linear(hidden)
        return lm.lm_head(hidden)

    # --- batched forward primitives (always [B, ...]; B=1 is just a batch of 1) ---
    def prefill(self, prompts: list[list[int]], chunk: int = 0, log=None,
                stop=None) -> tuple[BatchState, mx.array]:
        """Prefill B prompts. Returns (state, hidden ``[B, max_len, H]``).

        Row i's next-token hidden is at position ``lengths[i]-1`` (prompts are
        right-padded to max_len). Returns pre-lm_head hidden; get logits with
        ``logits(hidden)``. The caller decides how to consume logits.

        A single long prompt whose length exceeds ``chunk`` (>0) is fed in
        chunks so the attention scratch stays bounded (a 30k-token prompt's
        full-sequence attention would be tens of GB). The cache accumulates
        across chunks; only the last chunk's hidden is returned (all the caller
        needs is the final position). Batched or short prompts take the
        one-shot path unchanged.
        """
        lengths = [len(p) for p in prompts]
        max_len = max(lengths)

        if chunk and len(prompts) == 1 and max_len > chunk:
            cache = _make_cache(self.model, [0], None)
            state = BatchState(cache=cache, lengths=[0])
            h = self._run_chunked(state, prompts[0], chunk, log=log, stop=stop)
            return state, h

        padding = [max_len - n for n in lengths]
        cache = _make_cache(self.model, [0] * len(prompts), None)
        tokens = _right_pad_prompts(prompts, max_length=max_len)
        for c in cache:
            c.prepare(lengths=lengths, right_padding=padding)
            # mlx-lm bug: _make_cache sets ArraysCache.left_padding = [0,...],
            # so make_mask() takes the left_padding branch (pos >= 0, always True)
            # and never masks right-padding — pad tokens corrupt GatedDeltaNet
            # state. Clearing it forces the lengths branch (pos < lengths), which
            # masks the pad positions. (padding=0 -> masks nothing, still correct.)
            if isinstance(c, ArraysCache):
                c.left_padding = None
        h = self.model.language_model.model(tokens, cache=cache)
        for c in cache:
            c.finalize()
        return BatchState(cache=cache, lengths=list(lengths)), h

    def _make_empty_cache(self) -> list:
        """A fresh single-row cache (all layers) to prefill into from scratch."""
        return _make_cache(self.model, [0], None)

    def prefill_piece(self, state: BatchState, ids: list[int], total: int, *,
                      log=None) -> mx.array:
        """Feed one prefill piece into a single-row state."""
        piece = mx.array([ids], dtype=mx.int32)
        t0 = time.time()
        h = self.model.language_model.model(piece, cache=state.cache)
        mx.eval(h, *(c.state for c in state.cache))
        state.lengths[0] += len(ids)
        if log:
            dt = time.time() - t0
            log(f"prefill {state.lengths[0]}/{total} tok {len(ids) / dt:.0f} tok/s")
        return h

    def _run_chunked(self, state: BatchState, ids: list[int], chunk: int, *,
                     log=None, stop=None):
        """Feed ``ids`` into a single-row ``state`` in chunks (cache accumulates,
        attention stays incremental so the scratch is bounded). Advances
        state.lengths; returns the last chunk's hidden. Returns None if stop()
        fires between chunks."""
        model = self.model.language_model.model
        base = state.lengths[0]
        total = base + len(ids)
        h = None
        for s in range(0, len(ids), chunk):
            piece = mx.array([ids[s:s + chunk]], dtype=mx.int32)
            t0 = time.time()
            h = model(piece, cache=state.cache)
            mx.eval(h, *(c.state for c in state.cache))
            end = base + min(s + chunk, len(ids))       # absolute position
            state.lengths[0] = end
            if log:
                dt = time.time() - t0
                log(f"prefill {end}/{total} tok {(min(s + chunk, len(ids)) - s) / dt:.0f} tok/s")
            if stop and stop():
                return None
        return h

    def forward(self, state: BatchState, tokens: mx.array) -> mx.array:
        """Feed ``tokens`` (``[B, k]``) per row, return hidden ``[B, k, H]``.

        Returns ALL k positions' hidden (pre-lm_head; use ``logits()``); does
        not slice or sample. Advances the cache by k.
        """
        k = int(tokens.shape[1])
        h = self.model.language_model.model(tokens, cache=state.cache)
        state.lengths = [n + k for n in state.lengths]
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
