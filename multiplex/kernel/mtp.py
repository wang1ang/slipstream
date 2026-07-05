"""L2 — MTP speculative decoding (batched, take-min aligned).

The model ships an MTP (multi-token prediction) head in ``mtp.safetensors`` that
mlx-lm does not load. This module loads it and runs BATCHED speculative decode:
draft k tokens per row, verify with the trunk, accept each row's longest correct
prefix, take the min across rows so caches stay aligned. B=1 is just a batch of 1
— there is no separate single-sequence path (no routing in this layer).

The head is one full-attention transformer layer:
    [embed(next_token) ‖ trunk_hidden]  (each RMS-normed)
      -> fc (2H -> H)
      -> DecoderLayer, REUSING the trunk's own layer class (dense or MoE,
         whichever the trunk is — nothing model-specific is hardcoded here)
      -> norm
      -> trunk output head (engine.logits: tied embedding or lm_head)
The head loads generically from config: MoE vs dense follows the trunk, and a
prequantized head is quantized to the config's scheme before load (see Drafter).
Its KV cache is a plain KVCache (pure attention, no SSM), independent of the
trunk cache.
"""

from __future__ import annotations

import os
import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache, BatchKVCache
from mlx_lm.models.base import create_attention_mask
import mlx_lm.models.qwen3_5 as q5


def find_mtp(model_path: str) -> str | None:
    """Return the model's MTP sidecar path, or None for pure AR/headless."""
    model_dir = os.path.expanduser(model_path)
    candidates = []

    runtime = os.path.join(model_dir, "mtplx_runtime.json")
    if os.path.exists(runtime):
        try:
            with open(runtime, encoding="utf-8") as f:
                data = json.load(f)
            for key in ("mtp_sidecar_file", "mtp_file"):
                value = data.get(key)
                if isinstance(value, str):
                    candidates.append(os.path.join(model_dir, value))
        except Exception:
            pass

    candidates.extend([
        os.path.join(model_dir, "mtp.safetensors"),
        os.path.join(model_dir, "mtp", "weights.safetensors"),
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _stack_experts(weights: dict, n_experts: int) -> dict:
    """Turn numbered experts (mlp.experts.N.proj) into stacked switch_mlp.proj.

    mlx-lm's MoE block expects switch_mlp.{gate,up,down}_proj with a leading
    expert axis; the checkpoint stores them per-expert. Stack them.
    """
    out = {}
    per_expert: dict[str, list] = {}
    for k, v in weights.items():
        if ".mlp.experts." in k:
            head, tail = k.split(".mlp.experts.")
            idx, proj = tail.split(".", 1)          # "12", "gate_proj.weight"
            per_expert.setdefault(f"{head}.mlp.switch_mlp.{proj}", {})[int(idx)] = v
        else:
            out[k] = v
    for key, by_idx in per_expert.items():
        out[key] = mx.stack([by_idx[i] for i in range(n_experts)])
    return out


def _quant_config(model_path: str) -> dict:
    """Quantization scheme for the MTP head.

    Some artifacts quantize the draft head differently from the trunk body
    (e.g. Qwen3.6-27B uses an INT4 group-32 MTP sidecar with a group-64 trunk),
    so prefer the explicit MTP block and only fall back to the body default.
    """
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    q = cfg.get("mtplx_mtp_quantization") or cfg.get("quantization") or {}
    return {"group_size": int(q.get("group_size", 64)),
            "bits": int(q.get("bits", 4)),
            "mode": str(q.get("mode", "affine"))}


def _mtp_norms_are_delta_encoded(model_path: str) -> bool:
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    mtp_quant = cfg.get("mtplx_mtp_quantization")
    values = [
        cfg.get("mtplx_mtp_norm_encoding"),
        cfg.get("mtp_norm_encoding"),
    ]
    if isinstance(mtp_quant, dict):
        values.extend([
            mtp_quant.get("norm_encoding"),
            mtp_quant.get("norm_weight_encoding"),
        ])
    return any(
        str(value).strip().lower() in {"delta", "delta_plus_one", "mlx_delta"}
        for value in values
        if value is not None
    )


def _infer_prequant_group_size(weights: dict, bits: int) -> int | None:
    """Infer the sidecar's real quantization group size from packed tensors."""
    if bits <= 0 or 32 % bits:
        return None
    unpack = 32 // bits
    group_sizes = set()
    for key, scales in weights.items():
        if not key.endswith(".scales"):
            continue
        weight = weights.get(key[: -len(".scales")] + ".weight")
        if weight is None or not getattr(weight, "shape", None):
            continue
        groups = int(scales.shape[-1])
        if groups <= 0:
            continue
        input_dims = int(weight.shape[-1]) * unpack
        if input_dims % groups == 0:
            group_sizes.add(input_dims // groups)
    if len(group_sizes) == 1:
        return group_sizes.pop()
    return None


class MTPHead(nn.Module):
    """One MTP layer. Reuses the trunk's embed_tokens and lm_head (not owned)."""

    def __init__(self, targs: q5.TextModelArgs):
        super().__init__()
        eps = targs.rms_norm_eps
        self.pre_fc_norm_embedding = nn.RMSNorm(targs.hidden_size, eps=eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(targs.hidden_size, eps=eps)
        self.fc = nn.Linear(targs.hidden_size * 2, targs.hidden_size, bias=False)
        # A full-attention layer: layer_idx = full_attention_interval - 1.
        self.layers = [q5.DecoderLayer(targs, targs.full_attention_interval - 1)]
        self.norm = nn.RMSNorm(targs.hidden_size, eps=eps)

    def __call__(self, embed: mx.array, hidden: mx.array, cache) -> mx.array:
        """Return post-norm hidden ``[B, L, H]`` for the drafted position(s)."""
        e = self.pre_fc_norm_embedding(embed)
        h = self.pre_fc_norm_hidden(hidden)
        x = self.fc(mx.concatenate([e, h], axis=-1))
        mask = create_attention_mask(x, cache)
        x = self.layers[0](x, mask=mask, cache=cache)
        return self.norm(x)


class Drafter:
    """Loads the MTP head and drafts tokens against a trunk model."""

    def __init__(self, engine, mtp_path: str):
        self.engine = engine
        trunk = engine.model.language_model
        self.embed = trunk.model.embed_tokens
        # Draft logits reuse the trunk head via engine.logits (handles tied /
        # untied); the head shares the trunk's output projection.
        cfg = engine.model.args.text_config
        targs = q5.TextModelArgs.from_dict(cfg)
        self.head = MTPHead(targs)
        qc = _quant_config(engine.model_path)

        raw = mx.load(mtp_path)
        weights = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
        # MoE trunks store per-expert weights; stack them. Dense heads have no
        # experts (num_experts falsy) — nothing to stack.
        if getattr(targs, "num_experts", None):
            weights = _stack_experts(weights, targs.num_experts)

        # A prequantized head ships .scales/.biases; those exact modules must be
        # quantized to the SAME scheme BEFORE loading (mirrors mlx-lm's quantized
        # load). Quantization is MIXED — only the modules that actually carry a
        # .scales weight are quantized (e.g. fc / norms stay full precision), so
        # drive it by a predicate over the weight keys, not a blanket quantize.
        prequant = any(k.endswith(".scales") for k in weights)
        if prequant:
            inferred_group_size = _infer_prequant_group_size(weights, qc["bits"])
            if inferred_group_size is not None:
                qc["group_size"] = inferred_group_size

            scaled = {k[: -len(".scales")] for k in weights if k.endswith(".scales")}

            def is_quantized(path, _module):
                return path in scaled and {"group_size": qc["group_size"],
                                           "bits": qc["bits"], "mode": qc["mode"]}

            nn.quantize(self.head, class_predicate=is_quantized)

        # strict=False: heads vary by architecture (dense vs MoE, quantized or
        # not); load what matches and let the module keep its untouched norms.
        self.head.load_weights(list(weights.items()), strict=False)
        if _mtp_norms_are_delta_encoded(engine.model_path):
            for _, m in self.head.named_modules():
                if isinstance(m, nn.RMSNorm):
                    m.weight = m.weight + 1.0
        # A non-prequantized head is quantized to the model's own scheme (bits
        # follow the model, not a user knob); prequantized heads are done above.
        if not prequant:
            nn.quantize(self.head, group_size=qc["group_size"], bits=qc["bits"],
                        mode=qc["mode"])
        self.head.eval()

    def make_cache(self) -> list:
        return [KVCache()]

    @staticmethod
    def trim_cache_to(cache: list, size: int) -> None:
        target = max(0, int(size))
        cur = int(cache[0].size())
        if cur > target:
            cache[0].trim(cur - target)

    def merge_caches(self, caches: list[list]) -> list:
        if not caches:
            return self.make_cache()
        if len(caches) == 1:
            return caches[0]
        return [BatchKVCache.merge([cache[0] for cache in caches])]

    @staticmethod
    def extract_cache_row(cache: list, row: int) -> list:
        c = cache[0]
        if hasattr(c, "extract"):
            return [c.extract(row)]
        single = KVCache()
        if c.keys is not None:
            single.keys = mx.contiguous(c.keys[row:row + 1, :, :c.offset, :])
            single.values = mx.contiguous(c.values[row:row + 1, :, :c.offset, :])
            single.offset = int(c.offset)
        return [single]

    @staticmethod
    def filter_cache(cache: list, keep: list[int]) -> None:
        """Keep only rows ``keep`` in the draft KVCache. Plain KVCache has no
        filter(), so slice its [B, H, L, D] keys/values by row."""
        c = cache[0]
        if hasattr(c, "filter"):
            c.filter(keep)
        elif c.keys is not None:
            c.keys = c.keys[keep]
            c.values = c.values[keep]

    @staticmethod
    def _row_view(c, row: int | None = None):
        if isinstance(c, BatchKVCache):
            if row is None:
                if int(c.left_padding.shape[0]) != 1:
                    raise ValueError("row is required for batched MTP cache blocks")
                row = 0
            pad = int(c.left_padding[row])
            end = int(c._idx)
            return (
                c.keys[row:row + 1, :, pad:end, :],
                c.values[row:row + 1, :, pad:end, :],
                end - pad,
            )
        n = int(c.offset)
        if row is None:
            return c.keys[..., :n, :], c.values[..., :n, :], n
        return c.keys[row:row + 1, :, :n, :], c.values[row:row + 1, :, :n, :], n

    def clone_cache_block(self, cache: list, start: int, pos: int, *,
                          row: int | None = None):
        """Clone MTP KV deltas needed for trunk prefix ``[start:pos]``.

        MTP history for a trunk prefix of length ``pos`` contains transitions
        for token positions ``1..pos-1``. A trunk cache block ``[start:pos]``
        therefore contributes MTP slots ``max(1, start)..pos-1``.
        """
        start = int(start)
        pos = int(pos)
        mtp_start = max(1, start) - 1
        mtp_end = max(0, pos - 1)
        if mtp_end <= mtp_start:
            return [None]
        keys, values, length = self._row_view(cache[0], row=row)
        if mtp_end > length:
            raise ValueError(
                f"invalid MTP block slice [{mtp_start}:{mtp_end}] "
                f"for length={length}"
            )
        k = keys[..., mtp_start:mtp_end, :] + 0
        v = values[..., mtp_start:mtp_end, :] + 0
        mx.eval(k, v)
        return [[k, v]]

    def restore_cache_blocks(self, blocks: list) -> list:
        cache = self.make_cache()
        parts = [block[0] for block in blocks if block[0] is not None]
        if not parts:
            return cache
        keys = mx.concatenate([p[0] for p in parts], axis=2) \
            if len(parts) > 1 else parts[0][0] + 0
        values = mx.concatenate([p[1] for p in parts], axis=2) \
            if len(parts) > 1 else parts[0][1] + 0
        mx.eval(keys, values)
        cache[0].state = [keys, values]
        return cache

    def append_history(self, cache: list, hidden: mx.array, tokens) -> None:
        """Append committed MTP-history transitions without sampling drafts."""
        if hidden is None or int(hidden.shape[1]) == 0:
            return
        tok = tokens if hasattr(tokens, "shape") else mx.array(tokens, dtype=mx.int32)
        if len(tok.shape) == 1:
            tok = tok[None, :]
        if int(tok.shape[1]) == 0:
            return
        post = self.head(self.embed(tok), hidden, cache[0])
        mx.eval(post, *cache[0].state)

    def draft(self, hidden: mx.array, tokens: mx.array, k: int, cache: list) -> mx.array:
        """Draft k tokens per row (greedy). Returns draft tokens ``[B, k]``.

        hidden: [B, 1, H] trunk hidden of the last committed position.
        tokens: [B]       the next (committed) token id per row.
        Chains the head: each step feeds the head's own previous hidden and the
        token it just drafted.
        """
        if k == 0:                                  # no draft -> pure AR
            return mx.zeros((tokens.shape[0], 0), dtype=tokens.dtype)
        h = hidden
        tok = tokens[:, None]                       # [B, 1]
        drafts = []
        for _ in range(k):
            post = self.head(self.embed(tok), h, cache[0])   # [B, 1, H]
            tok = mx.argmax(self.engine.logits(post)[:, -1, :], axis=-1)[:, None]  # [B, 1]
            drafts.append(tok)
            h = post
        return mx.concatenate(drafts, axis=1)
