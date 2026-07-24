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


def build_qwen_head(engine, mtp_path: str):
    """Build the Qwen native-MTP head from a ``mtp.safetensors`` sidecar.

    Returns an object exposing ``__call__(embed, hidden, cache) -> post_hidden``
    and ``embed`` (the trunk's embed_tokens), which is all ``Drafter`` needs. The
    head reuses the trunk's own decoder-layer class and shares its output
    projection via ``engine.logits``.
    """
    cfg = engine.model.args.text_config
    targs = q5.TextModelArgs.from_dict(cfg)
    head = MTPHead(targs)
    head.embed = engine.model.language_model.model.embed_tokens
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

        nn.quantize(head, class_predicate=is_quantized)

    # strict=False: heads vary by architecture (dense vs MoE, quantized or
    # not); load what matches and let the module keep its untouched norms.
    head.load_weights(list(weights.items()), strict=False)
    if _mtp_norms_are_delta_encoded(engine.model_path):
        for _, m in head.named_modules():
            if isinstance(m, nn.RMSNorm):
                m.weight = m.weight + 1.0
    # A non-prequantized head is quantized to the model's own scheme (bits
    # follow the model, not a user knob); prequantized heads are done above.
    if not prequant:
        nn.quantize(head, group_size=qc["group_size"], bits=qc["bits"],
                    mode=qc["mode"])
    head.eval()
    return head


def _pair_layout(model_path: str) -> dict | None:
    """Return an assistant-pair bundle's ``layout`` dict, or None if the model
    at ``model_path`` is not a pair bundle. A pair bundle has ``mtplx_pair.json``
    at its root pointing at ``target/`` and ``assistant/`` subdirectories."""
    pair = os.path.join(os.path.expanduser(model_path), "mtplx_pair.json")
    if not os.path.isfile(pair):
        return None
    try:
        with open(pair, encoding="utf-8") as f:
            layout = (json.load(f).get("layout") or {})
    except (OSError, json.JSONDecodeError):
        return None
    return layout if isinstance(layout, dict) else {}


class GemmaHead:
    """Gemma 4 external-drafter MTP head (MTPLX assistant-backed).

    The Gemma drafter is a 4-layer transformer whose attention layers own no
    K/V — they must borrow the target's (shared-KV). Unlike Qwen's native head,
    it keeps NO KV cache of its own: every draft step re-reads the target's live
    K/V straight from ``engine``'s trunk cache. So the ``cache`` argument that
    ``Drafter`` threads through (a plain ``KVCache``) is used only as a position
    counter here — the head writes a 1-wide dummy per token to keep ``Drafter``'s
    trim/append bookkeeping consistent, but never attends to it.

    The assistant forward (``mtplx.backends.gemma4_assistant``) is reused as-is;
    this class only adapts it to ``Drafter``'s ``head(embed, hidden, cache)``
    step interface and wires the shared-KV extraction.
    """

    def __init__(self, engine, assistant, *, sliding_src: int, full_src: int):
        self.engine = engine
        self.assistant = assistant
        # Source layers whose cached K/V the drafter's sliding / full-attention
        # layers borrow: the LAST trunk layer of each attention type. This
        # mirrors MTPLX's forward_with_state, where the shared_kv_states dict
        # ends up holding the last-written K/V per layer_type.
        self._sliding_src = int(sliding_src)
        self._full_src = int(full_src)
        text_model = engine.model.language_model.model
        self.embed = text_model.embed_tokens
        self._embed_scale = float(getattr(text_model, "embed_scale", 1.0))
        self._last_logits = None
        # Position within the current draft block. Draft step i predicts the
        # token at sequence position off+i, so its query position is off-1+i:
        # it must ADVANCE across a multi-token draft block, not stay pinned at
        # off-1 (which is only correct for the first drafted token). Drafter
        # resets this at the start of each block via begin_draft().
        self._draft_step = 0

    def begin_draft(self) -> None:
        """Reset the per-block draft position counter (called once per block)."""
        self._draft_step = 0

    @staticmethod
    def _valid_kv(c):
        """Return a source layer's committed K/V ``[1, n_kv, L, D]``, skipping
        any left-padding (BatchKVCache/BatchRotatingKVCache carry a padded
        offset; a plain KVCache has a scalar offset and none)."""
        off = c.offset
        if hasattr(off, "shape") and off.shape:        # batched (1 row)
            pad = int(c.left_padding[0])
            end = int(c._idx)
            return c.keys[:, :, pad:end, :], c.values[:, :, pad:end, :]
        n = int(off)
        return c.keys[..., :n, :], c.values[..., :n, :]

    def _shared_kv(self, cache):
        ks, vs = self._valid_kv(cache[self._sliding_src])
        kf, vf = self._valid_kv(cache[self._full_src])
        return {"sliding_attention": (ks, vs), "full_attention": (kf, vf)}

    def _trunk_cache(self):
        """The live trunk KV cache the drafter borrows from. The engine stashes
        the cache of its most recent forward on ``last_trunk_cache``; at draft
        time that is the current committed decode cache."""
        cache = getattr(self.engine, "last_trunk_cache", None)
        if cache is None:
            raise RuntimeError("Gemma drafter: no trunk cache to borrow shared-KV from")
        return cache

    def _trunk_offset(self, cache) -> int:
        off = cache[self._full_src].offset
        return int(off.item()) if hasattr(off, "shape") and off.shape else int(off)

    def __call__(self, embed: mx.array, hidden: mx.array, cache) -> mx.array:
        """One (or L) draft step(s). ``embed`` is ``self.embed(tok)`` for the
        token(s) being predicted from, ``hidden`` the trunk hidden of the same
        position(s). Returns the drafter's post-hidden, shaped like ``hidden``.

        Attention borrows the trunk's live shared-KV at its current offset; the
        query position is pinned to ``offset-1`` (the position of the primary
        token that produced ``hidden``), matching the MLX-VLM/MTPLX drafter.
        """
        trunk = self._trunk_cache()
        off = self._trunk_offset(trunk)
        # Query position advances within the draft block: step i (0-based) drafts
        # the token at sequence position off+i, whose query position is off-1+i.
        # Pinning at off-1 is only correct for step 0 (k=1); k>=2 needs the shift
        # or the 2nd+ drafted token gets the wrong RoPE position and diverges.
        pos = max(off - 1 + self._draft_step, 0)
        self.assistant.set_shared_kv(
            self._shared_kv(trunk), off,
            position=pos, kv_valid_len=off,
        )
        self._draft_step += 1
        # embed arrives already scaled? No — Drafter calls self.embed(tok) which
        # is the raw nn.Embedding; apply the trunk's embed_scale here (mirrors
        # the assistant's own draft_step, which scales the token embedding).
        inputs_embeds = mx.concatenate([embed * self._embed_scale, hidden], axis=-1)
        post, logits = self.assistant(inputs_embeds)
        # Drafts come from the assistant's OWN vocab head, not the trunk's — the
        # trunk output projection expects trunk-space hidden, whereas ``post`` is
        # the assistant's backbone-space chaining state. Stash the step's logits
        # for Drafter to read via ``self.logits`` immediately after this call.
        self._last_logits = logits
        # Keep Drafter's KVCache position counter in step with the trunk without
        # ever attending to it: write one dummy slot per input token.
        L = int(inputs_embeds.shape[1])
        dummy = mx.zeros((inputs_embeds.shape[0], 1, L, 1), dtype=post.dtype)
        cache.update_and_fetch(dummy, dummy)
        return post

    def logits(self, post: mx.array) -> mx.array:
        """The assistant's logits for the step ``__call__`` just computed. The
        ``post`` argument (chaining hidden) is ignored — Drafter calls this right
        after ``__call__`` in the same step, so the stashed logits are current."""
        return self._last_logits


def _import_gemma_assistant():
    """Return MTPLX's ``load_gemma4_assistant_model``.

    The Gemma assistant model class + its shared-KV forward live in MTPLX, which
    is a sibling checkout rather than an installed dependency. Import it directly
    if it is already on the path; otherwise add the checkout root (from
    ``MTPLX_ROOT`` or a sibling ``../MTPLX``) and retry."""
    import importlib
    import sys

    def _load():
        mod = importlib.import_module("mtplx.backends.gemma4_assistant")
        return mod.load_gemma4_assistant_model

    try:
        return _load()
    except ModuleNotFoundError:
        pass
    candidates = []
    env_root = os.environ.get("MTPLX_ROOT")
    if env_root:
        candidates.append(os.path.expanduser(env_root))
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates.append(os.path.join(os.path.dirname(here), "MTPLX"))
    for root in candidates:
        if os.path.isdir(os.path.join(root, "mtplx")) and root not in sys.path:
            sys.path.insert(0, root)
    return _load()


def build_gemma_head(engine, assistant_dir: str):
    """Build a Gemma 4 external-drafter head from a pair bundle's assistant dir.

    Loads the MTPLX assistant (which honours its own 4-bit quant config), binds
    it to the trunk (for shared-KV geometry + input embedding), and returns a
    head exposing ``__call__(embed, hidden, cache) -> post_hidden`` and ``embed``
    — the same contract as the Qwen head. The assistant's 4-layer transformer
    forward is reused verbatim; nothing here re-implements it.
    """
    load_gemma4_assistant_model = _import_gemma_assistant()

    assistant = load_gemma4_assistant_model(assistant_dir)
    # bind() wires the trunk's embed_tokens/embed_scale and reads layer_types;
    # it accepts the mlx-lm wrapper (uses .language_model.model / .text_model).
    assistant.bind(engine.model)

    text_model = engine.model.language_model.model
    layer_types = [l.layer_type for l in text_model.layers]
    full_src = max(i for i, t in enumerate(layer_types) if t == "full_attention")
    sliding_src = max(i for i, t in enumerate(layer_types) if t == "sliding_attention")
    return GemmaHead(engine, assistant, sliding_src=sliding_src, full_src=full_src)


class Drafter:
    """Drafts tokens against a trunk model using an injected MTP ``head``.

    The head is model-specific (Qwen native head, Gemma assistant, …) and only
    needs to expose ``__call__(embed, hidden, cache) -> post_hidden`` plus an
    ``embed`` attribute. Everything else here — draft-cache management, the draft
    loop, committed-history append, verify hand-off — is model-agnostic.

    Draft tokens are read from ``self.draft_logits(post)``. A native head (Qwen)
    shares the trunk's output projection, so this defaults to ``engine.logits``.
    An external drafter with its own vocab head (Gemma assistant) instead exposes
    a ``logits`` method returning ITS logits for the step it just computed; the
    head's ``post_hidden`` is then only the chaining state, not a trunk hidden.
    """

    def __init__(self, engine, head):
        self.engine = engine
        self.head = head
        self.embed = head.embed
        # External drafters project drafts through their own head; native heads
        # reuse the trunk's (engine.logits). Predicate: head advertises .logits.
        self.draft_logits = getattr(head, "logits", None) or engine.logits

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
        # A stateless shared-KV head (Gemma) keeps no draft KV of its own — it
        # re-reads the trunk's live K/V each step — so there is no MTP history to
        # append, and running the head here would only perturb its block-position
        # counter. The native head (Qwen) has no begin_draft and needs the append.
        if hasattr(self.head, "begin_draft"):
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
        # Shared-KV heads track a query position that advances across this block;
        # reset it at the block start. Native heads (Qwen) don't define this.
        if hasattr(self.head, "begin_draft"):
            self.head.begin_draft()
        h = hidden
        tok = tokens[:, None]                       # [B, 1]
        drafts = []
        for _ in range(k):
            post = self.head(self.embed(tok), h, cache[0])   # [B, 1, H]
            tok = mx.argmax(self.draft_logits(post)[:, -1, :], axis=-1)[:, None]  # [B, 1]
            drafts.append(tok)
            h = post
        return mx.concatenate(drafts, axis=1)


def find_drafter(engine, mtp_path: str | None = None):
    """Build the right MTP drafter for a loaded model, or None for pure AR.

    Unified entry point over the drafter zoo: it picks a head builder by what the
    model ships, wraps it in the model-agnostic ``Drafter``, and returns that.
    Add a new MTP family by adding a head builder + a branch here — the draft
    loop, cache management, and scheduler contract stay shared.
      * ``mtp.safetensors`` sidecar  -> Qwen native MTP head (build_qwen_head)
      * assistant-pair bundle        -> Gemma external drafter (build_gemma_head)
    """
    # ``mtp_path`` is normally discovered beside the model, but the server's
    # --mtp option may point at a sidecar stored elsewhere.
    mtp_path = mtp_path or find_mtp(engine.model_path)
    if mtp_path is not None:
        return Drafter(engine, build_qwen_head(engine, mtp_path))

    # A pair bundle points at target/ (already the load path) + assistant/. The
    # engine loaded the target; the drafter is the sibling assistant dir. The
    # bundle root is the parent of the resolved target load path.
    bundle_root = os.path.dirname(os.path.normpath(engine.model_path))
    layout = _pair_layout(bundle_root)
    if layout is not None:
        assistant_dir = os.path.join(
            bundle_root, str(layout.get("assistant") or "assistant"))
        if os.path.isfile(os.path.join(assistant_dir, "config.json")):
            return Drafter(engine, build_gemma_head(engine, assistant_dir))
    return None
