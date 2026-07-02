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
from mlx_lm.models.cache import KVCache
from mlx_lm.models.base import create_attention_mask
import mlx_lm.models.qwen3_5 as q5


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
    """Default quantization scheme from the model's config.json. The block sits
    at the TOP level (not text_config), and mlx-lm drops it after load, so read
    the file. Mixed per-layer specs share the block; top-level = the default."""
    with open(os.path.join(model_path, "config.json")) as f:
        q = (json.load(f).get("quantization") or {})
    return {"group_size": int(q.get("group_size", 64)),
            "bits": int(q.get("bits", 4)),
            "mode": str(q.get("mode", "affine"))}


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

    def __init__(self, engine, mtp_path: str, bits: int | None = None):
        self.engine = engine
        trunk = engine.model.language_model
        self.embed = trunk.model.embed_tokens
        # Draft logits reuse the trunk head via engine.logits (handles tied /
        # untied); the head shares the trunk's output projection.
        cfg = engine.model.args.text_config
        targs = q5.TextModelArgs.from_dict(cfg)
        self.head = MTPHead(targs)

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
            qc = _quant_config(engine.model_path)
            scaled = {k[: -len(".scales")] for k in weights if k.endswith(".scales")}

            def is_quantized(path, _module):
                return path in scaled and {"group_size": qc["group_size"],
                                           "bits": qc["bits"], "mode": qc["mode"]}

            nn.quantize(self.head, class_predicate=is_quantized)

        # strict=False: heads vary by architecture (dense vs MoE, quantized or
        # not); load what matches and let the module keep its untouched norms.
        self.head.load_weights(list(weights.items()), strict=False)
        # This model's RMSNorm uses the (1 + weight) convention: the checkpoint
        # stores weight-1, and the forward computes (1 + w) * x. mlx-lm's nn.RMSNorm
        # is plain w * x, so add 1 to every norm weight in the head. (The trunk is
        # loaded by mlx-lm which already handles this; the head is loaded here.)
        for _, m in self.head.named_modules():
            if isinstance(m, nn.RMSNorm):
                m.weight = m.weight + 1.0
        # A non-prequantized head can be quantized here (faster draft, slight
        # accuracy loss); prequantized heads are already quantized above.
        if bits is not None and not prequant:
            nn.quantize(self.head, bits=bits)
        self.head.eval()

    def make_cache(self) -> list:
        return [KVCache()]

    @staticmethod
    def filter_cache(cache: list, keep: list[int]) -> None:
        """Keep only rows ``keep`` in the draft KVCache. Plain KVCache has no
        filter(), so slice its [B, H, L, D] keys/values by row."""
        for c in cache:
            if c.keys is not None:
                c.keys = c.keys[keep]
                c.values = c.values[keep]

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


def speculative_generate_batch(engine, drafter, prompts, *, max_tokens, k=3):
    """Batched greedy speculative decode with TAKE-MIN alignment.

    B prompts decode together. Each step:
      1. draft k tokens for every row (one batched MTP pass per depth)
      2. verify: one batched trunk forward of [primary, d1..dk] per row
      3. each row accepts its longest correct prefix; commit = MIN across rows
         (so every row advances the same amount -> caches stay aligned)
      4. every row emits its own (min accepted drafts + its correction token)
      5. repair uniformly by the same amount (SSM restore + attention trim + a
         batched re-run of the committed prefix)

    Take-min keeps all live rows in lockstep so the batched cache never goes
    ragged. "出": when a row hits EOS it is filtered out of the batch (its cache
    row dropped) so the batch shrinks and the rest run faster.

    Streams: yields ``(row_ids, step)`` per step, where ``row_ids`` are the
    ORIGINAL prompt indices still live (order matches the current batch rows) and
    ``step[i]`` is row ``row_ids[i]``'s new tokens this step. A row that finished
    this step appears one last time (with its final tokens) then is gone from
    later ``row_ids``. Greedy; each row's output == plain AR.
    """
    eos = engine.eos_token_ids
    # Rows may be unequal length: prefill right-pads + masks (engine handles it),
    # and each step advances every live row by the SAME amount, so the cache
    # stays aligned; finished rows are filtered out (not padded along).
    state, hidden = engine.prefill(prompts)
    lens = [len(p) for p in prompts]
    B = len(prompts)
    h = mx.concatenate(
        [hidden[i : i + 1, lens[i] - 1 : lens[i], :] for i in range(B)], axis=0
    )  # [B, 1, H] each row's last real hidden
    primary = mx.argmax(engine.logits(h)[:, -1, :], axis=-1)  # [B]
    row_ids = list(range(B))                     # live rows -> original indices

    first = [int(primary[i]) for i in range(B)]
    yield list(row_ids), [[t] for t in first]
    n = 1
    dcache = drafter.make_cache()

    # drop rows whose first token is already EOS
    keep0 = [i for i, t in enumerate(first) if t not in eos]
    if len(keep0) < B:
        engine.filter(state, keep0)
        drafter.filter_cache(dcache, keep0)
        primary = primary[mx.array(keep0)]
        h = h[mx.array(keep0)]
        row_ids = [row_ids[i] for i in keep0]

    while n < max_tokens and row_ids:
        B = len(row_ids)
        # 1. draft k for every live row
        drafts = drafter.draft(h, primary, k, dcache)             # [B, k]
        draft_ids = [[int(x) for x in drafts[i]] for i in range(B)]

        # 2. verify: batched forward [primary, d1..dk] per row
        snap = engine.snapshot_ssm(state)
        lengths_before = list(state.lengths)
        verify_in = mx.array([[int(primary[i])] + draft_ids[i] for i in range(B)])
        vhidden = engine.forward(state, verify_in)
        trunk_pred = mx.argmax(engine.logits(vhidden), axis=-1)   # [B, k+1]

        # 3. each row's accepted prefix, then take the min
        accs = []
        for i in range(B):
            a = 0
            for j in range(k):
                if draft_ids[i][j] == int(trunk_pred[i, j]):
                    a += 1
                else:
                    break
            accs.append(a)
        m = min(accs)

        # 4. emit m accepted drafts + correction per row; mark rows that hit EOS
        step, finished = [], []
        for i in range(B):
            toks = draft_ids[i][:m] + [int(trunk_pred[i, m])]
            for j, t in enumerate(toks):
                if t in eos:
                    toks = toks[: j + 1]
                    finished.append(i)
                    break
            step.append(toks)
        yield list(row_ids), step
        n += m + 1

        # 5. next primary = each row's correction; repair to committed length
        primary = trunk_pred[:, m]                                # [B]
        if m == k:
            h = vhidden[:, -1:, :]
        else:
            # verify advanced every row by k+1; keep only m+1 -> trim k-m (same
            # for all rows, so the batched cache stays aligned).
            engine.restore_ssm(state, snap)
            engine.trim_attention(state, k - m)
            state.lengths = list(lengths_before)
            commit_in = mx.array(
                [[int(verify_in[i, 0])] + draft_ids[i][:m] for i in range(B)]
            )  # [B, m+1]
            h = engine.forward(state, commit_in)[:, -1:, :]

        # 6. 出: drop finished rows from the batch (cache + draft cache + state)
        if finished:
            keep = [i for i in range(B) if i not in finished]
            if not keep:
                return
            engine.filter(state, keep)
            drafter.filter_cache(dcache, keep)
            primary = primary[mx.array(keep)]
            h = h[mx.array(keep)]
            row_ids = [row_ids[i] for i in keep]