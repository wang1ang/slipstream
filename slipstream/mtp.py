"""L2 — MTP speculative decoding (batched, take-min aligned).

The model ships an MTP (multi-token prediction) head in ``mtp.safetensors`` that
mlx-lm does not load. This module loads it and runs BATCHED speculative decode:
draft k tokens per row, verify with the trunk, accept each row's longest correct
prefix, take the min across rows so caches stay aligned. B=1 is just a batch of 1
— there is no separate single-sequence path (no routing in this layer).

The head (qwen3_5_moe) is one full-attention transformer layer:
    [embed(next_token) ‖ trunk_hidden]  (each RMS-normed)
      -> fc (4096->2048)
      -> DecoderLayer (self-attn + MoE), reusing the trunk's layer class
      -> norm
      -> trunk lm_head   (weights reused from the trunk)
Its KV cache is a plain KVCache (pure attention, no SSM), independent of the
trunk cache.
"""

from __future__ import annotations

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
        self.lm_head = trunk.lm_head
        targs = q5.TextModelArgs.from_dict(engine.model.args.text_config)
        self.head = MTPHead(targs)

        raw = mx.load(mtp_path)
        weights = {k[len("mtp."):]: v for k, v in raw.items() if k.startswith("mtp.")}
        weights = _stack_experts(weights, targs.num_experts)
        self.head.load_weights(list(weights.items()), strict=True)
        # This model's RMSNorm uses the (1 + weight) convention: the checkpoint
        # stores weight-1, and the forward computes (1 + w) * x. mlx-lm's nn.RMSNorm
        # is plain w * x, so add 1 to every norm weight in the head. (The trunk is
        # loaded by mlx-lm which already handles this; the head is loaded here.)
        for _, m in self.head.named_modules():
            if isinstance(m, nn.RMSNorm):
                m.weight = m.weight + 1.0
        # Optionally quantize the head's Linear/MoE weights (faster draft, slight
        # accuracy loss). RMSNorm is left untouched by nn.quantize.
        if bits is not None:
            nn.quantize(self.head, bits=bits)
        self.head.eval()

    def make_cache(self) -> list:
        return [KVCache()]

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
            tok = mx.argmax(self.lm_head(post)[:, -1, :], axis=-1)[:, None]  # [B, 1]
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

    Take-min keeps all rows in lockstep so the batched cache never goes ragged.
    Streams: yields one list[list[int]] per step — the new tokens for each row
    that step (a finished row yields []). Greedy; output per row == plain AR.
    """
    B = len(prompts)
    # Rows may be unequal length: prefill right-pads + masks (engine handles it),
    # and each step advances every row by the SAME amount (verify +k+1, repair
    # nets +m+1), so only the initial lengths differ — the cache stays aligned.
    state, hidden = engine.prefill(prompts)
    lens = [len(p) for p in prompts]
    h = mx.concatenate(
        [hidden[i : i + 1, lens[i] - 1 : lens[i], :] for i in range(B)], axis=0
    )  # [B, 1, H] each row's last real hidden
    primary = mx.argmax(engine.logits(h)[:, -1, :], axis=-1)  # [B]

    eos = engine.eos_token_ids
    last = [int(primary[i]) for i in range(B)]
    done = [t in eos for t in last]
    yield [[t] for t in last]                    # stream the first token per row
    dcache = drafter.make_cache()
    n = 1
    while n < max_tokens and not all(done):
        # 1. draft k for every row
        drafts = drafter.draft(h, primary, k, dcache)             # [B, k]
        draft_ids = [[int(x) for x in drafts[i]] for i in range(B)]

        # 2. verify: batched forward [primary, d1..dk] per row
        snap = engine.snapshot_ssm(state)
        lengths_before = list(state.lengths)   # per-row, may differ
        verify_in = mx.array(
            [[int(primary[i])] + draft_ids[i] for i in range(B)]
        )  # [B, k+1]
        vhidden = engine.forward(state, verify_in)
        vlogits = engine.logits(vhidden)                          # [B, k+1, V]
        trunk_pred = mx.argmax(vlogits, axis=-1)                  # [B, k+1]

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

        # 4. this step's new tokens per row = m accepted drafts + correction; a
        # finished row contributes nothing further (its stream already ended).
        step = []
        for i in range(B):
            if done[i]:
                step.append([])
                continue
            toks = draft_ids[i][:m] + [int(trunk_pred[i, m])]
            cut = toks
            for j, t in enumerate(toks):
                if t in eos:
                    cut = toks[: j + 1]
                    done[i] = True
                    break
            step.append(cut)
        yield step
        n += m + 1

        # 5. next primary = each row's correction; repair to committed length
        primary = trunk_pred[:, m]                                # [B]
        if m == k:
            h = vhidden[:, -1:, :]
        else:
            # verify advanced every row by k+1; keep only m+1 -> trim k-m (same
            # for all rows, so the batched cache stays aligned regardless of the
            # rows' differing lengths).
            engine.restore_ssm(state, snap)
            engine.trim_attention(state, k - m)
            state.lengths = list(lengths_before)
            commit_in = mx.array(
                [[int(verify_in[i, 0])] + draft_ids[i][:m] for i in range(B)]
            )  # [B, m+1]
            h = engine.forward(state, commit_in)[:, -1:, :]