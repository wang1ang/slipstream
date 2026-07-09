# Vision support in multiplex — status & path

## TL;DR

multiplex today is a **text-only** engine: it can *load* a Gemma 4 assistant-pair
bundle's text target and run text + MTP speculative decode, but it **cannot process
image input**. Adding vision is a bounded change (the underlying model already accepts
`input_embeddings`); the missing pieces are an image encoder, an embeds ingestion path,
and text↔image embed splicing. Vision and MTP can coexist — vision lives in prefill,
MTP accelerates text decode.

## Why it's not supported today

The whole request path is a **token-id stream**:

- `Hub`/`server` render the chat template and tokenize to ids (`hub.py`).
- `Engine.prefill(ids)` / `Engine.forward(tokens)` build `mx.array([ids])` and call
  `self.model.language_model.model(piece, cache=...)` (`kernel/engine.py:112-131`).
- There is no `pixel_values` / image path anywhere; `Drafter` (MTP) is also text-only.

So an image in the request has nowhere to enter.

## What's already in our favour

The trunk model's forward **already accepts `input_embeddings`**:

```
gemma4_text Model.__call__(self, inputs, cache=None,
                           input_embeddings=None, per_layer_inputs=None)
```

That means we do **not** need to modify the transformer to inject image features — we can
compute a full `[text_embeds ‖ image_embeds]` sequence outside and feed it via
`input_embeddings`. This is what shrinks the change.

## Gemma 4 vision, concretely

Gemma 4 12B (`gemma4_unified`) has **no separate SigLIP vision tower**. Its vision side is
a small **`vision_embedder`** (patchify → `patch_dense` 6912→3840 → norms + positional
embedding) plus `embed_vision`, which project image patches straight into the 3840-dim
**text** embedding space. Those weights are ~100 MB and are kept in the assistant-pair
bundle under `vision/` (`vision.safetensors` + `vision_config.json`) but are **not loaded**
by the text+MTP path. (Note: mlx-vlm's `gemma4_unified` module has a matching
`VisionEmbedder`; MTPLX's own vision path is Qwen-only and does **not** recognize Gemma's
`vision_embedder.*`.)

## Change path (4 pieces, bottom-up)

1. **Image encoder** (main new code)
   Turn pixels into image embeds in 3840-space. Two options:
   - **Reuse mlx-vlm**: load the `gemma4_unified` model and call its `encode_image(pixel_values)`
     (recognizes the `vision_embedder.*` weights out of the box). Adds an mlx-vlm dependency.
   - **Self-contained**: load `vision/vision.safetensors` and implement the `vision_embedder`
     + `embed_vision` forward here (keeps multiplex dependency-light, more code).

2. **Engine embeds ingress** (small)
   Add an `input_embeddings` path to `prefill`/`forward`: instead of `mx.array([ids])`, pass
   `self.model.language_model.model(None, cache=..., input_embeddings=embeds)`. The trunk
   already supports the kwarg — this is mostly plumbing it up through `Engine`.

3. **Text↔image splice** (new)
   Build the prefill embeds: `embed_tokens(text_ids)` with the image-placeholder positions
   replaced by the image embeds from (1). Standard VLM "scatter image embeds into text
   embeds" at the `<image>` token positions.

4. **Server / input layer** (new; optional for CLI)
   Accept OpenAI `image_url` message parts, preprocess pixels (resize/patch per Gemma's
   `processor_config.json`), and inject the `<image>` placeholder into the chat template.
   For a first cut this can be skipped — feed an image path directly in a test harness.

### Suggested staging
- First: (1)+(2)+(3) — encode an image, splice, prefill through `input_embeddings`, verify
  image Q&A from a CLI harness (no server).
- Then: (4) — wire the OpenAI image API.

Smallest route for (1): call mlx-vlm's `gemma4_unified.encode_image` rather than reimplement
the vision forward; multiplex then only owns (2)+(3)+(4).

## Vision × MTP

They compose. Image features enter during **prefill** (encoded once, spliced into the
prompt embeds). Decode afterwards is ordinary text decode and can use the existing Gemma
external-drafter MTP (`GemmaHead`) unchanged — MTP accelerates text-token generation and is
agnostic to how the prompt prefix was built. No MTP change is needed for vision.

## Current capability matrix (Gemma 4 12B assistant-pair)

| Path | Load | Text decode | MTP speedup | Image input |
|---|---|---|---|---|
| multiplex (this repo) | ✅ | ✅ | ✅ (exact) | ❌ (this doc) |
| MTPLX v2 | ✅ | ✅ | ✅ | ❌ (vision is Qwen-only) |
| mlx-vlm 0.6.4 | ✅ | ✅ | ✅ | ✅ (blocked on a transformers processor bug) |

The only stack that runs Gemma 4 vision today is mlx-vlm's `gemma4_unified`; it is the
natural encoder to borrow for piece (1).
