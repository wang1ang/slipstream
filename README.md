# multiplex

Local LLM serving on Apple Silicon, built on [mlx-lm](https://github.com/ml-explore/mlx-lm).

`multiplex` is a compact inference stack for personal agent workloads: fast local
generation, OpenAI-compatible HTTP endpoints, streaming, tool-call adaptation,
dynamic batching, and prefix reuse for long conversations.

The core bet is simple: local models should feel responsive even when an agent
fans out into several overlapping requests.

## Why

Most local serving stacks make a tradeoff:

- speculative decoding works well for one request;
- batching works for multiple requests;
- long agent conversations repeatedly resend the same prefix.

`multiplex` puts those paths in one engine:

- **MTP speculative decoding** when a model ships an MTP sidecar.
- **True batched decode** with several live sequences in one forward pass.
- **Dynamic join/leave** so requests can enter and exit the live batch mid-flight.
- **Prefix cache** so retries and multi-turn agent calls can reuse already-prefilled state.
- **OpenAI-compatible HTTP** for `/v1/responses`, `/v1/chat/completions`, and `/v1/models`.

If no MTP head is available, the same stack runs pure autoregressive decoding.

## Features

- Built directly on `mlx-lm` and MLX cache primitives.
- One engine thread owns MLX execution; HTTP threads submit requests and stream deltas.
- Streaming SSE for Chat Completions and Responses.
- Tool-call bridge for model text -> structured OpenAI-style tool calls.
- Model discovery under `~/.mtplx/models`.
- Optional on-disk prefix cache under `~/.cache/multiplex/prefixcache`.
- Interactive REPL for testing dynamic batching locally.

## Quick Start

Install:

```bash
pip install -r requirements.txt
pip install -e ".[cli]"
```

Start the HTTP server:

```bash
python -m multiplex.server --model /path/to/model --host 127.0.0.1 --port 8000
```

Or use a model directory name discovered under `~/.mtplx/models`:

```bash
python -m multiplex.server --model MODEL_NAME
```

Try the dynamic-batch REPL:

```bash
python try_engine.py --model /path/to/model
```

While one prompt is generating, submit another prompt to see it join the live
batch.

## API

`multiplex` exposes the endpoints most local OpenAI-compatible clients expect:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

Both streaming and non-streaming responses are supported.

Example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a tiny haiku about MLX."}]
  }'
```

## MTP Heads

MTP sidecars are discovered automatically from:

- `<model>/mtplx_runtime.json`
- `<model>/mtp.safetensors`
- `<model>/mtp/weights.safetensors`

When a sidecar is present, `multiplex` loads the MTP head and uses speculative
decode. When no sidecar is present, the scheduler switches to pure AR without
changing the serving API.

## Prefix Cache

Agent clients often resend long, nearly identical prompts. `multiplex` captures
cache state at chunk boundaries, finds the longest token-identical prefix on the
next request, restores that state, and only prefills the new tail.

By default, HTTP serving uses an automatic cache directory:

```text
~/.cache/multiplex/prefixcache/<model-name>-<model-path-sha>
```

Override it with:

```bash
python -m multiplex.server --model /path/to/model --prefix-cache-dir /path/to/cache
```

## Architecture

The stack stays deliberately narrow:

```text
server.py        OpenAI-compatible HTTP and SSE translation
hub.py           concurrent callers -> one MLX engine thread
scheduler.py     dynamic prefill/decode batch management
mtp.py           MTP draft/verify speculative decoding
engine.py        batched mlx-lm forward and cache operations
prefixcache/     prefix matching, cache snapshots, optional disk persistence
bridge/          message normalization and tool-call parsing
```

The lower layers do not know about HTTP, and the HTTP layer does not manage
model cache. That split keeps the serving surface small while leaving the engine
free to optimize batching and reuse.

## Requirements

- macOS with an available Metal device.
- Python 3.10+.
- `mlx` and `mlx-lm`.
- A local MLX model directory.

## Status

`multiplex` is an active local-serving experiment focused on Apple Silicon and
agent-style workloads. It is useful today for local testing and integration, and
the current work is centered on compatibility polish, prefix-cache measurement,
and smarter draft-depth policy.
