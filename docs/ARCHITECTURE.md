# Architecture

`multiplex/kernel/` contains the stable L1-L4 inference kernel:

```text
L4  kernel/hub.py           many caller threads -> one engine thread
L3  kernel/scheduler.py     prefill, merge, decode step, cancel, prefix cache
L2  kernel/mtp.py           MTP sidecar, draft generation, norm/quant loading
L1  kernel/engine.py        batched forward, logits, cache clone/filter/restore
    kernel/prefixcache/     L3 prefix-cache policy, state adapter, disk store
L5  server.py               OpenAI-compatible HTTP / SSE / JSON translation
```

The kernel is intended to be boring and stable. After this move, avoid editing
`multiplex/kernel/` unless the requested behavior truly requires changing the
core inference path. Prefer placing protocol, client compatibility, message
normalization, and parser changes at the package edge, especially in `bridge/`,
`registry.py`, tests, or docs.

Kernel-external dependency boundaries:

- L1-L3 must not depend on `multiplex/` modules outside `kernel/`. They may
  depend on third-party runtime packages such as MLX/MLX-LM, but not on bridge,
  registry, HTTP, or client adapter code.
- L4 is the first layer allowed to depend on kernel-external package code. Today
  that means only `bridge.normalize_messages_for_template` for model input
  normalization and `bridge.ThinkingParser` for model-native output marker
  parsing such as thinking blocks.
- L5 is outside `kernel/` and may depend on protocol/client adapters and CLI
  model selection. Today that means `bridge/` for tool-call stream filtering
  and `registry.py` in the CLI entrypoint.
- OpenAI-compatible JSON, SSE event names, HTTP status shapes, and
  client-specific wire quirks must stay at L5 and must not leak into L1-L4.

Compatibility shims remain at the old kernel module paths (`multiplex.engine`,
`multiplex.mtp`, `multiplex.scheduler`, `multiplex.hub`, and
`multiplex.prefixcache.*`) so existing scripts keep working while new core
imports can use `multiplex.kernel.*`.
