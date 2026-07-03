"""L5 — HTTP / OpenAI-compatible layer. PROTOCOL TRANSLATION ONLY.

Translates HTTP requests <-> internal (prompt_ids, tools, params) and back to
JSON / SSE. It does NOT schedule, batch, or manage cache: requests go to the L4
Hub, which runs the L3 scheduler on one engine thread and serves many HTTP
handler threads concurrently. Tool-call text <-> structured tool_calls is
handled by the bridge layer.

Endpoints (each a different wire format, all funneled to the same Hub):
  * POST /v1/chat/completions   — classic Chat Completions (+ SSE)
  * POST /v1/responses          — OpenAI Responses API (what Codex uses; +SSE)
  * GET  /v1/models

Sync stdlib http.server pairs with the Hub's blocking text queues directly.
"""

from __future__ import annotations

import os
import json
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .bridge import ToolCallStreamFilter, extract_tool_calls_with_thinking
from .engine import find_mtp
from .hub import Hub


def _split_tool_calls(text, tools):
    """Parse the model's generated text into (visible_text, tool_calls). Only
    attempts parsing when tools were offered; returns [] calls otherwise."""
    if not tools:
        return text, []
    res = extract_tool_calls_with_thinking("", text, None, tools)
    return res.cleaned_text, res.tool_calls or []


# The backend is the L4 Hub: it runs the scheduler on one engine thread and lets
# many HTTP handler threads submit requests + drain their text streams. So the
# HTTP server can be multi-threaded (handlers only touch queues, never MLX).


def _hex(prefix):
    return prefix + uuid.uuid4().hex


def _messages_from_chat(body):
    # Roles are normalized to the model's template downstream (Engine), so L5
    # only translates wire shape here — it passes roles through verbatim.
    return list(body.get("messages", []))


def _content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None:
                    text = part.get("output")
                if text is not None:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _arguments_text(arguments):
    if isinstance(arguments, str):
        return arguments
    if arguments is None:
        return "{}"
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))


def _function_call_message(item):
    call_id = str(item.get("call_id") or item.get("id") or "")
    function = item.get("function") if isinstance(item.get("function"), dict) else {}
    name = str(item.get("name") or function.get("name") or "")
    if not name:
        return None
    arguments = item.get("arguments", function.get("arguments", "{}"))
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": _arguments_text(arguments)},
        }],
    }


def _messages_from_responses(body):
    """Responses: optional `instructions` (system) + `input` (str or items).
    Roles are passed through verbatim (developer frames included); role mapping
    and system consolidation happen downstream in Hub.prompt_ids (bridge).
    """
    msgs = []
    if body.get("instructions"):
        msgs.append({"role": "system", "content": body["instructions"]})
    inp = body.get("input")
    if isinstance(inp, str):
        msgs.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            typ = item.get("type")
            if typ in ("message", None):
                msgs.append({
                    "role": item.get("role", "user"),
                    "content": _content_text(item.get("content")),
                })
            elif typ == "function_call":
                msg = _function_call_message(item)
                if msg is not None:
                    msgs.append(msg)
            elif typ == "function_call_output":
                msgs.append({
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": _content_text(item.get("output", item.get("content"))),
                })
    return msgs


def _assistant_messages(text, tool_calls=None):
    if tool_calls:
        return [{"role": "assistant", "content": text or "", "tool_calls": tool_calls}]
    if text:
        return [{"role": "assistant", "content": text}]
    return []


# --- SSE encoders -------------------------------------------------------------
def _sse(data, event=None):
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data)}\n\n"


def _stream_visible(backend, messages, max_tokens, tools):
    """Yield (visible_text_chunk, tool_calls). Visible chunks come as generated,
    with tool-call markup filtered out by the bridge when tools are offered; the
    final yield has an empty chunk and the tool_calls parsed from the full text
    (the oMLX contract: filter markup live, parse tool calls at completion)."""
    filt = ToolCallStreamFilter() if tools else None
    full = ""
    for text in backend.stream_messages(messages, max_tokens, tools):
        full += text
        visible = filt.feed(text) if filt else text
        if visible:
            yield visible, None
    if filt:
        tail = filt.finish()
        if tail:
            yield tail, None
        _, calls = _split_tool_calls(full, tools)
        yield "", calls


def _chat_stream(backend, messages, max_tokens, tools=None):
    rid, created = _hex("chatcmpl-"), int(time.time())

    def chunk(delta, finish=None):
        return _sse({
            "id": rid, "object": "chat.completion.chunk", "created": created,
            "model": backend.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })

    yield chunk({"role": "assistant"})
    calls = []
    for visible, tool_calls in _stream_visible(backend, messages, max_tokens, tools):
        if visible:
            yield chunk({"content": visible})
        if tool_calls:
            calls = tool_calls
    if calls:
        yield chunk({"tool_calls": [
            {"index": i, "id": c.get("id"), "type": "function",
             "function": c.get("function")} for i, c in enumerate(calls)]})
        yield chunk({}, finish="tool_calls")
    else:
        yield chunk({}, finish="stop")
    yield "data: [DONE]\n\n"


def _responses_stream(backend, messages, max_tokens, tools=None, on_complete=None):
    rid, mid = _hex("resp_"), _hex("msg_")
    base = {"id": rid, "object": "response", "model": backend.model_id}
    message_item = {"id": mid, "type": "message", "role": "assistant",
                    "content": [], "status": "in_progress"}

    yield _sse({"type": "response.created", "response": {**base, "status": "in_progress"}},
               "response.created")
    yield _sse({"type": "response.output_item.added", "item": message_item},
               "response.output_item.added")

    full, calls = "", []
    for visible, tool_calls in _stream_visible(backend, messages, max_tokens, tools):
        if visible:
            full += visible
            yield _sse({"type": "response.output_text.delta", "item_id": mid, "delta": visible},
                       "response.output_text.delta")
        if tool_calls:
            calls = tool_calls
    yield _sse({"type": "response.output_text.done", "item_id": mid, "text": full},
               "response.output_text.done")
    message_item = {**message_item, "status": "completed",
                    "content": [{"type": "output_text", "text": full, "annotations": []}]}
    yield _sse({"type": "response.output_item.done", "item": message_item},
               "response.output_item.done")

    output = [message_item]
    # A function_call output item per parsed tool call (added -> args delta -> done).
    for c in calls:
        fn = c.get("function") or {}
        item_id = _hex("fc_")
        call_id = str(c.get("id") or _hex("call_"))
        item = {"id": item_id, "type": "function_call", "call_id": call_id,
                "name": fn.get("name") or "", "arguments": fn.get("arguments") or ""}
        yield _sse({"type": "response.output_item.added",
                    "item": {**item, "status": "in_progress", "arguments": ""}},
                   "response.output_item.added")
        yield _sse({"type": "response.function_call_arguments.delta",
                    "item_id": item["id"], "delta": item["arguments"]},
                   "response.function_call_arguments.delta")
        yield _sse({"type": "response.output_item.done", "item": {**item, "status": "completed"}},
                   "response.output_item.done")
        output.append({**item, "status": "completed"})

    yield _sse({"type": "response.completed", "response": {
        **base, "status": "completed", "created_at": int(time.time()), "output": output,
    }}, "response.completed")
    if on_complete is not None:
        on_complete(rid, _assistant_messages(full, calls))


# --- non-stream bodies --------------------------------------------------------
def _chat_body(backend, text, tool_calls=None):
    message = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": _hex("chatcmpl-"), "object": "chat.completion",
        "created": int(time.time()), "model": backend.model_id,
        "choices": [{"index": 0,
                     "finish_reason": "tool_calls" if tool_calls else "stop",
                     "message": message}],
    }


def _responses_body(backend, text, tool_calls=None):
    # Responses uses separate output items: a message for text, a function_call
    # item per tool call (call_id/name/arguments), mirroring llama.cpp's mapping.
    output = []
    if text:
        output.append({"id": _hex("msg_"), "type": "message", "role": "assistant",
                       "status": "completed",
                       "content": [{"type": "output_text", "text": text, "annotations": []}]})
    for c in tool_calls or []:
        fn = c.get("function") or {}
        output.append({"id": _hex("fc_"), "type": "function_call", "status": "completed",
                       "call_id": str(c.get("id") or _hex("call_")),
                       "name": fn.get("name") or "",
                       "arguments": fn.get("arguments") or ""})
    return {
        "id": _hex("resp_"), "object": "response", "created_at": int(time.time()),
        "status": "completed", "model": backend.model_id, "output": output,
    }


# --- HTTP handler -------------------------------------------------------------
def make_handler(backend: Hub):
    response_history: dict[str, list[dict]] = {}
    response_history_order: list[str] = []
    response_history_lock = threading.Lock()

    def response_messages(body):
        prev_id = body.get("previous_response_id")
        if prev_id:
            with response_history_lock:
                previous = list(response_history.get(str(prev_id), []))
        else:
            previous = []
        return previous + _messages_from_responses(body)

    def remember_response(rid, messages):
        with response_history_lock:
            response_history[str(rid)] = messages
            response_history_order.append(str(rid))
            while len(response_history_order) > 32:
                old = response_history_order.pop(0)
                response_history.pop(old, None)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse_stream(self, gen):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for ev in gen:
                self.wfile.write(ev.encode())
                self.wfile.flush()

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": backend.model_id, "object": "model", "owned_by": "local"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.rstrip("/")
            stream = bool(body.get("stream", False))
            max_tokens = int(body.get("max_output_tokens")
                             or body.get("max_tokens") or 2048)

            tools = body.get("tools")
            if path == "/v1/chat/completions":
                msgs = _messages_from_chat(body)
                if stream:
                    self._sse_stream(_chat_stream(backend, msgs, max_tokens, tools))
                else:
                    text = "".join(backend.stream_messages(msgs, max_tokens, tools))
                    clean, calls = _split_tool_calls(text, tools)
                    self._json(200, _chat_body(backend, clean, calls))
            elif path == "/v1/responses":
                msgs = response_messages(body)
                if stream:
                    self._sse_stream(_responses_stream(
                        backend, msgs, max_tokens, tools,
                        on_complete=lambda rid, out: remember_response(rid, msgs + out),
                    ))
                else:
                    text = "".join(backend.stream_messages(msgs, max_tokens, tools))
                    clean, calls = _split_tool_calls(text, tools)
                    resp = _responses_body(backend, clean, calls)
                    remember_response(resp["id"], msgs + _assistant_messages(clean, calls))
                    self._json(200, resp)
            else:
                self._json(404, {"error": "not found"})

    return Handler


def serve(model_path: str, mtp_path: str | None, host="127.0.0.1", port=8000,
          debug=False, prefix_cache_dir="auto"):
    # mtp_path None -> auto-detect <model>/mtp.safetensors; a given path that is
    # absent (or "" to force it) -> headless (pure AR).
    if mtp_path is None:
        mtp_path = find_mtp(model_path)
    elif not os.path.exists(mtp_path):
        mtp_path = None
    print(f"[{'MTP head: ' + mtp_path if mtp_path else 'headless (pure AR)'}]")
    backend = Hub(model_path, mtp_path, debug=debug,
                  prefix_cache_dir=prefix_cache_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(backend))
    print(f"[serving {backend.model_id} on http://{host}:{port}  "
          f"(/v1/chat/completions, /v1/responses, /v1/models)]")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    import sys

    from . import registry

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="model path or name; default: scan ~/.mtplx/models")
    # Default: derive <model>/mtp.safetensors (present -> speculate, absent -> AR).
    ap.add_argument("--mtp", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                    help="log scheduler activity (prefill/join/advance/exit); "
                         "on by default, --no-debug to silence")
    ap.add_argument("--prefix-cache-dir", default="auto",
                    help="prefix cache load dir; default 'auto' reads "
                         "~/.cache/multiplex/prefixcache/<model>.")
    args = ap.parse_args()

    # registry.select behaves per-environment: a server run without a tty gets
    # the "list + pick with --model" error instead of an interactive prompt.
    try:
        entry = registry.select(args.model)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        sys.exit(str(e))
    serve(entry.path, args.mtp, host=args.host, port=args.port, debug=args.debug,
          prefix_cache_dir=args.prefix_cache_dir)
