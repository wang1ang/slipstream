import json
from types import SimpleNamespace

from multiplex.hub import Hub
from multiplex.server import _chat_stream


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class TinyTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt=True, **kwargs):
        body = "\n".join(
            f"{item['role']}:{item.get('content') or ''}" for item in messages
        )
        if add_generation_prompt:
            body += "\nassistant:"
            if kwargs.get("enable_thinking") is False:
                body += "<think>\n\n</think>\n\n"
            else:
                body += "<think>\n"
        return _ids(body)

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token)) for token in token_ids)


def _hub_with_raw_chunks(chunks):
    hub = Hub.__new__(Hub)
    hub.eng = SimpleNamespace(tokenizer=TinyTokenizer())
    hub._default_enable_thinking = None
    hub._stream_prompt_ids = lambda _prompt_ids, _max_tokens: iter(chunks)
    return hub


def test_l4_splits_prefilled_qwen_thinking_before_l5():
    hub = _hub_with_raw_chunks(["private ", "plan</thi", "nk>Visible answer"])

    parts = list(
        hub.stream_message_parts(
            [{"role": "user", "content": "hi"}],
            32,
            enable_thinking=True,
        )
    )

    reasoning = "".join(text for field, text in parts if field == "reasoning_content")
    content = "".join(text for field, text in parts if field == "content")

    assert reasoning == "private plan"
    assert content == "Visible answer"
    assert "</think>" not in content


def test_l4_reasoning_off_strips_orphan_close_marker():
    hub = _hub_with_raw_chunks(["</think>Visible answer"])

    parts = list(
        hub.stream_message_parts(
            [{"role": "user", "content": "hi"}],
            32,
            enable_thinking=False,
        )
    )

    assert parts == [("content", "Visible answer")]


def test_l5_chat_stream_maps_l4_reasoning_to_reasoning_content_delta():
    class Backend:
        model_id = "tiny"

        def stream_message_parts(self, *_args, **_kwargs):
            yield "reasoning_content", "private plan"
            yield "content", "Visible answer"

    payloads = []
    for event in _chat_stream(
        Backend(), [{"role": "user", "content": "hi"}], 32
    ):
        if not event.startswith("data: ") or event.strip() == "data: [DONE]":
            continue
        payloads.append(json.loads(event[len("data: ") :]))

    deltas = [payload["choices"][0]["delta"] for payload in payloads]
    assert {"reasoning_content": "private plan"} in deltas
    assert {"content": "Visible answer"} in deltas
