# SPDX-License-Identifier: Apache-2.0
"""OpenAI/OpenCode protocol bridge — message normalization and tool-call parsing.

Self-contained (stdlib only), adapted from oMLX; the Responses<->Chat mapping
(server.py) follows llama.cpp. Kept as its own package so the wire adaptation
lives apart from the engine/scheduler layers.
"""

from .adapter import normalize_messages_for_template
from .thinking import ThinkingParser, extract_thinking
from .tool_calling import (
    ToolCallExtraction,
    ToolCallStreamFilter,
    extract_tool_calls_with_thinking,
    parse_tool_calls,
)

__all__ = [
    "ThinkingParser",
    "ToolCallExtraction",
    "ToolCallStreamFilter",
    "extract_thinking",
    "extract_tool_calls_with_thinking",
    "normalize_messages_for_template",
    "parse_tool_calls",
]
