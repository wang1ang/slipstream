"""A local LLM inference framework built on mlx-lm.

Batched speculative decoding: keep MTP speedup while decoding several sequences
together, with sequences able to join/leave the live batch mid-flight.

Layers live under ``multiplex.kernel``: Engine (batched forward) -> Drafter
(MTP speculation) -> Scheduler (dynamic batch: prefill, merge, step) -> Hub
(funnel many requests onto one scheduler). The MTP head loads generically;
models with no head run pure AR.
"""

from .kernel import (
    BatchState,
    Drafter,
    Engine,
    Hub,
    PrefillGroup,
    Req,
    Scheduler,
    find_mtp,
)

__all__ = ["Engine", "BatchState", "Drafter", "find_mtp", "Scheduler", "Req",
           "PrefillGroup", "Hub"]
__version__ = "0.0.0"
