"""A local LLM inference framework built on mlx-lm.

Batched speculative decoding: keep MTP speedup while decoding several sequences
together, with sequences able to join/leave the live batch mid-flight.

Layers: Engine (batched forward) -> Drafter (MTP speculation) -> Scheduler
(dynamic batch: prefill, merge, step) -> Hub (funnel many requests onto one
scheduler). The MTP head loads generically; models with no head run pure AR.
"""

from .engine import Engine, BatchState
from .mtp import Drafter
from .scheduler import Scheduler, Req, PrefillGroup
from .hub import Hub

__all__ = ["Engine", "BatchState", "Drafter", "Scheduler", "Req",
           "PrefillGroup", "Hub"]
__version__ = "0.0.0"
