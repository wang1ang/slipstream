"""Model registry — discover models under a root directory.

Discovery ONLY: it finds model directories and resolves a user argument to one.
It does not validate weights (load-time does), and it does not decide whether a
model has an MTP head (that's the model's business — use engine.find_mtp on the
resolved path). Download/cache management will extend this module later.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

DEFAULT_ROOT = "~/.mtplx/models"


@dataclass
class ModelEntry:
    name: str      # directory name, for display / selection
    path: str      # absolute path to the model directory


def list_models(root: str = DEFAULT_ROOT) -> list[ModelEntry]:
    """Every immediate subdirectory of ``root`` that has a config.json — that
    marks it as a model. No weight validation (load-time handles that). Sorted
    by name for stable listing."""
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return []
    entries = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "config.json")):
            entries.append(ModelEntry(name=name, path=path))
    return entries


def resolve(arg: str | None, root: str = DEFAULT_ROOT) -> ModelEntry | list[ModelEntry]:
    """Resolve a user's model argument to a single model, or hand the caller a
    list to choose from.

    * ``arg`` is a path (absolute or ~/./..) — return that single ModelEntry
      (its name is the directory's basename); the path need not be under root.
    * ``arg`` is a bare name — match it against a discovered model by name.
    * ``arg`` is None — scan root: exactly one model returns it directly, zero
      or many return the list (the caller reports "none found" / lets the user
      pick). Selection UI belongs to the caller, not here.
    """
    if arg is not None:
        if arg.startswith(("/", "~", "./", "../")):
            path = os.path.expanduser(arg.rstrip("/"))
            return ModelEntry(name=os.path.basename(path), path=path)
        for e in list_models(root):
            if e.name == arg:
                return e
        raise FileNotFoundError(f"no model named {arg!r} under {os.path.expanduser(root)}")

    models = list_models(root)
    return models[0] if len(models) == 1 else models


def select(arg: str | None, root: str = DEFAULT_ROOT) -> ModelEntry:
    """Resolve to exactly ONE model, shared by every entry point so they behave
    identically. A single match (given arg, or the sole model) returns straight
    away. For several candidates the behaviour follows the environment, not the
    caller: an interactive terminal prompts a numbered choice; without a tty
    (e.g. a server started in the background) it lists them and raises, telling
    the user to pass --model. Zero models always raises.
    """
    r = resolve(arg, root)
    if not isinstance(r, list):
        return r

    where = os.path.expanduser(root)
    if not r:
        raise FileNotFoundError(f"no models found under {where}")

    lines = "\n".join(f"  [{i}] {e.name}" for i, e in enumerate(r))
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"multiple models under {where} — choose one with --model NAME:\n{lines}")
    print(lines)
    raw = input(f"pick a model [0-{len(r) - 1}]: ").strip()
    if not raw.isdigit() or not (0 <= int(raw) < len(r)):
        raise ValueError(f"invalid selection: {raw!r}")
    return r[int(raw)]
