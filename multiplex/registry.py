"""Model registry — discover model directories.

Discovery ONLY: it finds model directories and resolves a user argument to one.
It does not validate weights; load-time code does that.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse

DEFAULT_ROOT = "~/.mtplx/models"
DEFAULT_MODELS = (
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen3-1.7B-4bit",
    "mlx-community/Qwen3-4B-4bit",
    "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
)


@dataclass
class ModelEntry:
    name: str      # directory name, for display / selection
    path: str      # absolute path to the model directory


def list_models(root: str = DEFAULT_ROOT) -> list[ModelEntry]:
    """Every immediate subdirectory of ``root`` that is a model directory.

    A directory counts as a model if it has a ``config.json`` at its root
    (single-model layout), or a ``mtplx_pair.json`` at its root (assistant-pair
    bundle: the per-model config.json lives under target/ and assistant/).
    Discovery only — no weight validation (load-time handles that). Sorted by
    name for stable listing."""
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return []
    entries = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(os.path.join(path, "config.json")) or os.path.isfile(
            os.path.join(path, "mtplx_pair.json")
        ):
            entries.append(ModelEntry(name=name, path=path))
    return entries


def _hf_names(model: str) -> tuple[str, str]:
    """Return ``(repo_id, local_dir_name)`` for an HF id or HF model URL."""
    if model.startswith(("http://", "https://")):
        parsed = urlparse(model)
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co", "hf.co"}:
            raise ValueError(f"unsupported model URL: {model}")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"unsupported Hugging Face model URL: {model}")
        repo_id = "/".join(parts[:2])
    else:
        repo_id = model.strip("/")
    return repo_id, repo_id.replace("/", "--")


def download_model(model: str, root: str = DEFAULT_ROOT) -> ModelEntry:
    """Download an HF repo into the local model root and return its entry."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download models; install multiplex again"
        ) from e

    repo_id, local_name = _hf_names(model)
    root = os.path.expanduser(root)
    path = os.path.join(root, local_name)
    print(f"[downloading HF model {repo_id} -> {path}]", file=sys.stderr)
    snapshot_download(repo_id=repo_id, local_dir=path)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(f"downloaded model has no config.json: {path}")
    return ModelEntry(name=os.path.basename(path), path=path)


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
        _, local_name = _hf_names(arg)
        for e in list_models(root):
            if e.name == arg or e.name == local_name:
                return e
        return download_model(arg, root)

    models = list_models(root)
    return models[0] if len(models) == 1 else models


def _missing_default_models(models: list[ModelEntry]) -> list[str]:
    local_names = {e.name for e in models}
    missing = []
    for model in DEFAULT_MODELS:
        repo_id, local_name = _hf_names(model)
        if repo_id in local_names or local_name in local_names:
            continue
        missing.append(model)
    return missing


def select(arg: str | None, root: str = DEFAULT_ROOT) -> ModelEntry:
    """Resolve to exactly ONE model, shared by every entry point so they behave
    identically. A single match (given arg, or the sole model) returns straight
    away. For several candidates, or when only downloadable defaults are
    available, the behaviour follows the environment, not the caller: an
    interactive terminal prompts a numbered choice, with Enter selecting the
    first model; without a tty (e.g. a server started in the background) it
    lists them and raises, telling the user to pass --model.
    """
    r = resolve(arg, root)
    if not isinstance(r, list):
        return r

    where = os.path.expanduser(root)
    downloads = _missing_default_models(r)
    if not r and not downloads:
        raise FileNotFoundError(f"no models found under {where}")

    labels = [e.name for e in r]
    labels.extend(f"{_hf_names(model)[0]} (需下载)" for model in downloads)
    lines = "\n".join(f"  [{i}] {label}" for i, label in enumerate(labels))
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"multiple models under {where} — choose one with --model NAME:\n{lines}")
    print(lines)
    raw = input(f"pick a model [0-{len(labels) - 1}] (default 0): ").strip()
    if raw == "":
        raw = "0"
    if not raw.isdigit() or not (0 <= int(raw) < len(labels)):
        raise ValueError(f"invalid selection: {raw!r}")
    index = int(raw)
    if index < len(r):
        return r[index]
    return download_model(downloads[index - len(r)], root)
