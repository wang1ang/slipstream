import builtins
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path


def _load_registry():
    path = Path(__file__).parent / "multiplex" / "registry.py"
    spec = importlib.util.spec_from_file_location("registry_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


registry = _load_registry()


def test_missing_named_model_downloads_from_hf(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, root=registry.DEFAULT_ROOT):
        calls.append((repo_id, root))
        return registry.ModelEntry(name="org--repo", path=str(tmp_path / "org--repo"))

    monkeypatch.setattr(registry, "download_model", fake_download)

    entry = registry.select("org/repo", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert calls == [("org/repo", str(tmp_path))]


def test_existing_local_name_wins_over_hf_download(monkeypatch, tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("should not download an existing local model")

    monkeypatch.setattr(registry, "download_model", fail_download)

    entry = registry.select("local-model", root=str(tmp_path))

    assert entry.name == "local-model"
    assert entry.path == str(model_dir)


def test_existing_hf_style_local_dir_wins_over_download(monkeypatch, tmp_path):
    model_dir = tmp_path / "org--repo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("should not download an existing local model")

    monkeypatch.setattr(registry, "download_model", fail_download)

    entry = registry.select("org/repo", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert entry.path == str(model_dir)


def test_huggingface_url_downloads_repo_id(monkeypatch, tmp_path):
    calls = []

    def fake_download(model, root=registry.DEFAULT_ROOT):
        calls.append((model, root))
        return registry.ModelEntry(name="org--repo", path=str(tmp_path / "org--repo"))

    monkeypatch.setattr(registry, "download_model", fake_download)

    entry = registry.select("https://huggingface.co/org/repo/tree/main", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert calls == [
        ("https://huggingface.co/org/repo/tree/main", str(tmp_path))
    ]


def test_huggingface_url_reuses_existing_local_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "org--repo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("should not download an existing local model")

    monkeypatch.setattr(registry, "download_model", fail_download)

    entry = registry.select("https://huggingface.co/org/repo", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert entry.path == str(model_dir)


def test_missing_huggingface_url_calls_snapshot_download_without_network(
    monkeypatch, tmp_path
):
    calls = []

    def fake_snapshot_download(*, repo_id, local_dir):
        calls.append((repo_id, local_dir))
        Path(local_dir).mkdir(parents=True)
        (Path(local_dir) / "config.json").write_text("{}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    entry = registry.select("https://huggingface.co/org/repo/tree/main", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert entry.path == str(tmp_path / "org--repo")
    assert calls == [("org/repo", str(tmp_path / "org--repo"))]


def test_interactive_selection_can_download_default_model(
    monkeypatch, tmp_path, capsys
):
    calls = []
    default_model = "org/default-model"

    def fake_download(model, root=registry.DEFAULT_ROOT):
        calls.append((model, root))
        return registry.ModelEntry(
            name="org--default-model",
            path=str(tmp_path / "org--default-model"),
        )

    monkeypatch.setattr(registry, "DEFAULT_MODELS", (default_model,))
    monkeypatch.setattr(registry, "download_model", fake_download)
    monkeypatch.setattr(registry.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(builtins, "input", lambda _prompt: "0")

    entry = registry.select(None, root=str(tmp_path))

    assert entry.name == "org--default-model"
    assert calls == [(default_model, str(tmp_path))]
    assert "org/default-model (需下载)" in capsys.readouterr().out


def test_noninteractive_selection_marks_missing_default_models(
    monkeypatch, tmp_path
):
    for name in ("a-model", "b-model"):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")

    monkeypatch.setattr(registry, "DEFAULT_MODELS", ("org/default-model",))
    monkeypatch.setattr(registry.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    try:
        registry.select(None, root=str(tmp_path))
    except RuntimeError as e:
        message = str(e)
    else:
        raise AssertionError("expected noninteractive selection to fail")

    assert "a-model" in message
    assert "b-model" in message
    assert "org/default-model (需下载)" in message
