import os
from pathlib import Path

import pytest

from voice_omnivoice.app.model_resolution import (
    resolve_cached_model_path,
    snapshot_is_complete,
)

REPO = "shyngys879/KazakhTTS-OmniVoice"
FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")


def _snapshot(hub: Path, commit: str, files=FILES, mtime: float | None = None) -> Path:
    repo = hub / "models--shyngys879--KazakhTTS-OmniVoice"
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True)
    blobs = repo / "blobs"
    blobs.mkdir(exist_ok=True)
    for name in files:
        blob = blobs / f"{commit}-{name}"
        blob.write_text("x")
        (snap / name).symlink_to(blob)
    if mtime is not None:
        os.utime(snap, (mtime, mtime))
    return snap


def _point_main(hub: Path, commit: str) -> None:
    refs = hub / "models--shyngys879--KazakhTTS-OmniVoice" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(commit)


def _no_download(repo_id: str) -> str:
    raise AssertionError("Hub must not be contacted")


def _failing_download(repo_id: str) -> str:
    raise OSError("Distant resource does not seem to be on huggingface.co")


def test_complete_current_snapshot_is_used_without_hub(tmp_path):
    snap = _snapshot(tmp_path, "aaa")
    _point_main(tmp_path, "aaa")

    assert resolve_cached_model_path(REPO, tmp_path, _no_download) == str(snap)


def test_incomplete_current_snapshot_downloads(tmp_path):
    _snapshot(tmp_path, "new", files=("model.safetensors",))
    _point_main(tmp_path, "new")
    calls = []

    def download(repo_id):
        calls.append(repo_id)
        for name in FILES[:3]:
            (tmp_path / "models--shyngys879--KazakhTTS-OmniVoice" / "snapshots" / "new" / name).write_text("x")
        return str(tmp_path / "models--shyngys879--KazakhTTS-OmniVoice" / "snapshots" / "new")

    resolved = resolve_cached_model_path(REPO, tmp_path, download)

    assert calls == [REPO]
    assert resolved.endswith("snapshots/new")


def test_hub_failure_falls_back_to_newest_complete_snapshot(tmp_path):
    # The production incident: refs/main moved to a half-downloaded revision
    # and every Hub request fails, while a complete older copy is cached.
    _snapshot(tmp_path, "oldest", mtime=1_000)
    older = _snapshot(tmp_path, "older", mtime=2_000)
    _snapshot(tmp_path, "new", files=("model.safetensors",), mtime=3_000)
    _point_main(tmp_path, "new")

    assert resolve_cached_model_path(REPO, tmp_path, _failing_download) == str(older)


def test_incomplete_download_falls_back_to_cache(tmp_path):
    good = _snapshot(tmp_path, "good")
    partial = _snapshot(tmp_path, "partial", files=("model.safetensors",))
    _point_main(tmp_path, "partial")

    assert resolve_cached_model_path(REPO, tmp_path, lambda _: str(partial)) == str(good)


def test_no_cache_and_hub_failure_raises(tmp_path):
    with pytest.raises(RuntimeError, match="No complete local copy"):
        resolve_cached_model_path(REPO, tmp_path, _failing_download)


def test_local_directory_is_returned_as_is(tmp_path):
    assert resolve_cached_model_path(str(tmp_path), tmp_path / "hub", _no_download) == str(tmp_path)


def test_dangling_weight_symlink_is_incomplete(tmp_path):
    snap = _snapshot(tmp_path, "aaa")
    (snap / "model.safetensors").resolve().unlink()

    assert not snapshot_is_complete(snap)


def test_load_model_calls_cache_first_resolver_not_omnivoice_one():
    # Regression: load_model keeps OmniVoice's own `_resolve_model_path` in a
    # local named `resolve_model_path`, which once shadowed the cache-first
    # resolver and crashed startup with a TypeError.
    import ast
    import inspect

    from voice_omnivoice.app import main

    tree = ast.parse(inspect.getsource(main.OmniVoiceBackend.load_model).lstrip())
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "resolve_cached_model_path" in called
    assert "resolve_cached_model_path" not in assigned
