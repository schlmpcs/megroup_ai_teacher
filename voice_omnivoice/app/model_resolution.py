"""Resolve the OmniVoice checkpoint to a complete local snapshot.

OmniVoice's own loader calls ``snapshot_download(repo)`` on every start, so a
Hugging Face outage, a broken mirror, or a half-finished download of a new
upstream revision makes the service crash-loop even when a complete copy of the
model already sits in the cache. This resolver prefers the cache and only
touches the Hub when no usable local copy exists:

1. The snapshot that ``refs/main`` points to, if every required file is there.
2. Otherwise a Hub download (``download``), accepted only if it is complete.
3. If that fails, the newest complete cached snapshot of any revision.

A complete cached snapshot is therefore used without contacting the Hub, so a
new upstream revision is not picked up automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger("omnivoice")

REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def snapshot_is_complete(path: Path) -> bool:
    """True when the snapshot has config, tokenizer and weight files.

    ``is_file`` follows the cache's symlinks, so a link whose blob never
    finished downloading counts as missing.
    """
    if not path.is_dir():
        return False
    if not all((path / name).is_file() for name in REQUIRED_FILES):
        return False
    weights = list(path.glob("*.safetensors"))
    return bool(weights) and all(weight.is_file() for weight in weights)


def _repo_cache_dir(repo_id: str, hub_cache: Path) -> Path:
    return hub_cache / ("models--" + repo_id.replace("/", "--"))


def _current_snapshot(repo_dir: Path) -> Path | None:
    ref = repo_dir / "refs" / "main"
    try:
        commit = ref.read_text().strip()
    except OSError:
        return None
    return repo_dir / "snapshots" / commit if commit else None


def _complete_snapshots_newest_first(repo_dir: Path) -> list[Path]:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return []
    complete = [p for p in snapshots_dir.iterdir() if snapshot_is_complete(p)]
    return sorted(complete, key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_model_path(
    repo_id_or_path: str,
    hub_cache: str | Path,
    download: Callable[[str], str],
) -> str:
    """Return a local directory holding a complete copy of the model."""
    if Path(repo_id_or_path).is_dir():
        return repo_id_or_path

    repo_dir = _repo_cache_dir(repo_id_or_path, Path(hub_cache))
    current = _current_snapshot(repo_dir)
    if current is not None and snapshot_is_complete(current):
        logger.info("Using cached OmniVoice snapshot %s", current.name)
        return str(current)

    try:
        downloaded = Path(download(repo_id_or_path))
    except Exception as exc:  # noqa: BLE001 - any Hub failure falls back
        failure: Exception = exc
    else:
        if snapshot_is_complete(downloaded):
            logger.info("Downloaded OmniVoice snapshot %s", downloaded.name)
            return str(downloaded)
        failure = RuntimeError(f"downloaded snapshot {downloaded} is incomplete")

    for fallback in _complete_snapshots_newest_first(repo_dir):
        logger.warning(
            "OmniVoice Hub download failed (%s); using cached snapshot %s",
            failure,
            fallback.name,
        )
        return str(fallback)

    raise RuntimeError(
        f"No complete local copy of {repo_id_or_path} and the Hub download "
        f"failed: {failure}"
    ) from failure
