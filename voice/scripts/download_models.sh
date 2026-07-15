#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/models/hf_cache}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ.get("HF_HOME", "/models/hf_cache")
ru_backends = {
    item.strip().lower()
    for item in os.environ.get("TTS_RU_BACKENDS", "qwen,supertonic").split(",")
    if item.strip()
}
repos = [
    os.environ.get("STT_KK_BASE_MODEL", "openai/whisper-large-v3-turbo"),
    os.environ.get("STT_KK_MODEL", "RakhatM/whisper-large-v3-turbo-kk-lora"),
    os.environ.get("STT_RU_MODEL", "openai/whisper-large-v3-turbo"),
    os.environ.get("TTS_KK_MODEL", "facebook/mms-tts-kaz"),
]
if "mms" in ru_backends:
    repos.append(os.environ.get("TTS_RU_MODEL", "facebook/mms-tts-rus"))
if "qwen" in ru_backends:
    repos.append(
        os.environ.get(
            "TTS_RU_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
    )

for repo in dict.fromkeys(repos):
    snapshot_download(repo_id=repo, cache_dir=cache_dir)
    print(f"downloaded {repo}")
PY
