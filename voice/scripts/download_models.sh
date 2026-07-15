#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/models/hf_cache}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.environ.get("HF_HOME", "/models/hf_cache")
qwen_model = os.environ.get(
    "TTS_RU_QWEN_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
)
ru_backends = {
    item.strip().lower()
    for item in os.environ.get("TTS_RU_BACKENDS", "supertonic,qwen").split(",")
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
    repos.append(qwen_model)

for repo in dict.fromkeys(repos):
    # qwen-tts resolves its nested speech tokenizer through HF_HOME/hub and has
    # an upstream cache_dir mismatch for subfolder metadata. Keep Qwen in the
    # standard hub cache; the other existing models retain their legacy path.
    kwargs = {"repo_id": repo}
    if repo != qwen_model:
        kwargs["cache_dir"] = cache_dir
    snapshot_download(**kwargs)
    print(f"downloaded {repo}")
PY
