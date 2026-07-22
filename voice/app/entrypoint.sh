#!/bin/sh
set -u

# Persistence mode reduces driver reinitialization after idle periods. This is
# best-effort because some hosts restrict NVML administration from containers.
# The application must still start when the driver rejects the request.
if [ "${NVIDIA_PERSISTENCE_MODE:-1}" != "0" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi -pm 1; then
            echo "NVIDIA persistence mode enabled"
        else
            echo "WARNING: could not enable NVIDIA persistence mode" >&2
        fi
    else
        echo "WARNING: nvidia-smi is unavailable; persistence mode was not changed" >&2
    fi
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1
