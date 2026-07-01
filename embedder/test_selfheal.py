"""Self-check for the embedder CUDA self-heal predicate.

Run: ``python embedder/test_selfheal.py`` (no framework, no GPU, no torch).
Loaded by path because the proxy also ships a top-level ``app`` package.
"""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "embedder_app", pathlib.Path(__file__).with_name("app.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Poisoned-context and OOM RuntimeErrors must trigger self-heal...
assert _mod._is_cuda_error(
    RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable")
)
assert _mod._is_cuda_error(RuntimeError("CUDA out of memory"))
# ...but ordinary errors must NOT (else the container crash-loops on bad input).
assert not _mod._is_cuda_error(ValueError("inputs must be a non-empty list"))
assert not _mod._is_cuda_error(RuntimeError("some unrelated cpu failure"))

print("ok")
