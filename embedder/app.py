"""GPU embedding sidecar for BAAI/bge-m3 (dense + learned-sparse).

Serves a single FlagEmbedding BGEM3FlagModel and returns both the dense
1024-dim embedding and the learned-sparse (lexical) weights for each input.
The sparse output is shaped for Qdrant hybrid search (indices + values).

Designed to run on an NVIDIA RTX 3060 (12GB, Ampere, sm_86). FlagEmbedding
auto-selects CUDA when available; fp16 is always enabled to fit the GPU.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Model id is overridable via env; device is auto-selected by FlagEmbedding.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Loaded once at startup, shared across requests.
_model = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the BGE-M3 model once before serving traffic."""
    global _model
    # Imported lazily so the module can be inspected without heavy deps loaded.
    from FlagEmbedding import BGEM3FlagModel

    # use_fp16=True keeps the model within the 3060's 12GB budget.
    _model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)
    yield
    _model = None


app = FastAPI(title="bge-m3 embedder", lifespan=lifespan)


class EmbedRequest(BaseModel):
    """Batch of texts to embed."""

    inputs: list[str]


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/embed")
async def embed(req: EmbedRequest) -> dict:
    """Return dense + learned-sparse vectors for each input text.

    Response shape:
        {"embeddings": [
            {"dense": [...], "sparse": {"indices": [...], "values": [...]}},
            ...
        ]}
    """
    if not req.inputs:
        raise HTTPException(status_code=400, detail="inputs must be a non-empty list")

    out = _model.encode(
        req.inputs,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    # dense_vecs: numpy array [n, 1024]; one row per input.
    dense_vecs = out["dense_vecs"]
    # lexical_weights: list of dicts {token_id(str) -> weight(float)} per input.
    lexical_weights = out["lexical_weights"]

    embeddings = []
    for dense_row, weights in zip(dense_vecs, lexical_weights):
        # Preserve key/value pairing order from the dict.
        indices = [int(tok) for tok in weights]
        values = [float(w) for w in weights.values()]
        embeddings.append(
            {
                "dense": [float(x) for x in dense_row],
                "sparse": {"indices": indices, "values": values},
            }
        )

    return {"embeddings": embeddings}


if __name__ == "__main__":
    # Local run; the Dockerfile invokes uvicorn directly.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
