# bge-m3 embedder sidecar

A self-contained GPU service that serves `BAAI/bge-m3` and returns **both**
dense (1024-dim) and learned-sparse (lexical) vectors for each input text.
These feed the assistant's hybrid RAG search in Qdrant (dense + sparse).

## API

- `GET /health` -> `{"status": "ok"}`. The health check performs a small GPU
  encoding operation, so it also detects a poisoned CUDA context.
- `POST /embed` with `{"inputs": ["text", ...]}` (empty list -> HTTP 400)

  Response:

  ```json
  {
    "embeddings": [
      {"dense": [0.1, ...], "sparse": {"indices": [123, ...], "values": [0.4, ...]}}
    ]
  }
  ```

## Running

Requires an NVIDIA GPU (FlagEmbedding auto-selects CUDA; fp16 always on):

```bash
docker build -t embedder embedder/
docker run --gpus all --restart unless-stopped -p 8080:8080 \
  -v hf-cache:/root/.cache/huggingface embedder
```

On a CUDA failure the worker exits so Docker can restart it with a clean GPU
context. Keep the restart policy enabled in standalone deployments. The main
`docker-compose.yml` already sets `restart: unless-stopped`.

The model (~2GB) is downloaded on first start and cached in the Hugging Face
cache volume (`HF_HOME=/root/.cache/huggingface`), so subsequent starts are fast.

`EMBEDDING_MODEL` (default `BAAI/bge-m3`) overrides the model id.

Targeted at an NVIDIA RTX 3060 (12GB, Ampere, compute capability **sm_86**);
the base image bundles a CUDA 12.1 / PyTorch 2.3.1 build that supports it.
