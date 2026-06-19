# CLAUDE.md

## Repo Purpose

Thin, stateless FastAPI proxy putting an AI assistant inside a school VR lab
simulator (physics / chemistry / biology). Retrieval is now **local + hybrid**:
the knowledge base lives in a self-hosted **Qdrant** vector store, queried with
a local **BAAI/bge-m3** GPU embedder (dense + learned-sparse, fused by RRF).
Voice (STT/TTS) is now also **local**: a self-hosted GPU sidecar (Whisper
ru/kk/auto + MMS/Silero ru/kk — the `../vrrag_ttsstt` service) reached over HTTP.
Only generation (answers/hints) still uses **OpenAI cloud** (Responses API).
Scenario context is injected from local JSON.

So this repo self-hosts the vector DB + embedder + voice, and keeps only the LLM
in the cloud — a hybrid of "everything OpenAI" and "everything local".

Sibling reference projects: `../vrrag_dreamlab` (same idea, but fully self-hosted
Postgres/pgvector + local embeddings + vLLM) and `../vrrag_ttsstt` (the STT/TTS
sidecar this repo's voice layer now calls). We share the local-retrieval and
local-voice halves and differ only in keeping the LLM on OpenAI.

## Layout

```
app/
  main.py                 # FastAPI app, lifespan (env validation), CORS, health/ready
  core/config.py          # pydantic-settings; required: INTERNAL_API_KEY, OPENAI_API_KEY
  core/security.py        # Bearer INTERNAL_API_KEY auth
  api/routes.py           # /ask /v1/chat/completions /hint /stt /tts /voice_ask /admin/*
  services/
    openai_client.py      # single shared AsyncOpenAI client (monkeypatch target in tests)
    llm.py                # Responses API: retrieve->inject chunks, citations, streaming, hint rephrasing
    embeddings.py         # httpx client to the bge-m3 embedder sidecar (POST /embed -> dense+sparse)
    vectorstore.py        # Qdrant hybrid wrapper (dense+sparse named vectors, RRF Query API)
    errors.py             # LLMError family (LLMTimeoutError / LLMUpstreamError / LLMMalformedResponseError)
    scenarios.py          # load+format per-lab JSON into the system prompt
    corpus_meta.py        # derive subject/grade/lang/lab_id metadata from corpus paths
    voice.py              # httpx client to the local STT/TTS sidecar (../vrrag_ttsstt)
    ingestion.py          # to_markdown -> chunk+embed (PDF/DOCX/EPUB/TXT/MD) -> Qdrant; bulk ingest + manifest
    memory.py             # request-scoped chat-history trimming
embedder/                 # GPU sidecar container (FlagEmbedding BGEM3FlagModel, RTX 3060 / sm_86)
scripts/manage_corpus.py  # CLI: create-collection / upload / bulk-ingest / gen-manifest / list / status / delete
scenarios/*.json          # one file per lab; filename stem == scenario_id
tests/                    # pytest, OpenAI + Qdrant + embedder mocked (no network)
```

## Request path

```
routes.py
  -> verify_api_key (Bearer INTERNAL_API_KEY)
  -> scenarios.get_scenario_context(scenario_id)   # 404 if unknown
  -> llm.generate_answer / stream_answer
       -> embeddings.embed(query)                   # bge-m3 sidecar -> dense + sparse
       -> vectorstore.search(...)                   # Qdrant hybrid: 2 Prefetch branches, RRF fusion
       -> build_system_prompt(scenario_context, retrieved_chunks)   # inject chunk text
       -> client.responses.create(...)              # NO file_search tool
       -> citations rebuilt from retrieved chunk metadata (filename + doc_id)
  voice: voice.transcribe -> generate_answer -> voice.synthesize  (/voice_ask)
```

## Conventions

- Service layer raises `LLMError` subclasses (`LLMTimeoutError`/`LLMUpstreamError`/
  `LLMMalformedResponseError`), defined in `app/services/errors.py`; routes map
  them via `_handle_llm_error` (504/502).
- RAG is explicit retrieve -> inject -> generate (no hosted `file_search` tool):
  `embeddings.embed` hits the bge-m3 sidecar, `vectorstore.search` runs a Qdrant
  hybrid query (two `Prefetch` branches over the `dense` and `sparse` named
  vectors, fused by RRF), and the top chunks are injected verbatim into the
  system prompt. Citations are rebuilt from chunk metadata (`filename`, `doc_id`)
  — the response shape (`citations`, `primary_source`) is unchanged for the VR client.
- All OpenAI calls go through `app.services.openai_client.client`. Services bind
  `from app.services.openai_client import client`, so tests patch the name on
  each service module (or patch the higher-level functions in `app.api.routes`).
  Qdrant, the embedder and the STT/TTS sidecar are likewise reached through
  `vectorstore`/`embeddings`/`voice` (each a lazy `httpx.AsyncClient`, failures
  mapped onto the same `LLMError` family) and patched the same way in tests.
  Voice no longer touches OpenAI — `voice.transcribe`/`voice.synthesize` POST to
  the `../vrrag_ttsstt` sidecar (`VOICE_BASE_URL`, self-signed TLS, WAV out).
- Responses API param is `max_output_tokens` (not `max_tokens`).
- Qdrant collection has one point per chunk with two named vectors: `dense`
  (1024-d, cosine — `EMBEDDING_DIM`) and `sparse` (learned-sparse from bge-m3).
- Three scene-grounding inputs: the static lab JSON (`scenario_id` ->
  `get_scenario_context`), the live `scenario_state` (current_step + held_items),
  and the structured `lab` context (`subject/grade/lang/lab_number`) the
  simulator sends per request. `routes._lab_dict` composes the canonical
  `lab_id` (e.g. `physics-10-ru-02`); `llm._lab_grounding` then (a) scopes theory
  retrieval to that subject via `vectorstore.meta_filter(doc_type="textbook", …)`
  and (b) injects the lab's procedure verbatim, fetched from Qdrant by `lab_id`
  (`vectorstore.fetch_lab_instruction`). All blocks render via
  `build_system_prompt(...)`.
- Incomplete labs: if a `lab_id` has no instruction in Qdrant, `lab_incomplete`
  is set and the prompt tells the model to answer theory-only and not invent
  steps. `scripts/manage_corpus.py gen-manifest` writes `labs.json` reporting
  which labs are complete/stub and which paths it couldn't tag (topic-named labs
  with no number land there).
- KB metadata: every chunk payload carries `doc_type` (`textbook` |
  `lab_instruction`), `subject`, `grade`, `lang`, and (for labs) `lab_id` /
  `lab_number`, derived from the corpus path by `corpus_meta.parse_path`. Bulk
  ingest the whole tree with `manage_corpus.py bulk-ingest [CORPUS_ROOT]`.
- Prompts/answers are Russian-first; the model mirrors the question language.
  bge-m3 is multilingual (strong on Russian + Kazakh), so retrieval matches.
  Corpus is bilingual (рус/каз); `lang` is `ru`/`kk` (the `русс` typo folder maps
  to `ru`).

## Working here

- Behaviour changes: start in `app/api/routes.py` + `app/services/llm.py`.
- Retrieval changes: `app/services/vectorstore.py` (Qdrant query / RRF) and
  `app/services/embeddings.py` (embedder client); ingestion in
  `app/services/ingestion.py`.
- New settings: add to `app/core/config.py` and `.env.example`. Retrieval knobs:
  `QDRANT_URL`, `QDRANT_COLLECTION`, `EMBEDDING_BASE_URL`, `EMBEDDING_DIM`,
  `RETRIEVAL_TOP_K`, `RETRIEVAL_CANDIDATES`, `RETRIEVAL_SCORE_THRESHOLD`,
  `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CORPUS_ROOT`, `LABS_MANIFEST`. Voice knobs:
  `VOICE_BASE_URL`,
  `VOICE_VERIFY_SSL`, `VOICE_TIMEOUT_S` (STT/TTS language follows
  `DEFAULT_LANGUAGE`). (`OPENAI_VECTOR_STORE_ID` and the old `STT_MODEL`/
  `TTS_MODEL`/`TTS_VOICE`/`TTS_FORMAT`/`TTS_INSTRUCTIONS` are removed.)
- Keep the proxy stateless — no app DB. KB state lives in the Qdrant collection.
- Run `pytest` (fast, fully mocked — no network/GPU) before declaring done.

## Safe commands

```bash
pytest
uvicorn app.main:app --reload --port 8000
python -m scripts.manage_corpus status        # Qdrant collection status
python -m scripts.manage_corpus gen-manifest  # labs.json (no embedding; offline)
python -m scripts.manage_corpus bulk-ingest   # walk CORPUS_ROOT, tag+embed all files
docker compose up --build                     # api + qdrant + embedder
```

`docker compose up --build` brings up three services: `api`, `qdrant`, and the
`embedder` GPU sidecar (needs the NVIDIA Container Toolkit; first boot downloads
bge-m3, so it's slow).
