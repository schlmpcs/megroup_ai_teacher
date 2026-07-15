# CLAUDE.md

dont coauthor commits

## Repo Purpose

Thin, stateless FastAPI proxy putting an AI assistant inside a school VR lab
simulator (physics / chemistry / biology). Retrieval is now **local + hybrid**:
the knowledge base lives in a self-hosted **Qdrant** vector store, queried with
a local **BAAI/bge-m3** GPU embedder (dense + learned-sparse, fused by RRF).
Voice (STT/TTS) is now also **local**: an in-repo GPU sidecar (Whisper
ru/kk/auto + Qwen3-TTS 0.6B default/Supertonic selectable for ru + MMS kaz,
the `voice` compose service in `./voice`,
vendored from the former `../vrrag_ttsstt`) reached over HTTP.
Only generation (answers/hints) still uses **OpenAI cloud** (Responses API).
Scenario context is injected from local JSON.

So this repo self-hosts the vector DB + embedder + voice, and keeps only the LLM
in the cloud. It is a hybrid of "everything OpenAI" and "everything local".

Sibling reference project: `../vrrag_dreamlab` (same idea, but fully self-hosted
Postgres/pgvector + local embeddings + vLLM). The STT/TTS service that this
repo's voice layer calls is now vendored in-repo under `./voice` (originally the
standalone `../vrrag_ttsstt`). We share the local-retrieval and local-voice
halves and differ only in keeping the LLM on OpenAI.

## Layout

```text
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
    voice.py              # httpx client to the in-repo STT/TTS sidecar (./voice)
    ingestion.py          # to_markdown -> chunk+embed (PDF/DOCX/EPUB/TXT/MD) -> Qdrant; bulk ingest + manifest
    memory.py             # request-scoped chat-history trimming
embedder/                 # GPU sidecar container (FlagEmbedding BGEM3FlagModel, RTX 3060 / sm_86)
voice/                    # GPU STT/TTS sidecar (Whisper + Qwen3-TTS/Supertonic/MMS); vendored from ../vrrag_ttsstt
scripts/manage_corpus.py  # CLI: create-collection / upload / bulk-ingest / gen-manifest / list / status / delete
scenarios/*.json          # one file per lab; filename stem == scenario_id
tests/                    # pytest, OpenAI + Qdrant + embedder mocked (no network)
```

## Request path

```text
routes.py
  -> verify_api_key (Bearer INTERNAL_API_KEY)
  -> scenarios.get_scenario_context(scenario_id)   # 404 if unknown
  -> llm.generate_answer / stream_answer
       -> embeddings.embed(query)                   # bge-m3 sidecar -> dense + sparse
       -> vectorstore.search(...)                   # Qdrant hybrid: 2 Prefetch branches, RRF fusion
       -> build_system_prompt(scenario_context, retrieved_chunks)   # inject chunk text
       -> client.responses.create(...)              # NO file_search tool
       -> citations rebuilt from retrieved chunk metadata (filename + file_id + locators)
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
  system prompt. Citations are rebuilt from chunk metadata and keep the legacy
  `filename`/`file_id` fields while adding source type/path, lab metadata, chunk,
  page, chapter, section and `display_label` locators. The response shape
  (`citations`, `primary_source`) is unchanged for the VR client.
- All OpenAI calls go through `app.services.openai_client.client`. Services bind
  `from app.services.openai_client import client`, so tests patch the name on
  each service module (or patch the higher-level functions in `app.api.routes`).
  Qdrant, the embedder and the STT/TTS sidecar are likewise reached through
  `vectorstore`/`embeddings`/`voice` (each a lazy `httpx.AsyncClient`, failures
  mapped onto the same `LLMError` family) and patched the same way in tests.
  Voice no longer touches OpenAI. `voice.transcribe`/`voice.synthesize` POST to
  the in-repo `voice` sidecar (`./voice`, the `voice` compose service) at
  `VOICE_BASE_URL` (plain HTTP over the compose network, WAV out). Russian TTS
  defaults to Qwen3-TTS 0.6B; `backend=supertonic` selects the comparison model.
- Responses API param is `max_output_tokens` (not `max_tokens`).
- Qdrant collection has one point per chunk with two named vectors: `dense`
  (1024-d, cosine, configured by `EMBEDDING_DIM`) and `sparse` (learned-sparse
  from bge-m3).
- Three scene-grounding inputs: the static lab JSON (`scenario_id` ->
  `get_scenario_context`), the authoritative live `scenario_state` (current/next
  step ids and text, completed steps, held/visible items, allowed actions and
  last-action result),
  and the structured `lab` context (`subject/grade/lang/lab_number`) the
  simulator sends per request. `routes._lab_dict` composes the canonical
  `lab_id` (e.g. `physics-10-ru-02`); `llm._lab_grounding` then (a) scopes theory
  retrieval to that subject via `vectorstore.meta_filter(doc_type="textbook", …)`
  and (b) injects the lab's procedure verbatim, fetched from Qdrant by `lab_id`
  (`vectorstore.fetch_lab_instruction_record`). Lab-instruction payloads are also
  included when building citations. Citation ordering is intent-aware: procedure
  and current-step questions prefer the lab instruction, while theory questions
  prefer retrieved textbooks. All blocks render via `build_system_prompt(...)`.
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
  `VOICE_VERIFY_SSL`, `VOICE_TIMEOUT_S`, `VOICE_TTS_RU_DEFAULT_BACKEND`
  (STT/TTS language follows
  `DEFAULT_LANGUAGE`). (`OPENAI_VECTOR_STORE_ID` and the old `STT_MODEL`/
  `TTS_MODEL`/`TTS_VOICE`/`TTS_FORMAT`/`TTS_INSTRUCTIONS` are removed.)
- Keep the proxy stateless. There is no app DB; KB state lives in Qdrant.
- Run `pytest` (fast, fully mocked, no network/GPU) before declaring done.

## Safe commands

```bash
pytest
uvicorn app.main:app --reload --port 8000
python -m scripts.manage_corpus status        # Qdrant collection status
python -m scripts.manage_corpus gen-manifest  # labs.json (no embedding; offline)
python -m scripts.manage_corpus bulk-ingest   # walk CORPUS_ROOT, tag+embed all files
docker compose up --build                     # api + qdrant + embedder + voice
```

`docker compose up --build` brings up four services: `api`, `qdrant`, the
`embedder` GPU sidecar, and the `voice` GPU STT/TTS sidecar (both need the NVIDIA
Container Toolkit; the embedder and voice share the single GPU, and first boot
downloads bge-m3 + Whisper/Qwen3-TTS/Supertonic/MMS models, so it's slow).

## Running accuracy evals

Answer-accuracy eval = LLM-as-judge over a fixed teacher question set. Input is
`test_questions.md` (735 Q + reference answers; physics/chemistry/biology, grades
7–11, 5 Q per lab). It hits a **live, already-deployed** `/ask` endpoint. It does
**not** spin up local code, so deploy your change to the eval box first
(`git pull` + `docker compose build api` on the GPU server) or the run measures
the old code. Three steps:

```bash
# 1. Run the question set against a live /ask and emit grading inputs.
#    EVAL_API_KEY must match that server's INTERNAL_API_KEY (remote secret;
#    pass via env, never hardcode). ~13 min at 4 workers.
EVAL_BASE_URL=http://megroup-b560m-hdv-m-2:8001 EVAL_API_KEY=<server INTERNAL_API_KEY> \
  python -m scripts.eval.run_eval
#    -> eval_results.json/.md, eval_slim.json, eval_batches/batch_*.json (49 × 15 Q)

# 2. Grade every answer (this is an LLM-judge step, done by Claude subagents;
#    there is no grader script). Spawn ~10 parallel agents, each assigned a group
#    of batch_*.json files, each comparing `answer` vs `expected` and WRITING its
#    grades to eval_batches/grades_gNN.json as [{"id","score","verdict"}].
#    Rubric: score 0–100 on scientific correctness/coverage vs the reference;
#    verdict bands  >=70 correct / 40–69 partial / <40 incorrect; empty answer = 0;
#    judge meaning not wording; ignore citations/length/ru-vs-kk. Cover all 735 ids.

# 3. Merge grades + render the dashboard.
python -m scripts.eval.build_dashboard
#    -> grades.json, eval_graded.json, chart_data.json, eval_accuracy_dashboard.html
open eval_accuracy_dashboard.html   # KPIs, verdict donut, score hist, by subject/grade,
                                    # subject×grade heatmap, weakest/strongest labs
```

`build_dashboard.py` recomputes verdicts from score (so the bands always hold) and
defaults any ungraded id to 0/incorrect. Everything except `test_questions.md` and
the two scripts is a generated artifact (safe to delete and regenerate). Baseline
to beat: **75.9% strict** / 84.2% lenient / avg 74.0. The dominant failure mode is
retrieval/KB-coverage gaps, not generation. When accuracy is low, inspect the
lowest-scoring answers in `eval_graded.json` (most are grounded "нет информации в
материалах" refusals or wrong-grade citations), not the prompt.
