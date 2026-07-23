# Multi-Assistant Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one FastAPI deployment serve predefined assistant types, each with its own trusted prompt, Qdrant collection, and corpus root while sharing scenarios, embeddings, OpenAI generation, STT, and TTS.

**Architecture:** Add a small server-owned assistant profile registry and carry the validated profile identifier through generation, retrieval, memory, and ingestion. Pass the resolved Qdrant collection explicitly to every vector-store operation, and namespace answer cache and conversation memory by assistant type. Existing callers omit `assistant_type` and continue using `vr_lab_teacher`.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, Qdrant, bge-m3 embedder sidecar, OpenAI Responses API, pytest, SQLite ingestion queue

## Global Constraints

- Consumer requests may send only `assistant_type`; never accept raw prompts or Qdrant collection names.
- Use one unique Qdrant collection per assistant profile.
- Keep the existing `QDRANT_COLLECTION` and `CORPUS_ROOT` settings as the `vr_lab_teacher` defaults.
- Preserve current clients when `assistant_type` is omitted.
- Reuse the existing scenario, lab metadata, citation, language, STT, and TTS paths.
- Leave `/hint`, standalone `/stt`, and standalone `/tts` profile-independent.
- Keep the current admin UI on the default profile; use the CLI or admin API for non-default corpus operations until a profile selector is explicitly requested.
- Do not mutate `settings.QDRANT_COLLECTION` or `settings.CORPUS_ROOT` per request.
- Do not fall back from one assistant collection to another assistant collection.
- Preserve the current general-knowledge fallback rules inside the selected assistant profile.
- Keep the answer-cache clear operation global; per-profile invalidation is not needed.
- Keep tests fully mocked with no Qdrant, embedder, OpenAI, or voice network calls.
- Do not add dependencies.
- Do not add coauthor trailers to commits.
- Do not use em dashes.

## File Map

- Create: `app/services/assistant_profiles.py` - trusted profile registry and validation.
- Create: `tests/test_assistant_profiles.py` - registry validation and lookup tests.
- Modify: `app/main.py` - validate profiles during startup.
- Modify: `app/services/vectorstore.py` - explicit collection argument for every Qdrant operation.
- Modify: `app/services/ingestion.py` - pass collection through upload, bulk ingest, prune, list, delete, and status.
- Modify: `app/services/llm.py` - profile prompt composition, collection-aware retrieval, and cache isolation.
- Modify: `app/services/memory.py` - namespace conversations by assistant type.
- Modify: `app/api/routes.py` - consumer request fields, validation, forwarding, response metadata, and memory namespace.
- Modify: `app/api/admin_routes.py` - profile-scoped corpus operations and queued jobs.
- Modify: `app/services/ingestion_jobs.py` - persist upload-job options and preserve them on retry.
- Modify: `app/services/ingestion_worker.py` - resolve profile, confine corpus reads, and write to the selected collection.
- Modify: `scripts/manage_corpus.py` - add `--assistant-type` to corpus commands.
- Modify: `tests/test_vectorstore_write_safety.py`, `tests/test_corpus_units.py`, `tests/test_retrieval_units.py`.
- Modify: `tests/test_ingestion_units.py`, `tests/test_ingestion_safety_units.py` - existing ingestion doubles accept collection selection.
- Modify: `tests/test_llm_units.py`, `tests/test_cache_units.py`, `tests/test_memory_units.py`, `tests/test_api.py`.
- Modify: `tests/test_admin_auth_units.py`, `tests/test_admin_ingestion_api_units.py`, `tests/test_document_upload_api_units.py`, `tests/test_ingestion_jobs_units.py`, `tests/test_ingestion_worker_units.py`, `tests/test_manage_corpus_cli_units.py`.
- Modify: `README.md`, `.env.example`, `docs/memory-backend-guide.md`.

---

### Task 1: Add The Trusted Assistant Profile Registry

**Files:**
- Create: `app/services/assistant_profiles.py`
- Create: `tests/test_assistant_profiles.py`
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `settings.QDRANT_COLLECTION`, `settings.CORPUS_ROOT`.
- Produces: `AssistantProfile`, `DEFAULT_ASSISTANT_TYPE`, `ASSISTANT_PROFILES`, `get_assistant_profile()`, `validate_assistant_profiles()`.

- [ ] **Step 1: Write failing registry tests**

Create `tests/test_assistant_profiles.py`:

```python
import pytest

from app.core.config import settings
from app.services import assistant_profiles
from app.services.assistant_profiles import AssistantProfile


def test_default_profile_uses_existing_settings():
    profile = assistant_profiles.get_assistant_profile(None)

    assert profile.assistant_type == "vr_lab_teacher"
    assert profile.qdrant_collection == settings.QDRANT_COLLECTION
    assert profile.corpus_root == settings.CORPUS_ROOT
    assert "school VR laboratory" in profile.system_prompt


def test_unknown_profile_lists_allowed_identifiers():
    with pytest.raises(ValueError, match="vr_lab_teacher"):
        assistant_profiles.get_assistant_profile("missing")


def test_registry_rejects_duplicate_collections():
    profiles = {
        "first": AssistantProfile("first", "Prompt one", "shared", "/one"),
        "second": AssistantProfile("second", "Prompt two", "shared", "/two"),
    }

    with pytest.raises(ValueError, match="shared"):
        assistant_profiles.validate_assistant_profiles(profiles, default="first")
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/test_assistant_profiles.py -q
```

Expected: collection fails because `app.services.assistant_profiles` does not exist.

- [ ] **Step 3: Create the profile registry**

Create `app/services/assistant_profiles.py`:

```python
import re
from collections.abc import Mapping
from dataclasses import dataclass

from app.core.config import settings

DEFAULT_ASSISTANT_TYPE = "vr_lab_teacher"
_ASSISTANT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

VR_LAB_TEACHER_PROMPT = (
    "You are a friendly teaching assistant inside a school VR laboratory for "
    "physics, chemistry, and biology. Help the student understand the current "
    "activity, its safe next actions, and the theory directly related to it. "
    "Usually answer in 1 to 4 concise sentences in a warm teacher-like tone."
)


@dataclass(frozen=True)
class AssistantProfile:
    assistant_type: str
    system_prompt: str
    qdrant_collection: str
    corpus_root: str


ASSISTANT_PROFILES = {
    DEFAULT_ASSISTANT_TYPE: AssistantProfile(
        assistant_type=DEFAULT_ASSISTANT_TYPE,
        system_prompt=VR_LAB_TEACHER_PROMPT,
        qdrant_collection=settings.QDRANT_COLLECTION,
        corpus_root=settings.CORPUS_ROOT,
    )
}


def available_assistant_types() -> tuple[str, ...]:
    return tuple(sorted(ASSISTANT_PROFILES))


def get_assistant_profile(assistant_type: str | None) -> AssistantProfile:
    key = assistant_type or DEFAULT_ASSISTANT_TYPE
    try:
        return ASSISTANT_PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(available_assistant_types())
        raise ValueError(
            f"Unknown assistant_type '{key}'. Available: {choices}"
        ) from exc


def validate_assistant_profiles(
    profiles: Mapping[str, AssistantProfile] | None = None,
    *,
    default: str = DEFAULT_ASSISTANT_TYPE,
) -> None:
    configured = ASSISTANT_PROFILES if profiles is None else profiles
    if default not in configured:
        raise ValueError(f"Default assistant profile '{default}' is missing")

    collections: dict[str, str] = {}
    for key, profile in configured.items():
        if not _ASSISTANT_TYPE_RE.fullmatch(key):
            raise ValueError(f"Invalid assistant_type '{key}'")
        if profile.assistant_type != key:
            raise ValueError(f"Profile key '{key}' does not match its assistant_type")
        if not profile.system_prompt.strip():
            raise ValueError(f"Assistant profile '{key}' has an empty system prompt")
        if not profile.qdrant_collection.strip():
            raise ValueError(f"Assistant profile '{key}' has an empty collection")
        if not profile.corpus_root.strip():
            raise ValueError(f"Assistant profile '{key}' has an empty corpus root")
        previous = collections.get(profile.qdrant_collection)
        if previous is not None:
            raise ValueError(
                f"Qdrant collection '{profile.qdrant_collection}' is shared by "
                f"assistant profiles '{previous}' and '{key}'"
            )
        collections[profile.qdrant_collection] = key
```

- [ ] **Step 4: Validate profiles during FastAPI startup**

In `app/main.py`, import the registry and call the validator before initializing
ingestion jobs:

```python
from app.services.assistant_profiles import (
    ASSISTANT_PROFILES,
    validate_assistant_profiles,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = missing_required_env_vars()
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
    validate_assistant_profiles()
    ingestion_jobs.initialize()
    ingestion_jobs.cleanup_pending_deletions()
    ingestion_jobs.cleanup_stale_tmp()
    collections = ", ".join(
        f"{key}={profile.qdrant_collection}"
        for key, profile in sorted(ASSISTANT_PROFILES.items())
    )
    logging.getLogger("assistant").info(
        "Retrieval backend: Qdrant %s, assistant collections [%s], embedder %s",
        settings.QDRANT_URL,
        collections,
        settings.EMBEDDING_BASE_URL,
    )
    yield
```

Replace the old single-collection startup log with this profile-aware log.

Update `test_startup_resumes_pending_deletions_after_schema_init` in
`tests/test_api.py` so the event order is explicit:

```python
monkeypatch.setattr(
    main,
    "validate_assistant_profiles",
    lambda: events.append("profiles"),
)

assert events == ["profiles", "initialize", "pending", "stale"]
```

- [ ] **Step 5: Run registry and startup tests**

Run:

```bash
pytest tests/test_assistant_profiles.py tests/test_api.py::test_startup_resumes_pending_deletions_after_schema_init -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the registry**

```bash
git add app/services/assistant_profiles.py app/main.py tests/test_assistant_profiles.py tests/test_api.py
git commit -m "feat: add assistant profile registry"
```

### Task 2: Make Qdrant And Ingestion Collection-Aware

**Files:**
- Modify: `app/services/vectorstore.py`
- Modify: `app/services/ingestion.py`
- Modify: `tests/test_vectorstore_write_safety.py`
- Modify: `tests/test_corpus_units.py`
- Modify: `tests/test_retrieval_units.py`
- Modify: `tests/test_ingestion_units.py`
- Modify: `tests/test_ingestion_safety_units.py`

**Interfaces:**
- Consumes: profile `qdrant_collection` strings.
- Produces: optional `collection_name` keyword on all vector-store and ingestion operations.

- [ ] **Step 1: Write failing collection-routing tests**

Add to `tests/test_vectorstore_write_safety.py`:

```python
async def test_hybrid_search_uses_explicit_collection(monkeypatch):
    seen = []

    class _Client:
        async def query_points(self, **kwargs):
            seen.append(kwargs["collection_name"])
            return SimpleNamespace(points=[])

    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client())

    result = await vectorstore.hybrid_search(
        [0.1],
        [],
        [],
        top_k=5,
        candidates=10,
        collection_name="domain_kb",
    )

    assert result == []
    assert seen == ["domain_kb"]
```

Add to `tests/test_retrieval_units.py`:

```python
async def test_upload_document_uses_explicit_collection(monkeypatch):
    seen = []

    async def _embed_texts(texts):
        return [Embedding(dense=[0.1], sparse_indices=[], sparse_values=[])]

    async def _ensure_collection(collection_name=None):
        seen.append(("ensure", collection_name))

    async def _upsert_points(points, collection_name=None):
        seen.append(("upsert", collection_name))
        return len(points)

    monkeypatch.setattr(ingestion, "to_markdown", lambda *args, **kwargs: "usable text")
    monkeypatch.setattr(ingestion.embeddings, "embed_texts", _embed_texts)
    monkeypatch.setattr(ingestion.vectorstore, "ensure_collection", _ensure_collection)
    monkeypatch.setattr(ingestion.vectorstore, "upsert_points", _upsert_points)

    await ingestion.upload_document(
        "notes.md",
        b"notes",
        collection_name="domain_kb",
    )

    assert seen == [("ensure", "domain_kb"), ("upsert", "domain_kb")]
```

Add to `tests/test_corpus_units.py`:

```python
async def test_collection_status_does_not_fall_back_to_default(monkeypatch):
    seen = []

    class _Client:
        async def collection_exists(self, name):
            seen.append(name)
            return False

    monkeypatch.setattr(vectorstore, "get_client", lambda: _Client())

    status = await vectorstore.collection_status("domain_kb")

    assert seen == ["domain_kb"]
    assert status["status"] == "unconfigured"
    assert status["collection"] == "domain_kb"
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/test_vectorstore_write_safety.py::test_hybrid_search_uses_explicit_collection tests/test_retrieval_units.py::test_upload_document_uses_explicit_collection tests/test_corpus_units.py::test_collection_status_does_not_fall_back_to_default -q
```

Expected: FAIL because the functions do not accept `collection_name`.

- [ ] **Step 3: Add one collection resolver in the vector store**

Add near the vector constants in `app/services/vectorstore.py`:

```python
def _collection_name(collection_name: str | None) -> str:
    return collection_name or settings.QDRANT_COLLECTION
```

Change the public signatures to:

```python
async def ensure_collection(collection_name: str | None = None) -> None:
async def upsert_points(points: list[dict], collection_name: str | None = None) -> int:
async def hybrid_search(
    dense: list[float],
    sparse_indices: list[int],
    sparse_values: list[float],
    top_k: int,
    candidates: int,
    query_filter: "models.Filter | None" = None,
    collection_name: str | None = None,
) -> list[dict]:
async def fetch_lab_instruction_record(
    lab_id: str,
    collection_name: str | None = None,
) -> dict | None:
async def fetch_lab_instruction(
    lab_id: str,
    collection_name: str | None = None,
) -> str:
async def list_documents(collection_name: str | None = None) -> list[dict]:
async def delete_document(
    doc_id: str,
    collection_name: str | None = None,
) -> bool:
async def collection_status(collection_name: str | None = None) -> dict:
```

At the start of each function, resolve once and use that local value for every
Qdrant call in the function:

```python
collection = _collection_name(collection_name)
```

For example, `hybrid_search()` must call:

```python
result = await client.query_points(
    collection_name=collection,
    prefetch=[
        models.Prefetch(
            query=dense,
            using=DENSE,
            limit=candidates,
            filter=visible_filter,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
            using=SPARSE,
            limit=candidates,
            filter=visible_filter,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=max(top_k, candidates),
    with_payload=True,
)
```

`fetch_lab_instruction()` and `collection_status()` must forward the same
resolved collection to their helper calls:

```python
record = await fetch_lab_instruction_record(
    lab_id,
    collection_name=collection,
)

document_rows = await list_documents(collection_name=collection)
```

- [ ] **Step 4: Thread collection names through ingestion**

Add `collection_name: str | None = None` to `upload_document()`,
`prune_missing_corpus_documents()`, and `bulk_ingest_tree()` in
`app/services/ingestion.py`.

Use these exact calls in `upload_document()`:

```python
await vectorstore.ensure_collection(collection_name=collection_name)
n = await vectorstore.upsert_points(
    points,
    collection_name=collection_name,
)
```

Use these calls in `prune_missing_corpus_documents()`:

```python
for document in await vectorstore.list_documents(collection_name=collection_name):
    source = document.get("source_path")
    file_id = document.get("file_id")
    if (
        source
        and file_id
        and not source.startswith("admin_uploads/")
        and corpus_meta.parse_path(source) is not None
        and file_id not in present_doc_ids
        and await vectorstore.delete_document(
            file_id,
            collection_name=collection_name,
        )
    ):
        pruned += 1
```

In `bulk_ingest_tree()`, pass `collection_name` to each `upload_document()` call
and to `prune_missing_corpus_documents()`.

Change the wrappers at the bottom of `ingestion.py` to:

```python
async def list_documents(collection_name: str | None = None, **_) -> list[dict]:
    return await vectorstore.list_documents(collection_name=collection_name)


async def delete_document(
    file_id: str,
    collection_name: str | None = None,
    **_,
) -> bool:
    return await vectorstore.delete_document(
        file_id,
        collection_name=collection_name,
    )


async def corpus_status(collection_name: str | None = None, **_) -> dict:
    return await vectorstore.collection_status(collection_name=collection_name)
```

- [ ] **Step 5: Update existing test doubles for the new optional keyword**

Change monkeypatched vector-store and ingestion functions in
`tests/test_corpus_units.py`, `tests/test_retrieval_units.py`,
`tests/test_ingestion_units.py`, `tests/test_ingestion_safety_units.py`, and
`tests/test_vectorstore_write_safety.py` to accept `collection_name=None` or
`**kwargs` when the production caller now forwards the keyword. Preserve every
existing assertion and add collection assertions only where the selected
collection is the behavior under test.

- [ ] **Step 6: Run the storage and ingestion tests**

Run:

```bash
pytest tests/test_vectorstore_write_safety.py tests/test_corpus_units.py tests/test_retrieval_units.py tests/test_ingestion_units.py tests/test_ingestion_safety_units.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit collection-aware storage**

```bash
git add app/services/vectorstore.py app/services/ingestion.py tests/test_vectorstore_write_safety.py tests/test_corpus_units.py tests/test_retrieval_units.py tests/test_ingestion_units.py tests/test_ingestion_safety_units.py
git commit -m "feat: scope Qdrant operations by collection"
```

### Task 3: Route Profiles Through Prompting, Retrieval, And Answer Cache

**Files:**
- Modify: `app/services/llm.py`
- Modify: `tests/test_llm_units.py`
- Modify: `tests/test_cache_units.py`
- Modify: `tests/test_corpus_units.py`

**Interfaces:**
- Consumes: `get_assistant_profile(assistant_type)` and collection-aware vector-store calls.
- Produces: `assistant_type` keyword on `generate_answer()` and `stream_answer()`.

- [ ] **Step 1: Write failing profile-grounding and cache tests**

Add to `tests/test_llm_units.py`:

```python
from app.services import assistant_profiles
from app.services.assistant_profiles import AssistantProfile


async def test_generate_answer_uses_profile_prompt_and_collection(monkeypatch):
    captured = {}
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )

    async def _retrieve(
        query,
        query_filter=None,
        lang=None,
        fallback_filter=None,
        collection_name=None,
    ):
        captured["collection"] = collection_name
        return []

    async def _create(**kwargs):
        captured["instructions"] = kwargs["instructions"]
        return SimpleNamespace(
            output_text="Answer",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        )

    monkeypatch.setattr(llm, "_retrieve", _retrieve)
    monkeypatch.setattr(llm.client.responses, "create", _create)

    await llm.generate_answer(
        "Question",
        assistant_type="domain_assistant",
        answer_language="en",
    )

    assert captured["collection"] == "domain_kb"
    assert "You are the domain assistant." in captured["instructions"]
    assert "school VR laboratory" not in captured["instructions"]
```

Add to `tests/test_cache_units.py`:

```python
async def test_answer_cache_isolated_by_assistant_type(monkeypatch):
    fake = _FakeClient()
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )
    monkeypatch.setattr(llm, "client", fake)
    monkeypatch.setattr(llm, "_retrieve", _no_retrieve)

    await llm.generate_answer("Same question")
    await llm.generate_answer(
        "Same question",
        assistant_type="domain_assistant",
    )

    assert len(fake.responses.calls) == 2
```

Import `assistant_profiles` and `AssistantProfile` at the top of
`tests/test_cache_units.py`.

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/test_llm_units.py::test_generate_answer_uses_profile_prompt_and_collection tests/test_cache_units.py::test_answer_cache_isolated_by_assistant_type -q
```

Expected: FAIL because generation does not accept or resolve `assistant_type`.

- [ ] **Step 3: Split profile identity from shared prompt rules**

In `app/services/llm.py`, import:

```python
from app.services.assistant_profiles import (
    DEFAULT_ASSISTANT_TYPE,
    AssistantProfile,
    get_assistant_profile,
)
```

Replace `_BASE_SYSTEM_PROMPT` with domain-neutral shared invariants:

```python
_SHARED_SYSTEM_PROMPT = (
    "Rules:\n"
    "1. The requested answer language is stated explicitly below and must be "
    "followed. Supported answer languages are Russian, Kazakh, and English.\n"
    "2. The retrieved knowledge, laboratory instruction, static scenario, and "
    "live scene state are authoritative. Do not contradict them. Evidence may "
    "be written in another supported language. Translate it faithfully into the "
    "requested answer language without changing facts, measurements, or steps.\n"
    "3. General domain knowledge is allowed only when a special fallback mode "
    "below explicitly enables it. Otherwise use only the supplied evidence.\n"
    "4. If the permitted evidence does not support an answer, say so briefly and "
    "do not invent details.\n"
    "5. Follow the selected assistant profile for tone, scope, and answer length.\n"
    "6. Do not append a source line. Citations are attached separately.\n"
    "7. Use earlier conversation turns only to resolve context, never as factual "
    "evidence.\n"
)
```

Change the general fallback and forced-fallback wording from "scientific
knowledge" to "reliable, widely accepted knowledge relevant to the selected
assistant's domain." Keep the private answer markers and fallback decision logic
unchanged.

Add the final optional parameter to `build_system_prompt()`:

```python
def build_system_prompt(
    scenario_context: Optional[str] = None,
    scenario_state: Optional[str] = None,
    knowledge_context: Optional[str] = None,
    lab_instruction: Optional[str] = None,
    lab_incomplete: bool = False,
    answer_language: Optional[LanguageCode] = None,
    allow_general_knowledge: bool = False,
    strict_lab_scope: bool = False,
    assistant_instruction: Optional[str] = None,
) -> str:
    language = answer_language or normalize_language_code(
        settings.DEFAULT_LANGUAGE,
        field="DEFAULT_LANGUAGE",
    )
    instruction = (
        assistant_instruction
        or get_assistant_profile(DEFAULT_ASSISTANT_TYPE).system_prompt
    )
    prompt = f"{instruction.strip()}\n\n{_SHARED_SYSTEM_PROMPT}"
```

Keep the existing language, scope, knowledge, lab, scenario, and fallback block
assembly unchanged after that initialization.

- [ ] **Step 4: Pass collection through retrieval and lab grounding**

Add `collection_name: str | None = None` to `_search()` and `_retrieve()`. Pass
it to every search call:

```python
chunks = await vectorstore.hybrid_search(
    dense,
    sparse_indices,
    sparse_values,
    top_k=settings.RETRIEVAL_TOP_K,
    candidates=settings.RETRIEVAL_CANDIDATES,
    collection_name=collection_name,
    **kwargs,
)
```

Change `_lab_grounding()` to accept the same keyword and fetch the procedure
from that collection:

```python
record = await vectorstore.fetch_lab_instruction_record(
    lab_id,
    collection_name=collection_name,
)
```

Change `_prepare_answer_grounding()` to consume an `AssistantProfile`:

```python
async def _prepare_answer_grounding(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab: Optional[dict],
    retrieval_query: Optional[str] = None,
    answer_language: Optional[LanguageCode] = None,
    assistant_profile: Optional[AssistantProfile] = None,
) -> _AnswerGrounding:
    profile = assistant_profile or get_assistant_profile(DEFAULT_ASSISTANT_TYPE)
```

Pass `profile.qdrant_collection` to `_lab_grounding()` and `_retrieve()`, then
pass `profile.system_prompt` to `build_system_prompt()` as
`assistant_instruction`.

- [ ] **Step 5: Isolate the answer cache and public generation functions**

Add `assistant_type` to `_answer_cache_key()` and its returned tuple:

```python
def _answer_cache_key(
    query: str,
    scenario_context: Optional[str],
    scenario_state: Optional[str],
    lab: Optional[dict],
    max_tokens: Optional[int],
    answer_language: LanguageCode = "ru",
    assistant_type: str = DEFAULT_ASSISTANT_TYPE,
) -> tuple:
    return (
        assistant_type,
        _WS_RE.sub(" ", query).strip().casefold(),
        scenario_context or "",
        scenario_state or "",
        tuple(sorted((k, str(v)) for k, v in (lab or {}).items())),
        max_tokens or 0,
        answer_language,
    )
```

Add `assistant_type: str = DEFAULT_ASSISTANT_TYPE` to the end of
`generate_answer()` and `stream_answer()`. Resolve once at the start:

```python
profile = get_assistant_profile(assistant_type)
```

Use `profile.assistant_type` in the cache key and pass `profile` to
`_prepare_answer_grounding()` in both generation paths.

- [ ] **Step 6: Update LLM test doubles for the new collection keyword**

Update monkeypatched `_retrieve()`, `hybrid_search()`, and
`fetch_lab_instruction_record()` functions in `tests/test_llm_units.py`,
`tests/test_cache_units.py`, and `tests/test_corpus_units.py` to accept the new
optional keywords. Update monkeypatched `_lab_grounding()` functions to accept
`collection_name=None`. Keep the existing retrieval, language, citation, and
prompt assertions unchanged.

- [ ] **Step 7: Run LLM, cache, and corpus tests**

Run:

```bash
pytest tests/test_llm_units.py tests/test_cache_units.py tests/test_corpus_units.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit profile-aware generation**

```bash
git add app/services/llm.py tests/test_llm_units.py tests/test_cache_units.py tests/test_corpus_units.py
git commit -m "feat: route generation through assistant profiles"
```

### Task 4: Add Consumer API Selection And Conversation Isolation

**Files:**
- Modify: `app/services/memory.py`
- Modify: `app/api/routes.py`
- Modify: `tests/test_memory_units.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `get_assistant_profile()`, profile-aware `generate_answer()` and `stream_answer()`.
- Produces: optional `assistant_type` on `/ask`, `/v1/chat/completions`, `/voice_ask`, and conversation deletion.

- [ ] **Step 1: Write failing conversation namespace test**

Add to `tests/test_memory_units.py`:

```python
def test_conversation_memory_isolated_by_namespace():
    store = _store()
    history = store.history_for(
        "shared-id",
        "Teacher question",
        namespace="vr_lab_teacher",
    )
    store.remember(
        "shared-id",
        history,
        "Teacher answer",
        namespace="vr_lab_teacher",
    )

    domain_history = store.history_for(
        "shared-id",
        "Domain question",
        namespace="domain_assistant",
    )

    assert domain_history == [{"role": "user", "content": "Domain question"}]
```

- [ ] **Step 2: Write failing consumer API tests**

In `tests/test_api.py`, extend the `fake_answer` signature and captured call:

```python
async def _gen(
    query,
    scenario_context=None,
    chat_history=None,
    max_tokens=None,
    scenario_state=None,
    lab=None,
    answer_language=None,
    assistant_type="vr_lab_teacher",
):
    _gen.calls.append(
        {
            "query": query,
            "scenario_context": scenario_context,
            "chat_history": chat_history,
            "scenario_state": scenario_state,
            "lab": lab,
            "answer_language": answer_language,
            "assistant_type": assistant_type,
        }
    )
```

Add tests using a temporary profile:

```python
def test_ask_selects_assistant_profile(client, auth, fake_answer, monkeypatch):
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )

    response = client.post(
        "/ask",
        json={"query": "Question", "assistant_type": "domain_assistant"},
        headers=auth,
    )

    assert response.status_code == 200
    assert response.json()["assistant_type"] == "domain_assistant"
    assert fake_answer.calls[-1]["assistant_type"] == "domain_assistant"


def test_ask_rejects_unknown_assistant_before_generation(client, auth, fake_answer):
    response = client.post(
        "/ask",
        json={"query": "Question", "assistant_type": "missing"},
        headers=auth,
    )

    assert response.status_code == 422
    assert fake_answer.calls == []


def test_ask_conversation_history_isolated_by_assistant_profile(
    client,
    auth,
    fake_answer,
    monkeypatch,
):
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )
    conversation_id = "shared-profile-session"
    client.post(
        "/ask",
        json={
            "query": "Teacher question",
            "conversation_id": conversation_id,
        },
        headers=auth,
    )

    response = client.post(
        "/ask",
        json={
            "query": "Domain question",
            "conversation_id": conversation_id,
            "assistant_type": "domain_assistant",
        },
        headers=auth,
    )

    assert response.status_code == 200
    assert fake_answer.calls[-1]["chat_history"] == [
        {"role": "user", "content": "Domain question"}
    ]
```

Import `assistant_profiles` and `AssistantProfile` in `tests/test_api.py`.

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
pytest tests/test_memory_units.py::test_conversation_memory_isolated_by_namespace tests/test_api.py::test_ask_selects_assistant_profile tests/test_api.py::test_ask_rejects_unknown_assistant_before_generation -q
```

Expected: FAIL because memory and request models do not support profile selection.

- [ ] **Step 4: Namespace conversation memory**

In `app/services/memory.py`, add a private key helper:

```python
def _conversation_key(namespace: str, conversation_id: str) -> tuple[str, str]:
    return namespace, conversation_id
```

Change the public methods while preserving an empty namespace default for direct
legacy callers:

```python
def history_for(
    self,
    conversation_id: str,
    query: str,
    *,
    namespace: str = "",
) -> list[dict]:
    stored = self._cache.get(_conversation_key(namespace, conversation_id)) or []
    return trim_history(
        [*stored, {"role": "user", "content": query}],
        max_messages=self.max_messages,
        max_chars=self.max_chars,
    )


def remember(
    self,
    conversation_id: str,
    history: list[dict],
    answer: str,
    *,
    namespace: str = "",
) -> None:
    if not answer or not self.enabled:
        return
    updated = trim_history(
        [*history, {"role": "assistant", "content": answer}],
        max_messages=self.max_messages,
        max_chars=self.max_chars,
    )
    self._cache.put(_conversation_key(namespace, conversation_id), updated)


def clear(self, conversation_id: str, *, namespace: str = "") -> bool:
    return self._cache.delete(_conversation_key(namespace, conversation_id))
```

- [ ] **Step 5: Add one route-level profile resolver**

Import the profile API in `app/api/routes.py`:

```python
from app.services.assistant_profiles import (
    DEFAULT_ASSISTANT_TYPE,
    AssistantProfile,
    get_assistant_profile,
)
```

Add `Query` to the existing `fastapi` imports because conversation deletion
uses a query parameter.

Add:

```python
def _assistant_profile_or_422(assistant_type: str | None) -> AssistantProfile:
    try:
        return get_assistant_profile(assistant_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

Add this field to `AskRequest` and `ChatCompletionRequest`:

```python
assistant_type: str = Field(
    default=DEFAULT_ASSISTANT_TYPE,
    min_length=1,
    max_length=64,
    pattern=r"^[a-z][a-z0-9_]*$",
)
```

Add this multipart parameter to `voice_ask_endpoint()`:

```python
assistant_type: str = Form(DEFAULT_ASSISTANT_TYPE, min_length=1, max_length=64),
```

- [ ] **Step 6: Thread the profile through every consumer path**

At the start of `/ask`, `/v1/chat/completions`, and `/voice_ask`, resolve the
profile before loading memory or calling generation:

```python
profile = _assistant_profile_or_422(req.assistant_type)
```

For multipart voice use:

```python
profile = _assistant_profile_or_422(assistant_type)
```

Pass `assistant_type=profile.assistant_type` to every `generate_answer()` and
`stream_answer()` call.

Use the profile identifier as the memory namespace:

```python
chat_history = conversation_memory.history_for(
    conversation_id,
    req.query,
    namespace=profile.assistant_type,
)

conversation_memory.remember(
    conversation_id,
    chat_history,
    result.answer,
    namespace=profile.assistant_type,
)
```

Apply the same namespace to streaming text and voice memory writes.

- [ ] **Step 7: Echo the resolved type in responses and stream metadata**

Add `"assistant_type": profile.assistant_type` to:

- `/ask` JSON responses and streaming `done` events;
- `/v1/chat/completions` response metadata and streaming metadata frames;
- `/voice_ask` JSON responses, initial `question` events, and final `done` events.

Change conversation deletion to accept and validate the namespace:

```python
async def clear_conversation(
    conversation_id: ConversationId,
    request: Request,
    assistant_type: str = Query(DEFAULT_ASSISTANT_TYPE),
):
    profile = _assistant_profile_or_422(assistant_type)
    return {
        "conversation_id": conversation_id,
        "assistant_type": profile.assistant_type,
        "cleared": conversation_memory.clear(
            conversation_id,
            namespace=profile.assistant_type,
        ),
    }
```

Update the existing clear-conversation assertion in `tests/test_api.py` to
include the default `assistant_type`.

- [ ] **Step 8: Update stream test doubles and add voice coverage**

Every explicit `_stream()` or `_generate()` test double in `tests/test_api.py`
that receives route keywords must accept `assistant_type=None` or `**kwargs`.

Extend `test_voice_ask_full_pipeline` to send:

```python
"assistant_type": "domain_assistant"
```

after registering the temporary profile, then assert both the response and
`fake_answer.calls[-1]` contain `domain_assistant`.

- [ ] **Step 9: Run memory and consumer API tests**

Run:

```bash
pytest tests/test_memory_units.py tests/test_api.py -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit consumer profile selection**

```bash
git add app/services/memory.py app/api/routes.py tests/test_memory_units.py tests/test_api.py
git commit -m "feat: select assistant profiles per request"
```

### Task 5: Scope Admin Ingestion And Queued Work By Profile

**Files:**
- Modify: `app/api/admin_routes.py`
- Modify: `app/services/ingestion_jobs.py`
- Modify: `app/services/ingestion_worker.py`
- Modify: `tests/test_admin_auth_units.py`
- Modify: `tests/test_admin_ingestion_api_units.py`
- Modify: `tests/test_document_upload_api_units.py`
- Modify: `tests/test_ingestion_jobs_units.py`
- Modify: `tests/test_ingestion_worker_units.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: profile collection and corpus root, collection-aware ingestion functions.
- Produces: profile-scoped admin operations and persisted `assistant_type` job options.

- [ ] **Step 1: Write failing admin scoping tests**

Add a helper profile in `tests/test_admin_ingestion_api_units.py` and register it
with `monkeypatch.setitem()` as in Task 4.

Add these imports:

```python
from app.services import assistant_profiles
from app.services.assistant_profiles import AssistantProfile
```

Add:

```python
def test_corpus_preview_uses_profile_root_and_collection(
    client,
    admin_auth,
    monkeypatch,
):
    captured = {}
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain-corpus",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )

    def _scan(root, *, subtree="", only=None):
        captured["root"] = root
        return {
            "root": root,
            "subtree": subtree,
            "total": 0,
            "candidates": [],
            "skipped": [],
            "errors": [],
            "present_doc_ids": set(),
            "duplicate_lab_ids": [],
            "counts_by_type": {},
            "counts_by_language": {},
        }

    async def _documents(collection_name=None):
        captured["collection"] = collection_name
        return []

    monkeypatch.setattr(admin_routes.ingestion, "scan_corpus_tree", _scan)
    monkeypatch.setattr(admin_routes.ingestion, "list_documents", _documents)

    response = client.post(
        "/admin/ingestion/corpus/preview",
        headers=admin_auth,
        json={
            "assistant_type": "domain_assistant",
            "subtree": "",
            "ocr": False,
            "prune": False,
        },
    )

    assert response.status_code == 200
    assert captured == {"root": "/domain-corpus", "collection": "domain_kb"}
```

Add:

```python
def test_upload_job_persists_assistant_type(client, admin_auth, monkeypatch):
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain-corpus",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )
    manifest = [
        {"filename": "notes.md", "relative_path": "notes.md", "ocr": False}
    ]

    response = client.post(
        "/admin/ingestion/jobs/upload",
        headers=admin_auth,
        files=[("files", ("notes.md", b"notes", "text/markdown"))],
        data={
            "manifest": json.dumps(manifest),
            "assistant_type": "domain_assistant",
        },
    )

    assert response.status_code == 202
    assert response.json()["options"] == {
        "assistant_type": "domain_assistant"
    }


def test_admin_rejects_unknown_assistant_before_qdrant(
    client,
    admin_auth,
    monkeypatch,
):
    async def _unexpected_status(**kwargs):
        raise AssertionError("Qdrant status must not run")

    monkeypatch.setattr(
        admin_routes.ingestion,
        "corpus_status",
        _unexpected_status,
    )

    response = client.get(
        "/admin/corpus_status?assistant_type=missing",
        headers=admin_auth,
    )

    assert response.status_code == 422
```

- [ ] **Step 2: Write failing queue retry test**

Add to `tests/test_ingestion_jobs_units.py`:

```python
def test_upload_retry_preserves_assistant_type(job_dir):
    job_id = "profile-upload"
    item_id = "failed-item"
    stored_path = f"uploads/{job_id}/{item_id}"
    source = job_dir / stored_path
    source.parent.mkdir(parents=True)
    source.write_text("content", encoding="utf-8")
    job = ingestion_jobs.enqueue_upload_job(
        job_id,
        [
            {
                "id": item_id,
                "position": 0,
                "filename": "notes.md",
                "relative_path": "notes.md",
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        ],
        options={"assistant_type": "domain_assistant"},
    )
    ingestion_jobs.update_item(
        item_id,
        status="failed",
        stage="done",
        error="boom",
    )
    ingestion_jobs.finish_job(job["id"], status="failed")

    retried = ingestion_jobs.retry_job(job["id"])

    assert retried["options"] == {"assistant_type": "domain_assistant"}
```

- [ ] **Step 3: Run the focused admin and queue tests and verify they fail**

Run:

```bash
pytest tests/test_admin_ingestion_api_units.py::test_corpus_preview_uses_profile_root_and_collection tests/test_admin_ingestion_api_units.py::test_upload_job_persists_assistant_type tests/test_admin_ingestion_api_units.py::test_admin_rejects_unknown_assistant_before_qdrant tests/test_ingestion_jobs_units.py::test_upload_retry_preserves_assistant_type -q
```

Expected: FAIL because admin operations and upload jobs do not carry profiles.

- [ ] **Step 4: Persist options for upload jobs without a schema migration**

Change `enqueue_upload_job()` in `app/services/ingestion_jobs.py` to:

```python
def enqueue_upload_job(
    job_id: str,
    items: list[dict],
    *,
    options: dict | None = None,
    retry_of: str | None = None,
) -> dict:
    created_at = _now()
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO jobs (
                id, kind, status, options_json, retry_of, cancel_requested,
                total_items, completed_items, failed_items, skipped_items,
                current_item, current_stage, error, warning, created_at,
                started_at, finished_at
            ) VALUES (?, 'upload', 'queued', ?, ?, 0, ?, 0, 0, 0, NULL, NULL,
                      NULL, NULL, ?, NULL, NULL)
            """,
            (
                job_id,
                _json_dump(options or {}),
                retry_of,
                len(items),
                created_at,
            ),
        )
        _insert_items(connection, job_id, items)
        _refresh_job_counts(connection, job_id)
        return _get_job(connection, job_id)
```

In `retry_job()`, preserve options:

```python
return enqueue_upload_job(
    retry_id,
    retry_items,
    options=original["options"],
    retry_of=original["id"],
)
```

- [ ] **Step 5: Add profile validation to admin routes**

Import the profile API in `app/api/admin_routes.py` and add:

```python
def _assistant_profile_or_422(assistant_type: str | None) -> AssistantProfile:
    try:
        return get_assistant_profile(assistant_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
```

Add `assistant_type: str = DEFAULT_ASSISTANT_TYPE` to `CorpusJobRequest`.

Add profile parameters to direct admin operations:

```python
assistant_type: str = Query(DEFAULT_ASSISTANT_TYPE)
```

for status, list, and delete; and:

```python
assistant_type: str = Form(DEFAULT_ASSISTANT_TYPE)
```

for synchronous upload and upload-job creation.

Resolve the profile once in each endpoint and pass
`collection_name=profile.qdrant_collection` to `ingestion.corpus_status()`,
`upload_document()`, `list_documents()`, and `delete_document()`.

- [ ] **Step 6: Scope corpus previews and jobs**

In `preview_corpus()`:

```python
profile = _assistant_profile_or_422(request.assistant_type)
scan = ingestion.scan_corpus_tree(
    profile.corpus_root,
    subtree=subtree,
)
documents = await ingestion.list_documents(
    collection_name=profile.qdrant_collection,
)
```

In `create_corpus_job()` validate the selected root and persist the type:

```python
profile = _assistant_profile_or_422(request.assistant_type)
ingestion.resolve_corpus_scope(profile.corpus_root, subtree)
return ingestion_jobs.enqueue_corpus_job(
    {
        "assistant_type": profile.assistant_type,
        "subtree": subtree,
        "ocr": settings.OCR_ENABLED if request.ocr is None else request.ocr,
        "prune": request.prune,
    }
)
```

In `create_upload_job()` pass:

```python
return ingestion_jobs.enqueue_upload_job(
    job_id,
    validated,
    options={"assistant_type": profile.assistant_type},
)
```

- [ ] **Step 7: Resolve the profile inside the ingestion worker**

Import `AssistantProfile` and `get_assistant_profile` in
`app/services/ingestion_worker.py`.

Change `_item_file()` to enforce the selected corpus root:

```python
def _item_file(item: dict, profile: AssistantProfile) -> Path:
    if item.get("source_path"):
        root = Path(profile.corpus_root).resolve(strict=True)
        path = Path(item["source_path"]).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Corpus source file is missing or outside profile corpus root")
        return path
    root = ingestion_jobs.data_dir().resolve(strict=True)
    path = (root / item["stored_path"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Stored upload file is missing or outside ingestion data")
    return path
```

Pass `profile` through `_run_item()`, `_run_upload_job()`, and
`_run_corpus_job()`. Use:

```python
content = await asyncio.to_thread(_item_file(item, profile).read_bytes)
result = await ingestion.upload_document(
    item["filename"],
    content,
    metadata=item["metadata"],
    doc_key=item["doc_key"],
    ocr=item["ocr"],
    progress=tracked_progress,
    should_cancel=should_cancel,
    collection_name=profile.qdrant_collection,
)
```

Use `profile.corpus_root` for `scan_corpus_tree()` and pass
`profile.qdrant_collection` to pruning.

Resolve legacy jobs to the default profile inside the existing `try` block in
`run_once()`, so a removed profile becomes a normal failed job:

```python
try:
    profile = get_assistant_profile(job["options"].get("assistant_type"))
    if job["kind"] == "upload":
        mutated, ambiguous = await _run_upload_job(job, profile)
    else:
        mutated, ambiguous = await _run_corpus_job(job, profile)
except Exception as exc:
    error = str(exc)[:2000]
    failed = ingestion_jobs.get_job(job["id"])
    warning = failed.get("warning") if failed else None
    if failed and _needs_cache_invalidation(failed):
        if not await invalidate_answer_cache():
            warning = _cache_warning(warning)
    if failed:
        _fail_active_items(failed, error)
    ingestion_jobs.finish_job(
        job["id"],
        status="failed",
        error=error,
        warning=warning,
    )
    return True
```

This is the existing failure block with profile resolution moved under it.

- [ ] **Step 8: Update admin and worker test doubles**

Update explicit ingestion test doubles in `tests/test_api.py`,
`tests/test_admin_auth_units.py`,
`tests/test_document_upload_api_units.py`, and
`tests/test_admin_ingestion_api_units.py` to accept `collection_name=None`.

Update `_run_item`, `_run_upload_job`, `_run_corpus_job`, `upload_document`,
`scan_corpus_tree`, `prune_missing_corpus_documents`, and `_item_file` test
doubles in `tests/test_ingestion_worker_units.py` for the profile argument.
Add one worker assertion that a `domain_assistant` job passes `domain_kb` to
`upload_document()`.

Replace tests that monkeypatch `settings.CORPUS_ROOT` in
`tests/test_admin_ingestion_api_units.py` and
`tests/test_ingestion_worker_units.py` with a patched default profile:

```python
from dataclasses import replace

from app.services import assistant_profiles


default_profile = assistant_profiles.get_assistant_profile(None)
monkeypatch.setitem(
    assistant_profiles.ASSISTANT_PROFILES,
    default_profile.assistant_type,
    replace(default_profile, corpus_root=str(corpus)),
)
```

This keeps tests aligned with the new server-owned configuration boundary.

Update the existing `queue_upload()` helper in that test file to accept an
optional `assistant_type` and persist it through upload-job options:

```python
def queue_upload(
    worker_store: Path,
    job_id: str,
    filenames: list[str],
    *,
    assistant_type: str | None = None,
) -> dict:
    upload_dir = worker_store / "uploads" / job_id
    upload_dir.mkdir(parents=True)
    items = []
    for position, filename in enumerate(filenames):
        item_id = f"item-{position}"
        stored_path = f"uploads/{job_id}/{item_id}"
        (worker_store / stored_path).write_text(filename, encoding="utf-8")
        items.append(
            {
                "id": item_id,
                "position": position,
                "filename": filename,
                "relative_path": filename,
                "stored_path": stored_path,
                "source_path": None,
                "metadata": None,
                "doc_key": None,
                "ocr": False,
            }
        )
    options = (
        {"assistant_type": assistant_type}
        if assistant_type is not None
        else None
    )
    return ingestion_jobs.enqueue_upload_job(
        job_id,
        items,
        options=options,
    )
```

- [ ] **Step 9: Run admin, queue, and worker tests**

Run:

```bash
pytest tests/test_admin_auth_units.py tests/test_admin_ingestion_api_units.py tests/test_document_upload_api_units.py tests/test_ingestion_jobs_units.py tests/test_ingestion_worker_units.py tests/test_api.py -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit profile-scoped ingestion**

```bash
git add app/api/admin_routes.py app/services/ingestion_jobs.py app/services/ingestion_worker.py tests/test_admin_auth_units.py tests/test_admin_ingestion_api_units.py tests/test_document_upload_api_units.py tests/test_ingestion_jobs_units.py tests/test_ingestion_worker_units.py tests/test_api.py
git commit -m "feat: scope ingestion by assistant profile"
```

### Task 6: Add Profile-Aware Corpus CLI And Operator Documentation

**Files:**
- Modify: `scripts/manage_corpus.py`
- Modify: `tests/test_manage_corpus_cli_units.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/memory-backend-guide.md`

**Interfaces:**
- Consumes: profile registry, collection-aware ingestion functions.
- Produces: `--assistant-type` on corpus management commands and documented integration steps.

- [ ] **Step 1: Write failing CLI selection test**

Add to `tests/test_manage_corpus_cli_units.py`:

```python
from app.services import assistant_profiles
from app.services.assistant_profiles import AssistantProfile


def test_status_uses_selected_assistant_collection(monkeypatch, capsys):
    profile = AssistantProfile(
        "domain_assistant",
        "You are the domain assistant.",
        "domain_kb",
        "/domain-corpus",
    )
    monkeypatch.setitem(
        assistant_profiles.ASSISTANT_PROFILES,
        profile.assistant_type,
        profile,
    )
    seen = []

    async def _status(collection_name=None):
        seen.append(collection_name)
        return {"status": "ready", "collection": collection_name}

    monkeypatch.setattr(manage_corpus.ingestion, "corpus_status", _status)
    monkeypatch.setattr(
        sys,
        "argv",
        ["manage_corpus", "status", "--assistant-type", "domain_assistant"],
    )

    manage_corpus.main()

    assert seen == ["domain_kb"]
    assert "domain_kb" in capsys.readouterr().out
```

- [ ] **Step 2: Run the CLI test and verify it fails**

Run:

```bash
pytest tests/test_manage_corpus_cli_units.py::test_status_uses_selected_assistant_collection -q
```

Expected: FAIL because the CLI does not accept `--assistant-type`.

- [ ] **Step 3: Add assistant selection to each corpus command**

In `scripts/manage_corpus.py`, import:

```python
from app.services.assistant_profiles import (
    DEFAULT_ASSISTANT_TYPE,
    AssistantProfile,
    get_assistant_profile,
)
```

Add:

```python
def _add_assistant_type(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--assistant-type",
        default=DEFAULT_ASSISTANT_TYPE,
        help="Server-owned assistant profile whose Qdrant collection is managed",
    )


def _profile_or_parser_error(
    parser: argparse.ArgumentParser,
    assistant_type: str,
) -> AssistantProfile:
    try:
        return get_assistant_profile(assistant_type)
    except ValueError as exc:
        parser.error(str(exc))
```

Call `_add_assistant_type()` for `create-collection`, `upload`, `bulk-ingest`,
`gen-manifest`, `list`, `status`, and `delete` subparsers. Change the
`bulk-ingest` and `gen-manifest` root defaults to `None`; after parsing, use the
selected profile root when the positional root is omitted:

```python
profile = _profile_or_parser_error(parser, args.assistant_type)
root = args.root or profile.corpus_root
```

Pass `profile.qdrant_collection` to collection creation, upload, bulk ingest,
list, status, and delete service calls. `gen-manifest` uses only the resolved
root.

- [ ] **Step 4: Update CLI helper signatures**

Use these signatures and calls:

```python
async def _create_collection(profile: AssistantProfile) -> None:
    await vectorstore.ensure_collection(
        collection_name=profile.qdrant_collection,
    )


async def _status(profile: AssistantProfile) -> None:
    print(
        await ingestion.corpus_status(
            collection_name=profile.qdrant_collection,
        )
    )
```

Apply the same profile parameter to `_upload()`, `_list()`, `_delete()`, and
`_bulk_ingest()`. Pass `collection_name=profile.qdrant_collection` to the
underlying service operation.

- [ ] **Step 5: Update existing CLI test doubles**

Change `_bulk_ingest_tree()` test doubles in
`tests/test_manage_corpus_cli_units.py` to accept `collection_name=None` and
assert the default profile uses `settings.QDRANT_COLLECTION`. Keep all current
OCR, prune, filtering, and exit-code assertions.

- [ ] **Step 6: Document backend selection and profile setup**

Add a `Multi-assistant profiles` section to `README.md` containing:

```json
{
  "assistant_type": "domain_assistant",
  "query": "Explain the current step"
}
```

Document that:

- the backend sends the same identifier on every text and voice turn;
- prompts, collection names, and corpus roots remain server-owned;
- each profile requires a unique collection and a mounted corpus root;
- omitted `assistant_type` uses `vr_lab_teacher`;
- operators create, ingest, and inspect each profile with the CLI;
- a new profile requires the API and ingestion worker to run the same revision;
- the voice sidecars require no profile changes or rebuild.

Update `.env.example` comments so `QDRANT_COLLECTION` and `CORPUS_ROOT` are
described as defaults for `vr_lab_teacher`, not global per-request selectors.

Update `docs/memory-backend-guide.md` to require a stable `assistant_type` with
each reused `conversation_id`, and show profile-aware deletion:

```http
DELETE /v1/conversations/{conversation_id}?assistant_type=domain_assistant
Authorization: Bearer <INTERNAL_API_KEY>
```

- [ ] **Step 7: Run CLI tests and documentation checks**

Run:

```bash
pytest tests/test_manage_corpus_cli_units.py -q
git diff --check
```

Expected: CLI tests pass and the diff check exits with status 0.

- [ ] **Step 8: Commit CLI and documentation**

```bash
git add scripts/manage_corpus.py tests/test_manage_corpus_cli_units.py README.md .env.example docs/memory-backend-guide.md
git commit -m "docs: explain multi-assistant setup"
```

### Task 7: Run Full Verification

**Files:**
- Modify only files required to fix failures introduced by Tasks 1 through 6.

**Interfaces:**
- Consumes: the complete profile-aware API, retrieval, memory, and ingestion implementation.
- Produces: one verified branch ready for review.

- [ ] **Step 1: Run the full mocked test suite**

Run:

```bash
pytest
```

Expected: all tests pass with no network or GPU access.

- [ ] **Step 2: Compile the Python packages**

Run:

```bash
python -m compileall -q app scripts
```

Expected: exit status 0 with no syntax errors.

- [ ] **Step 3: Check formatting and forbidden placeholders**

Run:

```bash
git diff --check
! rg -n -P "T[B]D|T[O]DO|F[I]XME|Co-authored-by|\x{2014}" app scripts tests README.md .env.example docs/memory-backend-guide.md
```

Expected: `git diff --check` exits 0. The search returns no newly introduced
placeholder, coauthor trailer, or em dash.

- [ ] **Step 4: Review profile isolation manually**

Confirm from the final diff that:

1. every consumer generation call receives the resolved assistant type;
2. every Qdrant read and write in a profile-aware path receives the resolved collection;
3. every server-corpus read is confined to the resolved profile root;
4. answer cache and conversation memory include the assistant type;
5. no request can submit a raw system prompt or collection name;
6. default requests still resolve to `vr_lab_teacher`.

- [ ] **Step 5: Commit any verification-only fixes**

If verification required code or test corrections, commit only those corrections:

```bash
git add app scripts tests README.md .env.example docs/memory-backend-guide.md
git commit -m "test: verify assistant profile isolation"
```

If no corrections were needed, do not create an empty commit.
