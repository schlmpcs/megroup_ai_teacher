# Multi-Assistant Profiles With Isolated Knowledge Bases

**Date:** 2026-07-23

## Summary

One FastAPI deployment will support multiple predefined assistant types. The
calling backend selects an assistant by sending `assistant_type` with each
request. Each assistant profile owns:

- its assistant-specific system instruction;
- its own Qdrant collection;
- its trusted server-side corpus root;
- the same shared retrieval, scenario, citation, language, STT, and TTS logic.

The consumer API must never accept a raw system prompt or Qdrant collection
name. It accepts only a validated profile identifier. The server resolves that
identifier to trusted configuration.

## Goals

1. Let the integration backend choose a predefined assistant per request.
2. Keep every assistant's documents and retrieval results isolated.
3. Reuse the existing bge-m3 embedder, Qdrant server, OpenAI generation, local
   scenarios, citations, conversation memory, STT, and TTS services.
4. Preserve current clients by defaulting omitted `assistant_type` to
   `vr_lab_teacher`.
5. Document how maintainers add a profile and ingest its corpus.

## Non-Goals

- Accepting arbitrary prompts from consumer requests.
- Accepting arbitrary Qdrant collection names from consumer requests.
- Running a separate FastAPI or voice deployment per assistant.
- Giving each assistant different embedding or speech models.
- Adding per-assistant scenario directories before a real need exists.
- Changing the standalone `/stt` or `/tts` contracts.

## Profile Registry

Add one small server-owned registry, for example in
`app/services/assistant_profiles.py`:

```python
DEFAULT_ASSISTANT_TYPE = "vr_lab_teacher"

ASSISTANT_PROFILES = {
    "vr_lab_teacher": {
        "system_prompt": VR_LAB_TEACHER_PROMPT,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "corpus_root": settings.CORPUS_ROOT,
    },
    "domain_assistant": {
        "system_prompt": DOMAIN_ASSISTANT_PROMPT,
        "qdrant_collection": "domain_assistant_kb",
        "corpus_root": "/data/domain-assistant-corpus",
    },
}
```

`domain_assistant` is an example identifier. A production profile uses the
identifier, instruction, collection name, and trusted corpus root selected for
that assistant.

Validate the registry at startup:

- `DEFAULT_ASSISTANT_TYPE` exists;
- identifiers use lowercase letters, digits, and underscores;
- prompts are non-empty;
- collection names are non-empty;
- corpus roots are non-empty;
- no two profiles use the same collection.

Collection uniqueness enforces the requirement that assistants have separate
knowledge bases.

## Prompt Composition

Do not replace the complete grounding prompt with an unchecked profile string.
Build the final system prompt from:

```text
assistant profile instruction
+ shared evidence, language, safety, and citation rules
+ selected lab instruction from the profile's collection
+ retrieved chunks from the profile's collection
+ shared static scenario
+ shared live scenario state
```

This lets profiles have different roles and behavior while retaining the rules
that make Qdrant evidence and simulator state authoritative. The existing VR
teacher text becomes the `vr_lab_teacher` profile instruction.

## Consumer API

Add `assistant_type` to the answer-producing endpoints:

- `/ask`: optional JSON field;
- `/v1/chat/completions`: optional top-level JSON field;
- `/voice_ask`: optional multipart form field.

Omission resolves to `vr_lab_teacher`. An unknown identifier returns HTTP `422`
before retrieval or generation starts.

### Text Request

```http
POST /ask
Authorization: Bearer <INTERNAL_API_KEY>
Content-Type: application/json

{
  "assistant_type": "domain_assistant",
  "query": "Explain the current step",
  "scenario_id": "physics_lab_02_heating",
  "lab": {
    "subject": "physics",
    "grade": 10,
    "lang": "en",
    "lab_number": 2
  }
}
```

### Voice Request

```bash
curl -X POST "$API_BASE/voice_ask" \
  -H "Authorization: Bearer $INTERNAL_API_KEY" \
  -F "file=@question.webm" \
  -F "assistant_type=domain_assistant" \
  -F "language=auto" \
  -F "scenario_id=physics_lab_02_heating" \
  -F "subject=physics" \
  -F "grade=10" \
  -F "lang=en" \
  -F "lab_number=2"
```

The API response and streaming `done` event should echo the resolved
`assistant_type`. The first `/voice_ask` streaming `question` event should also
include it so the client can confirm the active profile before playback.

`/stt` and `/tts` remain profile-independent. The `/voice_ask` route continues
to call the same `transcribe_with_language(...)` and `synthesize(...)` functions.
`/hint` remains the existing VR-specific operation until another profile has a
real requirement for profile-specific hint behavior.

## Runtime Data Flow

```text
request assistant_type
  -> resolve trusted assistant profile
  -> load shared scenario context and live state
  -> embed the question with the shared bge-m3 service
  -> search profile.qdrant_collection
  -> fetch lab instruction from profile.qdrant_collection
  -> compose profile instruction plus shared grounding blocks
  -> generate with the shared OpenAI client
  -> synthesize with the shared TTS service when requested
```

The collection name must be passed as an explicit argument through the LLM,
retrieval, lab-instruction, vector-store, ingestion, and admin paths. Never
temporarily mutate `settings.QDRANT_COLLECTION`; concurrent requests for two
assistant types would race and could query the wrong corpus.

The existing `QDRANT_COLLECTION=school_kb` setting remains the default
collection for backward compatibility. The profile registry is authoritative
for multi-assistant requests.

## Qdrant Isolation

Use one collection per assistant rather than one collection with an
`assistant_type` payload filter. Separate collections prevent:

- cross-assistant retrieval when a filter is missed;
- document ID collisions between corpora with matching paths;
- one assistant's prune or delete operation affecting another assistant;
- same-`lab_id` instructions from becoming ambiguous.

All collections keep the existing schema: named `dense` and `sparse` vectors,
bge-m3 dimension settings, cosine distance for dense vectors, and RRF fusion.
The Qdrant server and embedder service remain shared.

Vector-store operations that currently use `settings.QDRANT_COLLECTION` need a
collection argument, including collection creation, upsert, hybrid search, lab
instruction lookup, document listing, deletion, and status reporting. Existing
callers may default to the legacy collection, but every profile-aware path must
pass the resolved collection explicitly.

## Cache And Conversation Isolation

Add the resolved `assistant_type` to the answer-cache key. Otherwise the same
question and scenario could return an answer generated with another profile and
another knowledge base.

Namespace server-side conversation memory by both `assistant_type` and
`conversation_id`. Reusing the same external conversation ID across profiles
must not mix their histories. Conversation deletion should accept an optional
`assistant_type`, defaulting to `vr_lab_teacher` for existing clients.

The TTS cache needs no profile field. It is already keyed by output text,
language, backend, and voice, so identical spoken output can be reused safely.

## Ingestion And Administration

Every corpus operation must resolve an assistant profile before touching
Qdrant. Add `assistant_type` to:

- CLI collection creation, upload, bulk ingest, list, status, and delete;
- synchronous admin upload, list, status, and delete endpoints;
- upload and server-corpus ingestion jobs;
- admin previews that calculate existing or prunable documents.

Queued jobs store the validated `assistant_type`. The worker resolves that
profile and writes only to its collection. If a profile is removed while jobs
are queued, those jobs fail clearly instead of falling back to the default KB.
Server-corpus preview, ingest, and prune operations use the profile's trusted
`corpus_root`; callers do not submit arbitrary server filesystem paths.

Example corpus setup:

```bash
python -m scripts.manage_corpus create-collection \
  --assistant-type domain_assistant

python -m scripts.manage_corpus bulk-ingest /data/domain-corpus \
  --assistant-type domain_assistant

python -m scripts.manage_corpus status \
  --assistant-type domain_assistant
```

Omitting `--assistant-type` targets `vr_lab_teacher` for backward
compatibility. Destructive operations such as prune and delete must always act
on only the resolved profile collection.

## Adding An Assistant

1. Choose a stable lowercase identifier such as `domain_assistant`.
2. Write its assistant-specific instruction. Keep evidence, language, and
   citation enforcement in the shared prompt portion.
3. Choose a unique Qdrant collection name.
4. Choose and mount a trusted corpus root containing only that assistant's
   source documents.
5. Add the profile to the server registry and deploy the API and ingestion
   worker from the same revision.
6. Create the profile collection and ingest that assistant's corpus.
7. Confirm profile-scoped status shows the expected documents.
8. Send a test `/ask` request with the profile identifier and verify returned
   citations belong only to that corpus.
9. Configure the integration backend to send the same `assistant_type` on every
   text and voice turn in that conversation.

The integration backend chooses only the identifier. It does not send the
prompt or collection name.

## Error Handling

- Unknown `assistant_type`: HTTP `422` with the allowed identifiers.
- Missing or empty selected collection: report the profile KB as unconfigured;
  do not silently search the default collection.
- Missing profile corpus root during server-side ingestion: reject the job
  before scanning or writing documents.
- Qdrant, embedder, OpenAI, and voice failures continue through the existing
  `LLMError` mapping.
- A queued ingestion job whose profile no longer exists fails with a clear
  profile-configuration error.

## Testing

Add focused tests proving:

1. Omitted `assistant_type` preserves `vr_lab_teacher` behavior.
2. Unknown profile identifiers return `422` without Qdrant or OpenAI calls.
3. Each profile passes its own collection to theory retrieval and lab lookup.
4. Two profiles with the same query cannot share answer-cache entries.
5. Two profiles with the same conversation ID cannot share history.
6. `/voice_ask` forwards the profile while using the existing STT and TTS
   functions.
7. Ingestion, list, status, delete, and prune stay inside the selected
   collection.
8. Registry validation rejects duplicate collection names.

Run the full mocked suite with `pytest` before deployment. Then ingest a small
distinct document into each test collection and verify each profile cites only
its own document.

## Rollout

1. Deploy profile support with only `vr_lab_teacher` configured and verify all
   existing clients still work without `assistant_type`.
2. Add the next profile and create its empty collection.
3. Ingest and validate its corpus before routing consumer traffic to it.
4. Update the integration backend to send `assistant_type` consistently.
5. Monitor per-profile retrieval failures and citation results during rollout.

No voice-service rebuild is required. Both assistants continue using the same
running STT and TTS sidecars.
