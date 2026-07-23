# Multi-Assistant Quick Plan

## Goal

Run several assistant types in the same FastAPI service while sharing the
existing STT, TTS, embedder, scenario, citation, and OpenAI services. Each
assistant uses its own system prompt and Qdrant knowledge base.

## Configuration

Keep a server-owned profile registry:

```python
ASSISTANT_PROFILES = {
    "vr_lab_teacher": {
        "system_prompt": "You are a VR laboratory teaching assistant.",
        "qdrant_collection": "school_kb",
        "corpus_root": "/data/school-corpus",
    },
    "other_assistant": {
        "system_prompt": "You are the configured domain assistant.",
        "qdrant_collection": "other_assistant_kb",
        "corpus_root": "/data/other-corpus",
    },
}
```

The client sends only `assistant_type`. It must never send a raw prompt or
Qdrant collection name.

## API Changes

Add optional `assistant_type` to:

- `POST /ask` as a JSON field;
- `POST /v1/chat/completions` as a top-level JSON field;
- `POST /voice_ask` as a multipart form field.

Omitting it uses `vr_lab_teacher`. Unknown values return HTTP `422`.

Example:

```json
{
  "assistant_type": "other_assistant",
  "query": "Explain the current step",
  "scenario_id": "physics_lab_02_heating"
}
```

## Request Flow

```text
assistant_type
  -> resolve trusted profile
  -> transcribe audio when needed
  -> retrieve from profile.qdrant_collection
  -> combine profile prompt, Qdrant chunks, scenario, and live state
  -> generate the answer
  -> synthesize with the existing TTS service when needed
```

Do not change `settings.QDRANT_COLLECTION` during a request. Pass the selected
collection explicitly to vector-store functions so concurrent assistants cannot
query each other's KBs.

## Isolation

- Use one Qdrant collection per assistant.
- Include `assistant_type` in the answer-cache key.
- Namespace conversation memory by `(assistant_type, conversation_id)`.
- Fetch lab instructions from the selected collection.
- Keep the TTS cache unchanged because it already keys on text, language,
  backend, and voice.

## Corpus Management

Add `--assistant-type` to corpus commands and the same field to admin ingestion
requests. Resolve the profile server-side, then upload, list, delete, status,
and prune only inside that profile's collection.

Example:

```bash
python -m scripts.manage_corpus bulk-ingest /data/other-corpus \
  --assistant-type other_assistant
```

## Implementation Order

1. Add the profile registry and validation.
2. Add optional collection arguments to Qdrant and ingestion functions.
3. Pass `assistant_type` through LLM generation and consumer routes.
4. Isolate answer cache and conversation memory.
5. Add profile selection to corpus CLI and admin ingestion.
6. Add focused tests, then run `pytest`.

No voice-service changes or additional FastAPI deployment are required.
