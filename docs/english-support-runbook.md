# English support migration and operations runbook

English is supported end to end with canonical language code `en`. The existing
Russian default remains unchanged.

## Language semantics

The text generation precedence is:

1. Explicit request `language`
2. Detected language of the current user query
3. Detected language of the most recent unambiguous conversation turn
4. `lab.lang` when the query and recent conversation are ambiguous
5. `DEFAULT_LANGUAGE`, which defaults to `ru`

`language` and `lab.lang` have different meanings. `language` controls the answer.
`lab.lang` controls the preferred corpus language and participates in the exact
`lab_id`. A request may use `language=en` with `lab.lang=ru`; the assistant then
answers in English while grounding the exact procedure in
`physics-10-ru-02`, for example.

For `/voice_ask`, multipart `language` controls STT and accepts `auto`, `ru`,
`kk`, or `en`. Optional `response_language` controls the answer and TTS. Without
that override, both follow the resolved STT language. Multipart `lang` remains
the laboratory and corpus language.

Supported public and admin APIs:

- `/ask`: optional `language=ru|kk|en`
- `/v1/chat/completions`: optional VR extension `language=ru|kk|en`
- `/hint`: optional `language=ru|kk|en`, otherwise inferred from `hint_text`
- `/stt`: `language=auto|ru|kk|en`
- `/tts`: `language=en`, with Supertonic default and Qwen selectable
- `/voice_ask`: English STT, answer generation, retrieval, and TTS
- `/admin/documents`: structured `lang=en`
- `/admin/scenarios`: additive scenario language metadata
- `/health`, `/ready`, and `/admin/corpus_status`: additive language capabilities

## Corpus layout

Recommended English layouts:

```text
Corpus/
├── Laboratory works/
│   └── Physics/
│       └── Physics Grade 10/
│           └── en/
│               └── Lab work No. 2.docx
└── School materials/
    └── Biology/
        └── en/
            └── Biology Grade 9.epub
```

Recognized English tier aliases are `Laboratory works`, `Lab instructions`,
`School materials`, and `Textbooks`. Recognized subjects are `Physics`,
`Chemistry`, and `Biology`. Language tokens `en`, `eng`, and `english` map to
`en`. The example lab above produces `physics-10-en-02`.

English files use the existing BAAI/bge-m3 embeddings, 1024-dimensional dense
vectors, learned sparse vectors, Qdrant collection schema, and RRF retrieval.
No Qdrant collection recreation is required. Ingesting English documents only
adds or replaces points in the current collection.

Exact laboratory procedures remain language-specific. The service never silently
uses a Russian or Kazakh procedure for an `en` lab ID. If an English procedure is
missing, procedure questions retain theory-only safety behavior. To deliberately
use an existing Russian procedure with an English answer, send `language=en` and
`lab.lang=ru`.

## Ingest English documents

Set the corpus root and validate the offline manifest:

```bash
export CORPUS_ROOT=/absolute/path/to/Corpus
python -m scripts.manage_corpus gen-manifest "$CORPUS_ROOT" --out labs.json
```

The manifest includes `textbooks_by_language` and `labs_by_language`. Inspect
`missing_metadata` and English `stub` entries before ingestion.

Start the existing retrieval services and ingest:

```bash
docker compose up -d qdrant embedder
python -m scripts.manage_corpus create-collection
python -m scripts.manage_corpus bulk-ingest "$CORPUS_ROOT" --only '/en/'
```

`create-collection` is safe when the collection already exists. It does not
recreate or resize the collection.

For scanned English PDF or EPUB files, enable OCR:

```bash
python -m scripts.manage_corpus bulk-ingest "$CORPUS_ROOT" --ocr --only '/en/'
```

The API image explicitly installs Tesseract `eng`, `rus`, and `kaz` data. Host
CLI ingestion needs equivalent local Tesseract packages.

For one structured upload:

```bash
curl -sS "$API_BASE/admin/documents" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@Lab work No. 2.docx' \
  -F 'doc_type=lab_instruction' \
  -F 'subject=physics' \
  -F 'grade=10' \
  -F 'lang=en' \
  -F 'lab_number=2'
```

## Voice models and cold start

English Whisper processing uses the same `openai/whisper-large-v3-turbo` base
model already loaded for Russian and Kazakh. Only Kazakh uses the LoRA adapter.
Russian and English disable that adapter and select their own Whisper language
token. No second full Whisper checkpoint is loaded.

English TTS reuses the same Supertonic and Qwen3-TTS instances as Russian.
Supertonic is the default and Qwen is selectable. English is never passed through
Russian Latin-to-Cyrillic transliteration. OmniVoice is Kazakh-only.

The tested Supertonic package is pinned to `supertonic==1.3.1`, which officially
supports both `en` and `ru`. The first voice container start downloads the shared
Whisper, Qwen, and Supertonic assets and may be slow. Adding English does not add
another large Whisper, Qwen, or Supertonic model instance to GPU memory. Existing
cold-start and shared-GPU planning still applies. Supertonic assets are stored at
`TTS_SUPERTONIC_MODEL_DIR` inside the persistent `voice_hf_cache` volume, so voice
container recreation does not download them again.

Configuration:

```dotenv
DEFAULT_LANGUAGE=ru
VOICE_TTS_EN_DEFAULT_BACKEND=supertonic
TTS_EN_BACKEND=supertonic
TTS_SUPERTONIC_MODEL_DIR=/models/hf_cache/supertonic3
TTS_NORMALIZE_EN_NUMBERS=true
```

## Health checks

```bash
curl -sS "$API_BASE/health"
curl -sS "$API_BASE/ready"
curl -sS "$API_BASE/admin/corpus_status" \
  -H "Authorization: Bearer $KEY"
curl -sS http://localhost:8002/health
```

Expected additive fields include `supported_languages` with `en`, corpus
`documents_by_language.en`, voice `stt_models` containing `en`, and English
`tts_backends` plus `tts_default_backends.en=supertonic`.

## Manual API verification

Set common values first:

```bash
export API_BASE=http://localhost:8001
export KEY='replace-with-INTERNAL_API_KEY'
```

English `/ask`:

```bash
curl -sS "$API_BASE/ask" \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What is the boiling point of pure water?",
    "language": "en",
    "lab": {"subject": "physics", "grade": 10, "lang": "en", "lab_number": 2}
  }'
```

English OpenAI-compatible chat, non-streaming:

```bash
curl -sS "$API_BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4.1-mini",
    "language": "en",
    "messages": [{"role": "user", "content": "Explain why water boils."}]
  }'
```

Streaming chat:

```bash
curl -N "$API_BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "stream": true,
    "language": "en",
    "messages": [{"role": "user", "content": "What should I do next?"}]
  }'
```

English hint:

```bash
curl -sS "$API_BASE/hint" \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "hint_text": "Wear the safety glasses before lighting the burner.",
    "hint_level": 2,
    "language": "en"
  }'
```

Explicit and automatic English STT:

```bash
curl -sS "$API_BASE/stt" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@question_en.wav' \
  -F 'language=en'

curl -sS "$API_BASE/stt" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@question_en.wav' \
  -F 'language=auto'
```

English Supertonic and Qwen TTS:

```bash
curl -sS "$API_BASE/tts" \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Heat 250 mL to 25 °C.","language":"en"}' \
  -D supertonic.headers -o supertonic_en.wav

curl -sS "$API_BASE/tts" \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Heat 250 mL to 25 °C.","language":"en","backend":"qwen"}' \
  -D qwen.headers -o qwen_en.wav
```

English `/voice_ask`, non-streaming and streaming:

```bash
curl -sS "$API_BASE/voice_ask" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@question_en.wav' \
  -F 'language=auto' \
  -F 'subject=physics' \
  -F 'grade=10' \
  -F 'lang=en' \
  -F 'lab_number=2'

curl -N "$API_BASE/voice_ask" \
  -H "Authorization: Bearer $KEY" \
  -F 'file=@question_en.wav' \
  -F 'language=en' \
  -F 'response_language=en' \
  -F 'stream=true'
```

Corpus visibility:

```bash
curl -sS "$API_BASE/admin/documents" \
  -H "Authorization: Bearer $KEY" | python -m json.tool

curl -sS "$API_BASE/admin/corpus_status" \
  -H "Authorization: Bearer $KEY" | python -m json.tool
```

## Production prerequisite

The repository contains only a small English scenario and hermetic test fixtures.
It does not fabricate or ship authoritative English textbooks or laboratory
instructions. Production owners must supply licensed, authoritative English
source files, place them in a recognized layout or upload them with structured
metadata, ingest them, and verify the English manifest and corpus counts before
claiming complete English grounding coverage.
